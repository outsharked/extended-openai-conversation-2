"""Strategy orchestration for conversation compaction and retrieval."""

from __future__ import annotations

from collections.abc import Mapping
import json
import logging
from typing import Any

from openai import AsyncClient

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from ..const import (
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_CONTEXT_COMPACTION_INTRA_TURN,
    CONF_CONTEXT_COMPACTION_PROMPT,
    CONF_CONTEXT_COMPACTION_RECENT_TURNS,
    CONF_CONTEXT_COMPACTION_TARGET,
    CONF_CONTEXT_THRESHOLD,
    CONF_CONTEXT_TRUNCATE_STRATEGY,
    CONF_EMBEDDING_CHUNK_SIZE,
    CONF_EMBEDDING_HISTORY_RETENTION,
    CONF_EMBEDDINGS,
    CONF_EMBEDDINGS_MODEL,
    CONF_MAX_TOKENS,
    CONTEXT_TRUNCATE_RETRIEVAL_STRATEGIES,
    CONTEXT_TRUNCATE_STRATEGY_RETRIEVE,
    CONTEXT_TRUNCATE_STRATEGY_RETRIEVE_COMPACT,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CONTEXT_COMPACTION_INTRA_TURN,
    DEFAULT_CONTEXT_COMPACTION_PROMPT,
    DEFAULT_CONTEXT_COMPACTION_RECENT_TURNS,
    DEFAULT_CONTEXT_COMPACTION_SUMMARY_PREFIX,
    DEFAULT_CONTEXT_COMPACTION_TARGET,
    DEFAULT_CONTEXT_COMPACTION_TOKEN_LIMIT,
    DEFAULT_CONTEXT_THRESHOLD,
    DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
    DEFAULT_EMBEDDING_CHUNK_SIZE,
    DEFAULT_EMBEDDING_HISTORY_RETENTION,
    DEFAULT_MAX_TOKENS,
    DOMAIN,
)
from ..helpers import (
    async_create_chat_completion,
    async_get_context_compaction_text,
    get_model_config,
)
from .compaction import (
    _INTRA_TURN_SUMMARY_NATIVE_KEY,
    _ROLLING_SUMMARY_NATIVE_KEY,
    CompactionPlan,
    RequestTokenUsage,
    _compaction_transcript,
    _content_for_compaction,
    _estimate_content_tokens,
    _estimate_serialized_tokens,
    _group_compaction_contents,
    _intra_turn_reduction,
    _is_generated_compaction_summary,
)
from .embedding_memory import (
    EMBEDDING_MEMORY_SOURCE_CONVERSATION,
    EMBEDDING_MEMORY_SOURCE_INTRA_TURN,
    EmbeddingMemoryChunk,
    EmbeddingMemoryRecord,
    EmbeddingMemoryScope,
    EmbeddingMemoryStore,
    async_embed_text_chunks,
    embedding_history_retention,
)

_LOGGER = logging.getLogger(__name__)


class ContextManagementMixin:
    """Provide context reduction strategies to an OpenAI entity."""

    hass: HomeAssistant
    entry: Any
    subentry: ConfigSubentry
    _unsupported_params: dict[str, set[str]]

    @property
    def _client(self) -> AsyncClient:
        """Return the configured chat client."""
        raise NotImplementedError

    @property
    def _embedding_runtime(self) -> tuple[AsyncClient, EmbeddingMemoryStore]:
        """Return the configured embedding client and memory store."""
        raise NotImplementedError

    def _clear_message_history(self, chat_log: conversation.ChatLog) -> bool:
        """Keep only the system prompt and the latest conversation turn."""
        original_length = len(chat_log.content)
        last_user_message_index = None
        messages = chat_log.content
        for index in reversed(range(len(messages))):
            if isinstance(messages[index], conversation.UserContent):
                last_user_message_index = index
                break

        if last_user_message_index is not None:
            del messages[1:last_user_message_index]
        return len(messages) != original_length

    def _compaction_history_split_index(
        self,
        messages: list[conversation.Content],
        preserve_current_turn: bool,
        recent_turns_override: int | None = None,
    ) -> int | None:
        """Return where the verbatim tail begins for compaction."""
        user_message_indexes = [
            index
            for index, content in enumerate(messages)
            if isinstance(content, conversation.UserContent)
        ]
        recent_turns = (
            max(0, recent_turns_override)
            if recent_turns_override is not None
            else max(
                0,
                int(
                    self.subentry.data.get(
                        CONF_CONTEXT_COMPACTION_RECENT_TURNS,
                        DEFAULT_CONTEXT_COMPACTION_RECENT_TURNS,
                    )
                ),
            )
        )
        turns_to_keep = recent_turns + int(preserve_current_turn)
        if not preserve_current_turn:
            # Never replace the turn that produced the response currently being
            # returned, even when the configured tail is zero.
            turns_to_keep = max(1, turns_to_keep)
        if len(user_message_indexes) <= turns_to_keep:
            return None
        return user_message_indexes[-turns_to_keep] if turns_to_keep else len(messages)

    def _compaction_max_tokens(
        self,
        chat_log: conversation.ChatLog,
        retained_contents: list[conversation.Content],
        prompt_tokens: int
    ) -> int:
        """Return the available summary budget for the configured target."""
        options = self.subentry.data
        target_tokens = max(
            1,
            int(
                options.get(
                    CONF_CONTEXT_COMPACTION_TARGET,
                    DEFAULT_CONTEXT_COMPACTION_TARGET,
                )
            ),
        )

        estimated_history_tokens = _estimate_content_tokens(chat_log.content)
        non_history_tokens = max(0, prompt_tokens - estimated_history_tokens)
        retained_tokens = _estimate_content_tokens(retained_contents)
        available_tokens = target_tokens - non_history_tokens - retained_tokens

        max_output_tokens = int(options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS))
        return min(available_tokens, max_output_tokens)

    def _compaction_plan(
        self,
        chat_log: conversation.ChatLog,
        prompt_tokens: int,
        preserve_current_turn: bool,
        recent_turns_override: int | None,
    ) -> CompactionPlan | None:
        """Find the largest verbatim tail that leaves a usable summary budget."""
        requested_recent_turns = max(
            0,
            (
                recent_turns_override
                if recent_turns_override is not None
                else int(
                    self.subentry.data.get(
                        CONF_CONTEXT_COMPACTION_RECENT_TURNS,
                        DEFAULT_CONTEXT_COMPACTION_RECENT_TURNS,
                    )
                )
            ),
        )
        messages = chat_log.content
        for recent_turns in range(requested_recent_turns, -1, -1):
            history_split_index = self._compaction_history_split_index(
                messages,
                preserve_current_turn,
                recent_turns,
            )
            if history_split_index is None or history_split_index <= 1:
                continue

            retained_contents = [messages[0], *messages[history_split_index:]]
            max_summary_tokens = self._compaction_max_tokens(
                chat_log,
                retained_contents,
                prompt_tokens,
            )
            if max_summary_tokens <= 1:
                continue

            if recent_turns < requested_recent_turns:
                _LOGGER.info(
                    "Compaction target requires reducing the verbatim tail "
                    "from %s to %s recent turns",
                    requested_recent_turns,
                    recent_turns,
                )
            return CompactionPlan(
                history_split_index=history_split_index,
                recent_turns=recent_turns,
                max_summary_tokens=max_summary_tokens,
            )
        return None

    async def _summarize_compaction_groups(
        self,
        groups: list[list[conversation.Content]],
        max_summary_tokens: int,
        threshold: int,
        language: str | None,
    ) -> tuple[str, str]:
        """Summarize bounded complete conversation or tool-exchange groups."""
        prompt = str(
            self.subentry.data.get(
                CONF_CONTEXT_COMPACTION_PROMPT,
                DEFAULT_CONTEXT_COMPACTION_PROMPT,
            )
        ).strip()
        if not prompt:
            prompt = DEFAULT_CONTEXT_COMPACTION_PROMPT
        summary_prefix = DEFAULT_CONTEXT_COMPACTION_SUMMARY_PREFIX
        token_limit = DEFAULT_CONTEXT_COMPACTION_TOKEN_LIMIT.format(
            max_summary_tokens=max_summary_tokens
        )
        if language is not None:
            summary_prefix, token_limit = await async_get_context_compaction_text(
                self.hass, language, max_summary_tokens
            )
        system_prompt = f"{prompt}\n{token_limit}"
        model = self.subentry.data.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        model_config = get_model_config(model, self.entry.data.get(CONF_BASE_URL))
        summary = ""
        while groups:
            selected_groups: list[list[conversation.Content]] = []
            for group in groups:
                candidate_groups = [*selected_groups, group]
                candidate_payload = json.dumps(
                    {
                        "existing_summary": summary or None,
                        "transcript": [
                            _content_for_compaction(content)
                            for candidate_group in candidate_groups
                            for content in candidate_group
                        ],
                    },
                    ensure_ascii=False,
                    default=str,
                )
                candidate_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": candidate_payload},
                ]
                if (
                    _estimate_serialized_tokens(candidate_messages)
                    + max_summary_tokens
                    > threshold
                ):
                    break
                selected_groups = candidate_groups

            if not selected_groups:
                raise HomeAssistantError("A complete conversation or tool exchange is too large to compact safely")

            payload = json.dumps(
                {
                    "existing_summary": summary or None,
                    "transcript": [
                        _content_for_compaction(content)
                        for group in selected_groups
                        for content in group
                    ],
                },
                ensure_ascii=False,
                default=str,
            )
            request: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": payload},
                ],
                "stream": False,
            }
            if model_config["supports_max_completion_tokens"]:
                request["max_completion_tokens"] = max_summary_tokens
            elif model_config["supports_max_tokens"]:
                request["max_tokens"] = max_summary_tokens

            response = await async_create_chat_completion(
                self._client,
                self._unsupported_params.setdefault(model, set()),
                **request,
            )
            result = response.choices[0].message.content
            if not isinstance(result, str) or not result.strip():
                raise HomeAssistantError("The conversation compaction returned no summary")
            summary = result.strip()
            del groups[: len(selected_groups)]

        return summary_prefix, summary

    async def _compact_message_history(
        self,
        chat_log: conversation.ChatLog,
        token_usage: RequestTokenUsage,
        preserve_current_turn: bool = False,
        recent_turns_override: int | None = None,
        language: str | None = None,
    ) -> bool:
        """Replace older completed turns with a rolling conversation summary."""
        messages = chat_log.content
        plan = self._compaction_plan(
            chat_log,
            token_usage.prompt_tokens,
            preserve_current_turn,
            recent_turns_override,
        )
        if plan is None:
            _LOGGER.info("Context threshold exceeded, but no older history can be compacted within the configured target")
            return False

        history_split_index = plan.history_split_index
        history_to_summarize = messages[1:history_split_index]
        threshold = int(self.subentry.data.get(CONF_CONTEXT_THRESHOLD, DEFAULT_CONTEXT_THRESHOLD))
        summary_prefix, summary = await self._summarize_compaction_groups(
            _group_compaction_contents(history_to_summarize),
            plan.max_summary_tokens,
            threshold,
            language,
        )

        agent_id = next(
            (
                content.agent_id
                for content in reversed(messages)
                if isinstance(content, conversation.AssistantContent)
            ),
            getattr(self, "entity_id", None) or DOMAIN,
        )
        messages[1:history_split_index] = [
            conversation.AssistantContent(
                agent_id=agent_id,
                content=f"{summary_prefix}\n{summary}",
                native={_ROLLING_SUMMARY_NATIVE_KEY: True},
            )
        ]
        _LOGGER.info(
            "Context threshold exceeded, older conversation history compacted"
        )
        return True

    async def _archive_embedding_groups(
        self,
        chat_log: conversation.ChatLog,
        groups: list[list[conversation.Content]],
        *,
        source: str,
        turn_id: str | None = None,
    ) -> None:
        """Embed complete history or intra-turn groups in the conversation store."""
        conversation_id = chat_log.conversation_id
        if not isinstance(conversation_id, str) or not conversation_id:
            raise HomeAssistantError(
                "Conversation history cannot be archived without a conversation ID"
            )

        embeddings = self.entry.data.get(CONF_EMBEDDINGS)
        if not isinstance(embeddings, Mapping):
            raise HomeAssistantError("Embedding retrieval is not configured")
        model = embeddings.get(CONF_EMBEDDINGS_MODEL)
        if not isinstance(model, str) or not model:
            raise HomeAssistantError("An embedding model is not configured")
        chunk_size = embeddings.get(
            CONF_EMBEDDING_CHUNK_SIZE, DEFAULT_EMBEDDING_CHUNK_SIZE
        )
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
            raise HomeAssistantError("The embedding chunk size is invalid")

        embedding_client, memory_store = self._embedding_runtime
        embedded_groups = [
            await async_embed_text_chunks(
                embedding_client,
                model,
                _compaction_transcript(group),
                chunk_size,
            )
            for group in groups
        ]

        retention = embedding_history_retention(
            self.subentry.data.get(
                CONF_EMBEDDING_HISTORY_RETENTION,
                DEFAULT_EMBEDDING_HISTORY_RETENTION,
            )
        )
        records: list[EmbeddingMemoryRecord] = []
        for group, embedded_chunks in zip(groups, embedded_groups, strict=True):
            records.append(
                EmbeddingMemoryRecord.create(
                    model=model,
                    embedding=embedded_chunks[0].embedding,
                    searchable_text=embedded_chunks[0].text,
                    messages=[content.as_dict() for content in group],
                    retention=retention,
                    source=source,
                    turn_id=turn_id,
                    additional_chunks=(
                        EmbeddingMemoryChunk(chunk.embedding, chunk.text)
                        for chunk in embedded_chunks[1:]
                    ),
                )
            )
        await memory_store.async_append(
            EmbeddingMemoryScope(
                entry_id=self.entry.entry_id,
                agent_id=self.subentry.subentry_id,
                conversation_id=conversation_id,
            ),
            records,
        )

    async def _reduce_intra_turn_history(
        self,
        chat_log: conversation.ChatLog,
        token_usage: RequestTokenUsage,
        language: str | None = None,
    ) -> bool:
        """Reduce older completed tool exchanges from the active turn."""
        options = self.subentry.data
        if not options.get(
            CONF_CONTEXT_COMPACTION_INTRA_TURN,
            DEFAULT_CONTEXT_COMPACTION_INTRA_TURN,
        ):
            return False

        strategy = options.get(
            CONF_CONTEXT_TRUNCATE_STRATEGY,
            DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
        )
        if strategy == "clear":
            return False

        reduction = _intra_turn_reduction(chat_log.content)
        if reduction is None:
            return False

        archived = False
        if strategy in CONTEXT_TRUNCATE_RETRIEVAL_STRATEGIES:
            try:
                await self._archive_embedding_groups(
                    chat_log,
                    [list(group) for group in reduction.archive_groups],
                    source=EMBEDDING_MEMORY_SOURCE_INTRA_TURN,
                    turn_id=reduction.turn_id,
                )
                archived = True
            except Exception:
                _LOGGER.exception("Intra-turn embedding archival failed. Continuing with conversation compaction")

        replacement: str | None = None
        should_compact = strategy in (
            "compact",
            CONTEXT_TRUNCATE_STRATEGY_RETRIEVE_COMPACT,
        ) or (strategy == CONTEXT_TRUNCATE_STRATEGY_RETRIEVE and not archived)
        if should_compact:
            try:
                retained_contents = [
                    content
                    for index, content in enumerate(chat_log.content)
                    if index not in reduction.remove_indexes
                ]
                max_summary_tokens = self._compaction_max_tokens(
                    chat_log,
                    retained_contents,
                    token_usage.prompt_tokens,
                )
                if max_summary_tokens > 1:
                    threshold = int(options.get(CONF_CONTEXT_THRESHOLD, DEFAULT_CONTEXT_THRESHOLD))
                    summary_prefix, summary = await self._summarize_compaction_groups(
                        [list(group) for group in reduction.summary_groups],
                        max_summary_tokens,
                        threshold,
                        language,
                    )
                    replacement = f"{summary_prefix}\n{summary}"
                else:
                    _LOGGER.info("Intra-turn compaction cannot produce a summary within the configured target")
            except Exception:
                _LOGGER.exception("Intra-turn conversation compaction failed")

        if replacement is None:
            if not archived:
                return False
            replacement = json.dumps(
                {
                    "archived_tool_context": {
                        "completed_exchanges": len(reduction.archive_groups),
                        "available_via": "context_recall",
                    }
                },
                separators=(",", ":"),
            )

        agent_id = next(
            (
                content.agent_id
                for content in reversed(chat_log.content)
                if isinstance(content, conversation.AssistantContent)
            ),
            getattr(self, "entity_id", None) or DOMAIN,
        )
        retained_contents = [
            content
            for index, content in enumerate(chat_log.content)
            if index not in reduction.remove_indexes
        ]
        retained_contents.insert(
            reduction.insert_at,
            conversation.AssistantContent(
                agent_id=agent_id,
                content=replacement,
                native={_INTRA_TURN_SUMMARY_NATIVE_KEY: True},
            ),
        )
        chat_log.content[:] = retained_contents
        _LOGGER.info(
            "Context threshold exceeded, older completed tool exchanges reduced"
        )
        return True

    async def _archive_message_history(
        self,
        chat_log: conversation.ChatLog,
        preserve_current_turn: bool = False,
        recent_turns_override: int | None = None,
        remove_archived_history: bool = True,
    ) -> bool:
        """Embed and archive older complete turns before removing them."""
        messages = chat_log.content
        history_split_index = self._compaction_history_split_index(
            messages, preserve_current_turn, recent_turns_override
        )
        if history_split_index is None or history_split_index <= 1:
            _LOGGER.info("Context threshold exceeded, but there is no older history to archive")
            return False

        archivable_contents = [
            content
            for content in messages[1:history_split_index]
            if not _is_generated_compaction_summary(content)
        ]
        if archivable_contents:
            await self._archive_embedding_groups(
                chat_log,
                _group_compaction_contents(archivable_contents),
                source=EMBEDDING_MEMORY_SOURCE_CONVERSATION,
            )

        if remove_archived_history:
            del messages[1:history_split_index]
        _LOGGER.info("Context threshold exceeded, older conversation history archived")
        return True

    def _remove_compactable_history(
        self,
        chat_log: conversation.ChatLog,
        preserve_current_turn: bool,
        recent_turns_override: int | None,
    ) -> bool:
        """Remove the older history segment used by compaction strategies."""
        history_split_index = self._compaction_history_split_index(
            chat_log.content, preserve_current_turn, recent_turns_override
        )
        if history_split_index is None or history_split_index <= 1:
            return False
        del chat_log.content[1:history_split_index]
        return True

    async def _retrieve_and_compact_message_history(
        self,
        chat_log: conversation.ChatLog,
        token_usage: RequestTokenUsage,
        preserve_current_turn: bool,
        recent_turns_override: int | None,
        language: str | None,
    ) -> bool:
        """Archive verbatim history, then replace that history with a summary."""
        plan = self._compaction_plan(
            chat_log,
            token_usage.prompt_tokens,
            preserve_current_turn,
            recent_turns_override,
        )
        if plan is None:
            return False

        planned_recent_turns = plan.recent_turns
        archived = False
        try:
            archived = await self._archive_message_history(
                chat_log,
                preserve_current_turn,
                planned_recent_turns,
                remove_archived_history=False,
            )
        except Exception:
            _LOGGER.exception("Embedding archival failed. Continuing with conversation compaction")

        try:
            return await self._compact_message_history(
                chat_log,
                token_usage,
                preserve_current_turn,
                planned_recent_turns,
                language,
            )
        except Exception:
            if archived:
                _LOGGER.exception(
                    "Conversation compaction failed; using the archived history for retrieval"
                )
                return self._remove_compactable_history(
                    chat_log, preserve_current_turn, planned_recent_turns
                )
            _LOGGER.exception(
                "Embedding archival and conversation compaction failed, "
                "falling back to clearing history"
            )
            return self._clear_message_history(chat_log)

    async def _truncate_message_history(
        self,
        chat_log: conversation.ChatLog,
        token_usage: RequestTokenUsage,
        preserve_current_turn: bool = False,
        recent_turns_override: int | None = None,
        language: str | None = None,
    ) -> bool:
        """Truncate message history based on strategy."""
        options = self.subentry.data
        strategy = options.get(
            CONF_CONTEXT_TRUNCATE_STRATEGY, DEFAULT_CONTEXT_TRUNCATE_STRATEGY
        )

        if strategy == "clear":
            _LOGGER.info("Context threshold exceeded, conversation history cleared")
            return self._clear_message_history(chat_log)
        if strategy == "compact":
            try:
                return await self._compact_message_history(
                    chat_log,
                    token_usage,
                    preserve_current_turn,
                    recent_turns_override,
                    language,
                )
            except Exception:
                _LOGGER.exception("Conversation compaction failed; falling back to clearing history")
                return self._clear_message_history(chat_log)
        if strategy == CONTEXT_TRUNCATE_STRATEGY_RETRIEVE:
            try:
                return await self._archive_message_history(
                    chat_log,
                    preserve_current_turn,
                    recent_turns_override,
                )
            except Exception:
                _LOGGER.exception("Embedding archival failed. Falling back to conversation compaction")
                try:
                    return await self._compact_message_history(
                        chat_log,
                        token_usage,
                        preserve_current_turn,
                        recent_turns_override,
                        language,
                    )
                except Exception:
                    _LOGGER.exception("Conversation compaction failed. Falling back to clearing history")
                    return self._clear_message_history(chat_log)
        if strategy == CONTEXT_TRUNCATE_STRATEGY_RETRIEVE_COMPACT:
            return await self._retrieve_and_compact_message_history(
                chat_log,
                token_usage,
                preserve_current_turn,
                recent_turns_override,
                language,
            )
        return False

