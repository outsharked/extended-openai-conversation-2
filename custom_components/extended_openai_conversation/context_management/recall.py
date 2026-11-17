"""Automatic and model-directed conversation recall orchestration."""

from __future__ import annotations

from collections.abc import Mapping
import json
import logging
from typing import Any

from openai import AsyncClient
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.exceptions import HomeAssistantError

from ..const import (
    CONF_CONTEXT_RECALL_CLOSE_MATCH_TOLERANCE,
    CONF_CONTEXT_RECALL_MAX_RESULTS,
    CONF_CONTEXT_RECALL_MAX_TOKENS,
    CONF_CONTEXT_RECALL_RERANK,
    CONF_CONTEXT_RECALL_RERANK_MAX_CANDIDATE_TOKENS,
    CONF_CONTEXT_RECALL_RERANK_MIN_SCORE,
    CONF_CONTEXT_RECALL_SNAPSHOT_TTL_TIME,
    CONF_CONTEXT_RECALL_SNAPSHOT_TTL_TURNS,
    CONF_CONTEXT_TRUNCATE_STRATEGY,
    CONF_EMBEDDING_CHUNK_SIZE,
    CONF_EMBEDDINGS,
    CONF_EMBEDDINGS_MODEL,
    CONTEXT_TRUNCATE_RETRIEVAL_STRATEGIES,
    DEFAULT_CONTEXT_RECALL_CLOSE_MATCH_TOLERANCE,
    DEFAULT_CONTEXT_RECALL_MAX_RESULTS,
    DEFAULT_CONTEXT_RECALL_MAX_TOKENS,
    DEFAULT_CONTEXT_RECALL_RERANK,
    DEFAULT_CONTEXT_RECALL_RERANK_MAX_CANDIDATE_TOKENS,
    DEFAULT_CONTEXT_RECALL_RERANK_MIN_SCORE,
    DEFAULT_CONTEXT_RECALL_SNAPSHOT_TTL_TIME,
    DEFAULT_CONTEXT_RECALL_SNAPSHOT_TTL_TURNS,
    DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
    DEFAULT_EMBEDDING_CHUNK_SIZE,
    DEFAULT_EMBEDDING_RECALL_RESULTS,
)
from ..functions.context_recall import (
    build_browse_recalled_context_function_tool,
    build_context_recall_function_tool,
)
from ..runtime import OpenAIServiceRuntime
from .compaction import _compaction_transcript
from .embedding_memory import (
    EmbeddingMemoryMatch,
    EmbeddingMemoryScope,
    EmbeddingMemoryStore,
    async_embed_text_chunks,
    merge_embedding_matches,
    split_close_embedding_matches,
)
from .recall_snapshot import normalize_recall_snapshot_ttl_turns
from .reranker import RerankerClient, async_rerank_embedding_matches

_LOGGER = logging.getLogger(__name__)


class ContextRecallMixin:
    """Provide context-recall behaviour to an OpenAI entity."""

    entry: Any
    subentry: ConfigSubentry

    @property
    def _embedding_runtime(self) -> tuple[AsyncClient, EmbeddingMemoryStore]:
        """Return the configured embedding client and memory store."""
        raise NotImplementedError

    def _get_context_recall_function_tools(self) -> list[dict[str, Any]]:
        """Return recall when this agent uses embedding retrieval."""
        if (
            self.subentry.data.get(
                CONF_CONTEXT_TRUNCATE_STRATEGY,
                DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
            )
            not in CONTEXT_TRUNCATE_RETRIEVAL_STRATEGIES
            or not isinstance(self.entry.data.get(CONF_EMBEDDINGS), Mapping)
        ):
            return []
        max_results = int(
            self.subentry.data.get(
                CONF_CONTEXT_RECALL_MAX_RESULTS,
                DEFAULT_CONTEXT_RECALL_MAX_RESULTS,
            )
        )
        reranker_client, min_score, max_candidate_tokens = (
            self._context_recall_reranker()
        )
        configured_close_match_tolerance = self.subentry.data.get(
            CONF_CONTEXT_RECALL_CLOSE_MATCH_TOLERANCE,
            DEFAULT_CONTEXT_RECALL_CLOSE_MATCH_TOLERANCE,
        )
        close_match_tolerance = (
            float(configured_close_match_tolerance)
            if configured_close_match_tolerance
            else None
        )
        return [
            build_context_recall_function_tool(
                self.entry.entry_id,
                self.subentry.subentry_id,
                int(
                    self.subentry.data.get(
                        CONF_CONTEXT_RECALL_MAX_TOKENS,
                        DEFAULT_CONTEXT_RECALL_MAX_TOKENS,
                    )
                ),
                normalize_recall_snapshot_ttl_turns(
                    self.subentry.data.get(
                        CONF_CONTEXT_RECALL_SNAPSHOT_TTL_TURNS,
                        DEFAULT_CONTEXT_RECALL_SNAPSHOT_TTL_TURNS,
                    )
                ),
                max_results,
                snapshot_ttl_time=str(
                    self.subentry.data.get(
                        CONF_CONTEXT_RECALL_SNAPSHOT_TTL_TIME,
                        DEFAULT_CONTEXT_RECALL_SNAPSHOT_TTL_TIME,
                    )
                ),
                close_match_tolerance=close_match_tolerance,
                rerank=reranker_client is not None,
                rerank_min_score=min_score,
                rerank_max_candidate_tokens=max_candidate_tokens,
            ),
            build_browse_recalled_context_function_tool(
                self.entry.entry_id,
                self.subentry.subentry_id,
            ),
        ]

    def _context_recall_reranker(self) -> tuple[RerankerClient | None, float | None, int | None]:
        """Return this agent's enabled reranker and normalized controls."""
        runtime = self.entry.runtime_data
        if (
            not self.subentry.data.get(
                CONF_CONTEXT_RECALL_RERANK,
                DEFAULT_CONTEXT_RECALL_RERANK,
            )
            or not isinstance(runtime, OpenAIServiceRuntime)
            or runtime.reranker_client is None
        ):
            return None, None, None

        min_score_value = self.subentry.data.get(
            CONF_CONTEXT_RECALL_RERANK_MIN_SCORE,
            DEFAULT_CONTEXT_RECALL_RERANK_MIN_SCORE,
        )
        max_candidate_tokens_value = self.subentry.data.get(
            CONF_CONTEXT_RECALL_RERANK_MAX_CANDIDATE_TOKENS,
            DEFAULT_CONTEXT_RECALL_RERANK_MAX_CANDIDATE_TOKENS,
        )
        min_score = (
            float(min_score_value)
            if isinstance(min_score_value, str) and min_score_value
            else None
        )
        max_candidate_tokens = (
            int(max_candidate_tokens_value)
            if isinstance(max_candidate_tokens_value, str)
            and max_candidate_tokens_value
            else None
        )
        return runtime.reranker_client, min_score, max_candidate_tokens

    async def _async_recall_embedding_memories(
        self, chat_log: conversation.ChatLog
    ) -> tuple[list[EmbeddingMemoryMatch], list[EmbeddingMemoryMatch]]:
        """Return relevant archived turns for the latest user input."""
        if self.subentry.data.get(
            CONF_CONTEXT_TRUNCATE_STRATEGY,
            DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
        ) not in CONTEXT_TRUNCATE_RETRIEVAL_STRATEGIES:
            return [], []

        conversation_id = chat_log.conversation_id
        if not isinstance(conversation_id, str) or not conversation_id:
            return [], []
        latest_user_content = next(
            (
                content
                for content in reversed(chat_log.content)
                if isinstance(content, conversation.UserContent)
            ),
            None,
        )
        if latest_user_content is None:
            return [], []

        embeddings = self.entry.data.get(CONF_EMBEDDINGS)
        if not isinstance(embeddings, Mapping):
            return [], []
        model = embeddings.get(CONF_EMBEDDINGS_MODEL)
        if not isinstance(model, str) or not model:
            return [], []

        try:
            embedding_client, memory_store = self._embedding_runtime
            scope = EmbeddingMemoryScope(
                entry_id=self.entry.entry_id,
                agent_id=self.subentry.subentry_id,
                conversation_id=conversation_id,
            )
            if not await memory_store.async_has_memories(scope, model):
                return [], []
            chunk_size = embeddings.get(
                CONF_EMBEDDING_CHUNK_SIZE, DEFAULT_EMBEDDING_CHUNK_SIZE
            )
            if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
                raise HomeAssistantError("The embedding chunk size is invalid")
            query = _compaction_transcript([latest_user_content])
            embedded_query = await async_embed_text_chunks(
                embedding_client, model, query, chunk_size
            )
            reranker_client, min_score, max_candidate_tokens = (
                self._context_recall_reranker()
            )
            result_limit = DEFAULT_EMBEDDING_RECALL_RESULTS
            configured_close_match_tolerance = self.subentry.data.get(
                CONF_CONTEXT_RECALL_CLOSE_MATCH_TOLERANCE,
                DEFAULT_CONTEXT_RECALL_CLOSE_MATCH_TOLERANCE,
            )
            close_match_tolerance = (
                float(configured_close_match_tolerance)
                if configured_close_match_tolerance
                else None
            )
            candidate_limit = (
                int(
                    self.subentry.data.get(
                        CONF_CONTEXT_RECALL_MAX_RESULTS,
                        DEFAULT_CONTEXT_RECALL_MAX_RESULTS,
                    )
                )
                if reranker_client is not None or close_match_tolerance is not None
                else result_limit
            )
            matches = merge_embedding_matches(
                [
                    await memory_store.async_search(
                        scope,
                        chunk.embedding,
                        candidate_limit,
                        model,
                        excluded_turn_id=latest_user_content.created.isoformat(),
                    )
                    for chunk in embedded_query
                ],
                candidate_limit,
            )
            if reranker_client is None:
                return split_close_embedding_matches(
                    matches, result_limit, close_match_tolerance
                )
            reranker_result_limit = (
                candidate_limit
                if close_match_tolerance is not None
                else result_limit
            )
            try:
                reranked = merge_embedding_matches(
                    [
                        await async_rerank_embedding_matches(
                            reranker_client,
                            chunk.text,
                            matches,
                            max_results=reranker_result_limit,
                            min_score=min_score,
                            max_candidate_tokens=max_candidate_tokens,
                        )
                        for chunk in embedded_query
                    ],
                    reranker_result_limit,
                )
                return split_close_embedding_matches(
                    reranked, result_limit, close_match_tolerance
                )
            except Exception:  # pylint: disable=broad-except
                _LOGGER.exception("Embedding memory reranking failed. Using embedding order")
                return split_close_embedding_matches(
                    matches, result_limit, close_match_tolerance
                )
        except Exception:  # pylint: disable=broad-except
            _LOGGER.exception("Embedding memory retrieval failed. Continuing without recalled context")
            return [], []

    @staticmethod
    def _embedding_recall_message(recall_result: dict[str, Any], prefix: str) -> ChatCompletionUserMessageParam:
        """Render recalled turns as inert historical request context."""
        return ChatCompletionUserMessageParam(
            role="user",
            content=(
                f"{prefix}\n"
                f"{json.dumps(recall_result, ensure_ascii=False, default=str)}"
            ),
        )

    @classmethod
    def _place_embedding_recall(
        cls,
        messages: list[ChatCompletionMessageParam],
        recall_result: dict[str, Any] | None,
        prefix: str,
    ) -> None:
        """Insert temporary recalled context immediately before the latest user."""
        if recall_result is None:
            return
        latest_user_index = next(
            (
                index
                for index in reversed(range(len(messages)))
                if messages[index]["role"] == "user"
            ),
            len(messages),
        )
        messages.insert(
            latest_user_index,
            cls._embedding_recall_message(recall_result, prefix),
        )


