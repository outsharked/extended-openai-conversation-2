"""LLM-callable retrieval for archived conversation context."""

from __future__ import annotations

from collections.abc import Mapping
import json
import logging
from typing import Any

import probatio as vol

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from ..const import (
    CONF_EMBEDDINGS,
    CONF_EMBEDDINGS_MODEL,
    DEFAULT_CONTEXT_RECALL_MAX_TOKENS,
    DEFAULT_CONTEXT_RECALL_RESULTS,
    DEFAULT_CONTEXT_RECALL_SNAPSHOT_TTL_TURNS,
    MAX_CONTEXT_RECALL_RESULTS,
)
from ..embedding_memory import EmbeddingMemoryMatch, EmbeddingMemoryScope
from ..recall_snapshot import RecallSnapshotScope, RecallSnapshotStore
from ..runtime import OpenAIServiceRuntime
from .base import Function

_LOGGER = logging.getLogger(__name__)


def build_context_recall_function_tool(
    entry_id: str, agent_id: str, max_tokens: int, snapshot_ttl_turns: int
) -> dict[str, Any]:
    """Build the model-facing conversation recall tool."""
    return {
        "spec": {
            "name": "context_recall",
            "description": (
                "Semantically search older messages archived from this conversation. "
                "Use this when relevant earlier details are not present in the active "
                "conversation context. Results contain the original historical "
                "messages, timestamps, and relevance scores. Results may contain "
                "bounded excerpts marked as truncated when full turns exceed the "
                "configured recall budget. When multiple pages are available, use "
                "browse_recalled_context with the returned snapshot_id. This tool "
                "can access only the current conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 2000,
                        "description": (
                            "A focused semantic search query describing the earlier "
                            "information to recall."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_CONTEXT_RECALL_RESULTS,
                        "description": (
                            "Maximum historical turns to return, from 1 through 20. "
                            "Defaults to 5."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
        "function": {
            "type": "context_recall",
            "entry_id": entry_id,
            "agent_id": agent_id,
            "max_tokens": max_tokens,
            "snapshot_ttl_turns": snapshot_ttl_turns,
        },
    }


def build_browse_recalled_context_function_tool(entry_id: str, agent_id: str) -> dict[str, Any]:
    """Build the model-facing tool used to browse a recall snapshot."""
    return {
        "spec": {
            "name": "browse_recalled_context",
            "description": (
                "Read another page from a paginated context_recall result. Use the "
                "snapshot_id, current_page, and total_pages returned by recall. A "
                "successful read refreshes the snapshot's remaining turn lifetime."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "snapshot_id": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                            "Snapshot identifier returned by context_recall."
                        ),
                    },
                    "page": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "One-based page number to retrieve.",
                    },
                },
                "required": ["snapshot_id", "page"],
            },
        },
        "function": {
            "type": "browse_recalled_context",
            "entry_id": entry_id,
            "agent_id": agent_id,
        },
    }


def _estimated_tokens(value: Any) -> int:
    """Estimate serialized tokens using the integration's four-character rule."""
    serialized = json.dumps(value, ensure_ascii=False, default=str)
    return max(1, (len(serialized) + 3) // 4)


def _matches_returned(memories: list[dict[str, Any]]) -> int:
    """Count distinct archived records represented on a page."""
    return len(
        {
            memory["record_id"]
            for memory in memories
            if isinstance(memory.get("record_id"), str)
        }
    )


def _recall_page_result(
    memories: list[dict[str, Any]],
    matches_available: int,
    *,
    current_page: int,
    total_pages: int,
    snapshot_id: str | None = None,
    ttl_turns_remaining: int | None = None,
) -> dict[str, Any]:
    """Build one model-facing recall page."""
    result: dict[str, Any] = {
        "memories": memories,
        "matches_returned": _matches_returned(memories),
        "matches_available": matches_available,
        "truncated": total_pages > 1,
        "current_page": current_page,
        "total_pages": total_pages,
    }
    if snapshot_id is not None:
        result["snapshot_id"] = snapshot_id
    if ttl_turns_remaining is not None:
        result["ttl_turns_remaining"] = ttl_turns_remaining
    return result


def _memory_item(match: EmbeddingMemoryMatch, include_relevance: bool) -> dict[str, Any]:
    """Return a complete archived memory item."""
    memory: dict[str, Any] = {
        "record_id": match.record.record_id,
        "created_at": match.record.created_at,
        "messages": match.record.messages,
    }
    if include_relevance:
        memory["relevance"] = round(match.score, 6)
    return memory


def _fragment_memory(
    match: EmbeddingMemoryMatch,
    text: str,
    max_tokens: int,
    include_relevance: bool,
) -> list[dict[str, Any]]:
    """Split one oversized archived turn into bounded transcript fragments."""
    fragments: list[dict[str, Any]] = []
    remaining = text
    while remaining:
        fragment: dict[str, Any] = {
            "record_id": match.record.record_id,
            "created_at": match.record.created_at,
            "excerpt": "",
            "truncated": True,
            "fragment_index": 999999,
            "fragment_count": 999999,
        }
        if include_relevance:
            fragment["relevance"] = round(match.score, 6)

        low = 0
        high = len(remaining)
        while low < high:
            midpoint = (low + high + 1) // 2
            fragment["excerpt"] = remaining[:midpoint]
            candidate = _recall_page_result(
                [fragment],
                999999,
                current_page=999999,
                total_pages=999999,
                snapshot_id="0" * 32,
                ttl_turns_remaining=999999,
            )
            if _estimated_tokens(candidate) <= max_tokens:
                low = midpoint
            else:
                high = midpoint - 1

        # Very small configured budgets may not fit even the response metadata.
        # Return one empty marked excerpt instead of creating one page per character.
        if low == 0:
            fragment["fragment_index"] = 1
            fragment["fragment_count"] = 1
            fragments.append(fragment)
            break
        fragment["excerpt"] = remaining[:low]
        fragments.append(fragment)
        remaining = remaining[low:]

    if not fragments:
        fragments.append(
            {
                "record_id": match.record.record_id,
                "created_at": match.record.created_at,
                "excerpt": "",
                "truncated": True,
                "fragment_index": 1,
                "fragment_count": 1,
            }
        )
    fragment_count = len(fragments)
    for index, fragment in enumerate(fragments, start=1):
        fragment["fragment_index"] = index
        fragment["fragment_count"] = fragment_count
    return fragments


def _recall_pages(
    matches: list[EmbeddingMemoryMatch],
    max_tokens: int,
    *,
    include_relevance: bool = True,
) -> list[list[dict[str, Any]]]:
    """Split selected memories into pages bounded by estimated tokens."""
    items: list[dict[str, Any]] = []
    for match in sorted(matches, key=lambda item: item.record.created_at):
        memory = _memory_item(match, include_relevance)
        candidate = _recall_page_result(
            [memory],
            len(matches),
            current_page=999999,
            total_pages=999999,
            snapshot_id="0" * 32,
            ttl_turns_remaining=999999,
        )
        if _estimated_tokens(candidate) <= max_tokens:
            items.append(memory)
            continue
        transcript_chunks = [
            chunk.searchable_text for chunk in match.record.chunks
        ]
        if match.matched_text:
            try:
                transcript_chunks.remove(match.matched_text)
            except ValueError:
                pass
            transcript_chunks.insert(0, match.matched_text)
        transcript = "".join(transcript_chunks)
        items.extend(_fragment_memory(match, transcript, max_tokens, include_relevance))

    pages: list[list[dict[str, Any]]] = [[]]
    for item in items:
        candidate = _recall_page_result(
            [*pages[-1], item],
            len(matches),
            current_page=999999,
            total_pages=999999,
            snapshot_id="0" * 32,
            ttl_turns_remaining=999999,
        )
        if pages[-1] and _estimated_tokens(candidate) > max_tokens:
            pages.append([item])
        else:
            pages[-1].append(item)
    return pages


def build_paginated_recall_result(
    matches: list[EmbeddingMemoryMatch],
    max_tokens: int,
    *,
    snapshot_store: RecallSnapshotStore | None = None,
    snapshot_scope: RecallSnapshotScope | None = None,
    snapshot_ttl_turns: int = DEFAULT_CONTEXT_RECALL_SNAPSHOT_TTL_TURNS,
    include_relevance: bool = True,
) -> dict[str, Any]:
    """Return the first bounded page and snapshot later pages when available."""
    pages = _recall_pages(
        matches,
        max_tokens,
        include_relevance=include_relevance,
    )
    total_pages = len(pages)
    snapshot_id: str | None = None
    if total_pages > 1 and snapshot_store is not None and snapshot_scope is not None:
        snapshot_id = snapshot_store.create(
            snapshot_scope,
            pages,
            len(matches),
            snapshot_ttl_turns,
        )
    return _recall_page_result(
        pages[0],
        len(matches),
        current_page=1,
        total_pages=total_pages,
        snapshot_id=snapshot_id,
        ttl_turns_remaining=(
            snapshot_ttl_turns if snapshot_id is not None else None
        ),
    )


def _error(code: str, message: str) -> dict[str, Any]:
    """Return a model-facing recall error."""
    return {"memories": [], "error": {"code": code, "message": message}}


class ContextRecallFunction(Function):
    """Search the embedding archive for the current conversation."""

    def __init__(self) -> None:
        """Initialize the hidden function configuration schema."""
        super().__init__(
            vol.Schema(
                {
                    vol.Required("entry_id"): str,
                    vol.Required("agent_id"): str,
                    vol.Required("conversation_id"): str,
                    vol.Optional(
                        "max_tokens", default=DEFAULT_CONTEXT_RECALL_MAX_TOKENS
                    ): vol.All(int, vol.Range(min=1)),
                    vol.Optional(
                        "snapshot_ttl_turns",
                        default=DEFAULT_CONTEXT_RECALL_SNAPSHOT_TTL_TURNS,
                    ): vol.All(int, vol.Range(min=1)),
                }
            )
        )

    async def execute(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Search archived turns using the configured embeddings endpoint."""
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return _error("invalid_query", "A non-empty recall query is required")
        query = query.strip()
        if len(query) > 2000:
            return _error("invalid_query", "The recall query exceeds 2000 characters")

        max_results = arguments.get(
            "max_results", DEFAULT_CONTEXT_RECALL_RESULTS
        )
        max_tokens = function_config.get(
            "max_tokens", DEFAULT_CONTEXT_RECALL_MAX_TOKENS
        )
        snapshot_ttl_turns = function_config.get(
            "snapshot_ttl_turns", DEFAULT_CONTEXT_RECALL_SNAPSHOT_TTL_TURNS
        )
        if (
            not isinstance(max_results, int)
            or isinstance(max_results, bool)
            or not 1 <= max_results <= MAX_CONTEXT_RECALL_RESULTS
        ):
            return _error(
                "invalid_max_results",
                "max_results must be an integer from 1 through 20",
            )

        entry = hass.config_entries.async_get_entry(function_config["entry_id"])
        if entry is None or not isinstance(entry.runtime_data, OpenAIServiceRuntime):
            return _error("recall_unavailable", "Conversation recall is unavailable")
        runtime = entry.runtime_data
        embeddings = entry.data.get(CONF_EMBEDDINGS)
        if (
            not isinstance(embeddings, Mapping)
            or runtime.embedding_client is None
            or runtime.embedding_memory is None
        ):
            return _error("recall_unavailable", "Conversation recall is unavailable")
        model = embeddings.get(CONF_EMBEDDINGS_MODEL)
        if not isinstance(model, str) or not model:
            return _error("recall_unavailable", "Conversation recall is unavailable")

        scope = EmbeddingMemoryScope(
            entry_id=function_config["entry_id"],
            agent_id=function_config["agent_id"],
            conversation_id=function_config["conversation_id"],
        )
        try:
            if not await runtime.embedding_memory.async_has_memories(scope, model):
                return build_paginated_recall_result([], max_tokens)
            response = await runtime.embedding_client.embeddings.create(
                model=model,
                input=query,
            )
            if not response.data or not response.data[0].embedding:
                return _error(
                    "recall_failed",
                    "The embeddings endpoint returned no recall vector",
                )
            matches = await runtime.embedding_memory.async_search(
                scope,
                list(response.data[0].embedding),
                max_results,
                model,
            )
        except Exception:  # pylint: disable=broad-except
            _LOGGER.exception("Context recall failed")
            return _error("recall_failed", "Conversation recall failed")

        return build_paginated_recall_result(
            matches,
            max_tokens,
            snapshot_store=runtime.recall_snapshots,
            snapshot_scope=RecallSnapshotScope(
                agent_id=function_config["agent_id"],
                conversation_id=function_config["conversation_id"],
            ),
            snapshot_ttl_turns=snapshot_ttl_turns,
        )


class BrowseRecalledContextFunction(Function):
    """Read another page from a short-lived recall snapshot."""

    def __init__(self) -> None:
        """Initialize the hidden function configuration schema."""
        super().__init__(
            vol.Schema(
                {
                    vol.Required("entry_id"): str,
                    vol.Required("agent_id"): str,
                    vol.Required("conversation_id"): str,
                }
            )
        )

    async def execute(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return a scoped snapshot page and refresh its lifetime."""
        snapshot_id = arguments.get("snapshot_id")
        page = arguments.get("page")
        if not isinstance(snapshot_id, str) or not snapshot_id:
            return _error(
                "invalid_snapshot_id", "A non-empty snapshot_id is required"
            )
        if not isinstance(page, int) or isinstance(page, bool) or page < 1:
            return _error("invalid_page", "page must be a positive integer")

        entry = hass.config_entries.async_get_entry(function_config["entry_id"])
        if entry is None or not isinstance(entry.runtime_data, OpenAIServiceRuntime):
            return _error("recall_unavailable", "Conversation recall is unavailable")
        snapshot_page = entry.runtime_data.recall_snapshots.browse(
            RecallSnapshotScope(
                agent_id=function_config["agent_id"],
                conversation_id=function_config["conversation_id"],
            ),
            snapshot_id,
            page,
        )
        if snapshot_page is None:
            return _error(
                "snapshot_not_found",
                "The recall snapshot or requested page is unavailable",
            )
        return _recall_page_result(
            list(snapshot_page.memories),
            snapshot_page.matches_available,
            current_page=snapshot_page.current_page,
            total_pages=snapshot_page.total_pages,
            snapshot_id=snapshot_id,
            ttl_turns_remaining=snapshot_page.ttl_turns_remaining,
        )
