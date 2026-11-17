"""LLM-callable retrieval for archived conversation context."""

from __future__ import annotations

from collections.abc import Mapping
import logging
from typing import Any

import probatio as vol

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from ..const import (
    CONF_EMBEDDING_CHUNK_SIZE,
    CONF_EMBEDDINGS,
    CONF_EMBEDDINGS_MODEL,
    DEFAULT_CONTEXT_RECALL_MAX_RESULTS,
    DEFAULT_CONTEXT_RECALL_MAX_TOKENS,
    DEFAULT_CONTEXT_RECALL_RERANK,
    DEFAULT_CONTEXT_RECALL_RESULTS,
    DEFAULT_CONTEXT_RECALL_SNAPSHOT_TTL_TURNS,
    DEFAULT_EMBEDDING_CHUNK_SIZE,
)
from ..context_management import (
    EmbeddingMemoryScope,
    RecallSnapshotScope,
    async_embed_text_chunks,
    async_rerank_embedding_matches,
    merge_embedding_matches,
    split_close_embedding_matches,
)
from ..context_management.recall_result import (
    build_paginated_recall_result,
    build_recall_page_result,
)
from ..runtime import OpenAIServiceRuntime
from .base import Function

_LOGGER = logging.getLogger(__name__)


def build_context_recall_function_tool(
    entry_id: str,
    agent_id: str,
    max_tokens: int,
    snapshot_ttl_turns: int,
    max_results: int = DEFAULT_CONTEXT_RECALL_MAX_RESULTS,
    *,
    close_match_tolerance: float | None = None,
    rerank: bool = DEFAULT_CONTEXT_RECALL_RERANK,
    rerank_min_score: float | None = None,
    rerank_max_candidate_tokens: int | None = None,
) -> dict[str, Any]:
    """Build the model-facing conversation recall tool."""
    default_results = min(DEFAULT_CONTEXT_RECALL_RESULTS, max_results)
    function_config: dict[str, Any] = {
        "type": "context_recall",
        "entry_id": entry_id,
        "agent_id": agent_id,
        "max_tokens": max_tokens,
        "max_results": max_results,
        "close_match_tolerance": close_match_tolerance,
        "snapshot_ttl_turns": snapshot_ttl_turns,
        "rerank": rerank,
    }
    if rerank_min_score is not None:
        function_config["rerank_min_score"] = rerank_min_score
    if rerank_max_candidate_tokens is not None:
        function_config["rerank_max_candidate_tokens"] = (
            rerank_max_candidate_tokens
        )
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
                        "maximum": max_results,
                        "description": (
                            "Maximum historical turns to return, from 1 through "
                            f"{max_results}. Defaults to {default_results}."
                        ),
                        "default": default_results,
                    },
                },
                "required": ["query"],
            },
        },
        "function": function_config,
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
                        "max_results", default=DEFAULT_CONTEXT_RECALL_MAX_RESULTS
                    ): vol.All(int, vol.Range(min=1)),
                    vol.Optional("close_match_tolerance", default=None): vol.Any(
                        None,
                        vol.All(
                            vol.Coerce(float),
                            vol.Range(min=0, min_included=False),
                        ),
                    ),
                    vol.Optional(
                        "snapshot_ttl_turns",
                        default=DEFAULT_CONTEXT_RECALL_SNAPSHOT_TTL_TURNS,
                    ): vol.All(int, vol.Range(min=1)),
                    vol.Optional(
                        "rerank", default=DEFAULT_CONTEXT_RECALL_RERANK
                    ): bool,
                    vol.Optional("rerank_min_score"): float,
                    vol.Optional("rerank_max_candidate_tokens"): vol.All(
                        int, vol.Range(min=1)
                    ),
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
        max_results_cap = function_config.get(
            "max_results", DEFAULT_CONTEXT_RECALL_MAX_RESULTS
        )
        close_match_tolerance = function_config.get("close_match_tolerance")
        snapshot_ttl_turns = function_config.get(
            "snapshot_ttl_turns", DEFAULT_CONTEXT_RECALL_SNAPSHOT_TTL_TURNS
        )
        if (
            not isinstance(max_results, int)
            or isinstance(max_results, bool)
            or not 1 <= max_results <= max_results_cap
        ):
            return _error(
                "invalid_max_results",
                f"max_results must be an integer from 1 through {max_results_cap}",
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
        chunk_size = embeddings.get(
            CONF_EMBEDDING_CHUNK_SIZE, DEFAULT_EMBEDDING_CHUNK_SIZE
        )
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
            return _error("recall_unavailable", "Conversation recall is unavailable")

        scope = EmbeddingMemoryScope(
            entry_id=function_config["entry_id"],
            agent_id=function_config["agent_id"],
            conversation_id=function_config["conversation_id"],
        )
        try:
            if not await runtime.embedding_memory.async_has_memories(scope, model):
                return build_paginated_recall_result([], max_tokens)
            embedded_query = await async_embed_text_chunks(
                runtime.embedding_client, model, query, chunk_size
            )
            rerank = bool(function_config.get("rerank", False))
            reranker_client = runtime.reranker_client if rerank else None
            candidate_limit = (
                max_results_cap
                if reranker_client is not None or close_match_tolerance is not None
                else max_results
            )
            matches = merge_embedding_matches(
                [
                    await runtime.embedding_memory.async_search(
                        scope,
                        chunk.embedding,
                        candidate_limit,
                        model,
                    )
                    for chunk in embedded_query
                ],
                candidate_limit,
            )
        except Exception:  # pylint: disable=broad-except
            _LOGGER.exception("Context recall failed")
            return _error("recall_failed", "Conversation recall failed")

        if reranker_client is not None:
            reranker_result_limit = (
                candidate_limit
                if close_match_tolerance is not None
                else max_results
            )
            try:
                matches = merge_embedding_matches(
                    [
                        await async_rerank_embedding_matches(
                            reranker_client,
                            chunk.text,
                            matches,
                            max_results=reranker_result_limit,
                            min_score=function_config.get("rerank_min_score"),
                            max_candidate_tokens=function_config.get(
                                "rerank_max_candidate_tokens"
                            ),
                        )
                        for chunk in embedded_query
                    ],
                    reranker_result_limit,
                )
            except Exception:  # pylint: disable=broad-except
                _LOGGER.exception(
                    "Context recall reranking failed; using embedding order"
                )

        matches, close_matches = split_close_embedding_matches(
            matches, max_results, close_match_tolerance
        )

        return build_paginated_recall_result(
            matches,
            max_tokens,
            close_matches=close_matches,
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
        return build_recall_page_result(
            list(snapshot_page.memories),
            snapshot_page.matches_available,
            current_page=snapshot_page.current_page,
            total_pages=snapshot_page.total_pages,
            snapshot_id=snapshot_id,
            ttl_turns_remaining=snapshot_page.ttl_turns_remaining,
        )
