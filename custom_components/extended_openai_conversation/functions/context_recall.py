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
    MAX_CONTEXT_RECALL_RESULTS,
)
from ..embedding_memory import EmbeddingMemoryMatch, EmbeddingMemoryScope
from ..runtime import OpenAIServiceRuntime
from .base import Function

_LOGGER = logging.getLogger(__name__)


def build_context_recall_function_tool(
    entry_id: str, agent_id: str, max_tokens: int
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
                "configured recall budget. This tool can access only the current "
                "conversation."
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
        },
    }


def _estimated_tokens(value: Any) -> int:
    """Estimate serialized tokens using the integration's four-character rule."""
    serialized = json.dumps(value, ensure_ascii=False, default=str)
    return max(1, (len(serialized) + 3) // 4)


def build_bounded_recall_result(
    matches: list[EmbeddingMemoryMatch],
    max_tokens: int,
    *,
    include_relevance: bool = True,
) -> dict[str, Any]:
    """Return the most relevant memories within an estimated token budget."""
    selected: list[dict[str, Any]] = []
    truncated = False

    def result() -> dict[str, Any]:
        memories = sorted(selected, key=lambda item: item["created_at"])
        return {
            "memories": memories,
            "matches_returned": len(memories),
            "matches_available": len(matches),
            "truncated": truncated,
        }

    for match in matches:
        memory: dict[str, Any] = {
            "created_at": match.record.created_at,
            "messages": match.record.messages,
        }
        if include_relevance:
            memory["relevance"] = round(match.score, 6)
        selected.append(memory)
        if _estimated_tokens(result()) <= max_tokens:
            continue

        selected.pop()
        truncated = True
        matched_text = match.matched_text or match.record.searchable_text
        excerpt: dict[str, Any] = {
            "created_at": match.record.created_at,
            "excerpt": "",
            "truncated": True,
        }
        if include_relevance:
            excerpt["relevance"] = round(match.score, 6)

        low = 0
        high = len(matched_text)
        while low < high:
            midpoint = (low + high + 1) // 2
            excerpt["excerpt"] = matched_text[:midpoint]
            selected.append(excerpt)
            fits = _estimated_tokens(result()) <= max_tokens
            selected.pop()
            if fits:
                low = midpoint
            else:
                high = midpoint - 1
        if low:
            excerpt["excerpt"] = matched_text[:low]
            selected.append(excerpt)
        break

    if len(selected) < len(matches):
        truncated = True
    return result()


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
                return build_bounded_recall_result([], max_tokens)
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

        return build_bounded_recall_result(
            matches,
            max_tokens,
        )
