"""Tests for the archived conversation recall function."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from custom_components.extended_openai_conversation.const import (
    CONF_EMBEDDINGS,
    CONF_EMBEDDINGS_MODEL,
)
from custom_components.extended_openai_conversation.embedding_memory import (
    EmbeddingMemoryMatch,
    EmbeddingMemoryRecord,
    EmbeddingMemoryScope,
)
from custom_components.extended_openai_conversation.functions.context_recall import (
    ContextRecallFunction,
    build_bounded_recall_result,
    build_context_recall_function_tool,
)
from custom_components.extended_openai_conversation.runtime import OpenAIServiceRuntime


def _record(record_id: str, created_at: str, content: str) -> EmbeddingMemoryRecord:
    """Return a stored memory record."""
    return EmbeddingMemoryRecord(
        record_id=record_id,
        created_at=created_at,
        expires_at="2027-01-01T00:00:00+00:00",
        model="embedding-model",
        embedding=[1.0, 0.0],
        searchable_text=content,
        messages=[{"role": "user", "content": content}],
    )


def test_context_recall_tool_keeps_scope_out_of_model_arguments():
    """Only the search query and result limit are exposed to the model."""
    tool = build_context_recall_function_tool("endpoint-1", "agent-1", 4000)

    assert tool["spec"]["name"] == "context_recall"
    assert tool["spec"]["parameters"]["required"] == ["query"]
    assert set(tool["spec"]["parameters"]["properties"]) == {
        "query",
        "max_results",
    }
    assert tool["function"] == {
        "type": "context_recall",
        "entry_id": "endpoint-1",
        "agent_id": "agent-1",
        "max_tokens": 4000,
    }


async def test_context_recall_returns_original_messages_chronologically(hass):
    """Recall ranks semantically but presents selected history in time order."""
    embedding_client = MagicMock()
    embedding_client.embeddings.create = AsyncMock(
        return_value=SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[1.0, 0.0])]
        )
    )
    memory_store = MagicMock()
    memory_store.async_has_memories = AsyncMock(return_value=True)
    newer = EmbeddingMemoryMatch(
        _record("newer", "2026-02-01T00:00:00+00:00", "Later correction"),
        0.95,
    )
    older = EmbeddingMemoryMatch(
        _record("older", "2026-01-01T00:00:00+00:00", "Earlier preference"),
        0.9,
    )
    memory_store.async_search = AsyncMock(return_value=[newer, older])
    entry = MagicMock()
    entry.data = {
        CONF_EMBEDDINGS: {CONF_EMBEDDINGS_MODEL: "embedding-model"}
    }
    entry.runtime_data = OpenAIServiceRuntime(
        client=MagicMock(),
        embedding_client=embedding_client,
        embedding_memory=memory_store,
    )
    hass.config_entries.async_get_entry.return_value = entry
    config = {
        "type": "context_recall",
        "entry_id": "endpoint-1",
        "agent_id": "agent-1",
        "conversation_id": "conversation-1",
    }

    result = await ContextRecallFunction().execute(
        hass,
        config,
        {"query": "What temperature did I settle on?", "max_results": 2},
        None,
        [],
    )

    scope = EmbeddingMemoryScope(
        entry_id="endpoint-1",
        agent_id="agent-1",
        conversation_id="conversation-1",
    )
    memory_store.async_has_memories.assert_awaited_once_with(
        scope, "embedding-model"
    )
    embedding_client.embeddings.create.assert_awaited_once_with(
        model="embedding-model",
        input="What temperature did I settle on?",
    )
    memory_store.async_search.assert_awaited_once_with(
        scope, [1.0, 0.0], 2, "embedding-model"
    )
    assert result == {
        "memories": [
            {
                "created_at": "2026-01-01T00:00:00+00:00",
                "relevance": 0.9,
                "messages": [
                    {"role": "user", "content": "Earlier preference"}
                ],
            },
            {
                "created_at": "2026-02-01T00:00:00+00:00",
                "relevance": 0.95,
                "messages": [
                    {"role": "user", "content": "Later correction"}
                ],
            },
        ],
        "matches_returned": 2,
        "matches_available": 2,
        "truncated": False,
    }


def test_context_recall_bounds_oversized_results():
    """An oversized turn returns a bounded excerpt of its matching chunk."""
    match = EmbeddingMemoryMatch(
        _record(
            "large",
            "2026-01-01T00:00:00+00:00",
            "irrelevant record prefix " * 500,
        ),
        0.9,
        "relevant later chunk " * 500,
    )

    result = build_bounded_recall_result([match], 100)

    assert result["matches_returned"] == 1
    assert result["matches_available"] == 1
    assert result["truncated"] is True
    assert result["memories"][0]["truncated"] is True
    assert "messages" not in result["memories"][0]
    assert result["memories"][0]["excerpt"].startswith("relevant later chunk")
    assert "irrelevant record prefix" not in result["memories"][0]["excerpt"]
    assert len(json.dumps(result)) <= 400


async def test_context_recall_skips_embedding_when_archive_is_empty(hass):
    """An empty conversation archive returns normally without an API call."""
    embedding_client = MagicMock()
    embedding_client.embeddings.create = AsyncMock()
    memory_store = MagicMock()
    memory_store.async_has_memories = AsyncMock(return_value=False)
    entry = MagicMock()
    entry.data = {
        CONF_EMBEDDINGS: {CONF_EMBEDDINGS_MODEL: "embedding-model"}
    }
    entry.runtime_data = OpenAIServiceRuntime(
        client=MagicMock(),
        embedding_client=embedding_client,
        embedding_memory=memory_store,
    )
    hass.config_entries.async_get_entry.return_value = entry

    result = await ContextRecallFunction().execute(
        hass,
        {
            "type": "context_recall",
            "entry_id": "endpoint-1",
            "agent_id": "agent-1",
            "conversation_id": "conversation-1",
        },
        {"query": "earlier details"},
        None,
        [],
    )

    assert result == {
        "memories": [],
        "matches_returned": 0,
        "matches_available": 0,
        "truncated": False,
    }
    embedding_client.embeddings.create.assert_not_awaited()
    memory_store.async_search.assert_not_called()


async def test_context_recall_returns_runtime_failures_as_tool_results(hass):
    """A failed embeddings request cannot abort the whole agent response."""
    embedding_client = MagicMock()
    embedding_client.embeddings.create = AsyncMock(
        side_effect=RuntimeError("endpoint unavailable")
    )
    memory_store = MagicMock()
    memory_store.async_has_memories = AsyncMock(return_value=True)
    entry = MagicMock()
    entry.data = {
        CONF_EMBEDDINGS: {CONF_EMBEDDINGS_MODEL: "embedding-model"}
    }
    entry.runtime_data = OpenAIServiceRuntime(
        client=MagicMock(),
        embedding_client=embedding_client,
        embedding_memory=memory_store,
    )
    hass.config_entries.async_get_entry.return_value = entry

    result = await ContextRecallFunction().execute(
        hass,
        {
            "type": "context_recall",
            "entry_id": "endpoint-1",
            "agent_id": "agent-1",
            "conversation_id": "conversation-1",
        },
        {"query": "earlier details"},
        None,
        [],
    )

    assert result == {
        "memories": [],
        "error": {
            "code": "recall_failed",
            "message": "Conversation recall failed",
        },
    }
