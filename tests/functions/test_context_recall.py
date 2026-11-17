"""Tests for the archived conversation recall function."""

from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from custom_components.extended_openai_conversation.const import (
    CONF_EMBEDDINGS,
    CONF_EMBEDDINGS_MODEL,
    DEFAULT_CONTEXT_RECALL_MAX_RESULTS,
    DEFAULT_CONTEXT_RECALL_SNAPSHOT_TTL_TIME,
)
from custom_components.extended_openai_conversation.context_management import (
    EmbeddingMemoryMatch,
    EmbeddingMemoryRecord,
    EmbeddingMemoryScope,
    RecallSnapshotScope,
    RecallSnapshotStore,
    RerankerResult,
)
from custom_components.extended_openai_conversation.context_management.recall_result import (
    build_paginated_recall_result,
)
from custom_components.extended_openai_conversation.functions.context_recall import (
    BrowseRecalledContextFunction,
    ContextRecallFunction,
    build_browse_recalled_context_function_tool,
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
    tool = build_context_recall_function_tool(
        "endpoint-1", "agent-1", 4000, 3
    )

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
        "max_results": DEFAULT_CONTEXT_RECALL_MAX_RESULTS,
        "close_match_tolerance": None,
        "snapshot_ttl_turns": 3,
        "snapshot_ttl_time": DEFAULT_CONTEXT_RECALL_SNAPSHOT_TTL_TIME,
        "rerank": False,
    }
    assert tool["spec"]["parameters"]["properties"]["max_results"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": DEFAULT_CONTEXT_RECALL_MAX_RESULTS,
        "description": "Maximum historical turns to return, from 1 through 20. Defaults to 5.",
        "default": 5,
    }


def test_browse_recalled_context_tool_keeps_scope_out_of_model_arguments():
    """The model supplies only the snapshot identifier and requested page."""
    tool = build_browse_recalled_context_function_tool("endpoint-1", "agent-1")

    assert tool["spec"]["name"] == "browse_recalled_context"
    assert tool["spec"]["parameters"]["required"] == ["snapshot_id", "page"]
    assert tool["function"] == {
        "type": "browse_recalled_context",
        "entry_id": "endpoint-1",
        "agent_id": "agent-1",
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
        input=["What temperature did I settle on?"],
    )
    memory_store.async_search.assert_awaited_once_with(
        scope, [1.0, 0.0], 2, "embedding-model"
    )
    assert result == {
        "memories": [
            {
                "record_id": "older",
                "created_at": "2026-01-01T00:00:00+00:00",
                "relevance": 0.9,
                "messages": [
                    {"role": "user", "content": "Earlier preference"}
                ],
            },
            {
                "record_id": "newer",
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
        "current_page": 1,
        "total_pages": 1,
    }


async def test_context_recall_reranks_configured_candidate_pool(hass):
    """Manual recall reranks the configured pool before applying its requested limit."""
    embedding_client = MagicMock()
    embedding_client.embeddings.create = AsyncMock(
        return_value=SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[1.0, 0.0])]
        )
    )
    matches = [
        EmbeddingMemoryMatch(
            _record(
                f"record-{index}",
                f"2026-01-0{index + 1}T00:00:00+00:00",
                text,
            ),
            score,
        )
        for index, (text, score) in enumerate(
            (("first", 0.9), ("second", 0.8), ("best after reranking", 0.7))
        )
    ]
    memory_store = MagicMock()
    memory_store.async_has_memories = AsyncMock(return_value=True)
    memory_store.async_search = AsyncMock(return_value=matches)
    reranker_client = MagicMock()
    reranker_client.async_rerank = AsyncMock(
        return_value=[
            RerankerResult(
                index=2,
                score=0.99,
                document="best after reranking",
            )
        ]
    )
    entry = MagicMock()
    entry.data = {
        CONF_EMBEDDINGS: {CONF_EMBEDDINGS_MODEL: "embedding-model"}
    }
    entry.runtime_data = OpenAIServiceRuntime(
        client=MagicMock(),
        embedding_client=embedding_client,
        embedding_memory=memory_store,
        reranker_client=reranker_client,
    )
    hass.config_entries.async_get_entry.return_value = entry

    result = await ContextRecallFunction().execute(
        hass,
        {
            "type": "context_recall",
            "entry_id": "endpoint-1",
            "agent_id": "agent-1",
            "conversation_id": "conversation-1",
            "max_results": 10,
            "rerank": True,
            "rerank_min_score": 0.5,
            "rerank_max_candidate_tokens": 100,
        },
        {"query": "earlier details", "max_results": 1},
        None,
        [],
    )

    memory_store.async_search.assert_awaited_once_with(
        EmbeddingMemoryScope(
            entry_id="endpoint-1",
            agent_id="agent-1",
            conversation_id="conversation-1",
        ),
        [1.0, 0.0],
        10,
        "embedding-model",
    )
    reranker_client.async_rerank.assert_awaited_once_with(
        "earlier details",
        [
            "Recorded at: 2026-01-01T00:00:00+00:00\nConversation content:\nfirst",
            "Recorded at: 2026-01-02T00:00:00+00:00\nConversation content:\nsecond",
            (
                "Recorded at: 2026-01-03T00:00:00+00:00\n"
                "Conversation content:\nbest after reranking"
            ),
        ],
        max_results=1,
        min_score=0.5,
        max_candidate_tokens=100,
    )
    assert result["matches_returned"] == 1
    assert result["memories"][0]["record_id"] == "record-2"
    assert result["memories"][0]["relevance"] == 0.99


async def test_context_recall_reranker_failure_uses_embedding_order(hass):
    """A reranker outage falls back to the requested embedding-ranked results."""
    embedding_client = MagicMock()
    embedding_client.embeddings.create = AsyncMock(
        return_value=SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[1.0, 0.0])]
        )
    )
    matches = [
        EmbeddingMemoryMatch(
            _record(
                f"record-{index}",
                f"2026-01-0{index + 1}T00:00:00+00:00",
                f"memory {index}",
            ),
            1.0 - index / 10,
        )
        for index in range(3)
    ]
    memory_store = MagicMock()
    memory_store.async_has_memories = AsyncMock(return_value=True)
    memory_store.async_search = AsyncMock(return_value=matches)
    reranker_client = MagicMock()
    reranker_client.async_rerank = AsyncMock(
        side_effect=RuntimeError("reranker unavailable")
    )
    entry = MagicMock()
    entry.data = {
        CONF_EMBEDDINGS: {CONF_EMBEDDINGS_MODEL: "embedding-model"}
    }
    entry.runtime_data = OpenAIServiceRuntime(
        client=MagicMock(),
        embedding_client=embedding_client,
        embedding_memory=memory_store,
        reranker_client=reranker_client,
    )
    hass.config_entries.async_get_entry.return_value = entry

    result = await ContextRecallFunction().execute(
        hass,
        {
            "type": "context_recall",
            "entry_id": "endpoint-1",
            "agent_id": "agent-1",
            "conversation_id": "conversation-1",
            "max_results": 10,
            "rerank": True,
        },
        {"query": "earlier details", "max_results": 2},
        None,
        [],
    )

    assert result["matches_returned"] == 2
    assert [memory["record_id"] for memory in result["memories"]] == [
        "record-0",
        "record-1",
    ]


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

    result = build_paginated_recall_result([match], 100)

    assert result["matches_returned"] == 1
    assert result["matches_available"] == 1
    assert result["truncated"] is True
    assert result["memories"][0]["truncated"] is True
    assert "messages" not in result["memories"][0]
    excerpt = result["memories"][0]["excerpt"]
    assert excerpt
    assert match.matched_text.startswith(excerpt)
    assert "irrelevant record prefix" not in result["memories"][0]["excerpt"]
    assert len(json.dumps(result)) <= 400


async def test_close_recall_matches_are_exposed_through_snapshot(hass):
    """Close excluded candidates are advertised and browsable as marked records."""
    selected = EmbeddingMemoryMatch(
        _record("selected", "2026-01-01T00:00:00+00:00", "Selected memory"),
        0.9,
    )
    close = EmbeddingMemoryMatch(
        _record("close", "2026-01-02T00:00:00+00:00", "Close memory"),
        0.88,
    )
    runtime = OpenAIServiceRuntime(client=MagicMock())
    scope = RecallSnapshotScope("agent-1", "conversation-1")

    result = build_paginated_recall_result(
        [selected],
        4000,
        close_matches=[close],
        snapshot_store=runtime.recall_snapshots,
        snapshot_scope=scope,
        snapshot_ttl_turns=3,
    )

    assert result["matches_available"] == 2
    assert result["close_matches"] == {
        "count": 1,
        "highest_relevance": 0.88,
        "lowest_relevance": 0.88,
        "first_page": 2,
        "last_page": 2,
        "message": (
            "Similarly ranked historical records are available on the "
            "indicated snapshot pages. Use browse_recalled_context to "
            "inspect their contents and timestamps."
        ),
    }
    assert result["snapshot_id"]
    entry = MagicMock(runtime_data=runtime)
    hass.config_entries.async_get_entry.return_value = entry

    close_page = await BrowseRecalledContextFunction().execute(
        hass,
        {
            "type": "browse_recalled_context",
            "entry_id": "endpoint-1",
            "agent_id": "agent-1",
            "conversation_id": "conversation-1",
        },
        {"snapshot_id": result["snapshot_id"], "page": 2},
        None,
        [],
    )

    assert close_page["memories"][0]["record_id"] == "close"
    assert close_page["memories"][0]["close_match"] is True


async def test_context_recall_snapshots_later_pages_for_scoped_browsing(hass):
    """Large recall results can be paged only by their owning conversation."""
    matches = [
        EmbeddingMemoryMatch(
            _record(
                f"record-{index}",
                f"2026-01-0{index + 1}T00:00:00+00:00",
                f"Historical detail {index} " * 100,
            ),
            1.0 - index / 10,
        )
        for index in range(2)
    ]
    runtime = OpenAIServiceRuntime(client=MagicMock())
    scope = RecallSnapshotScope("agent-1", "conversation-1")
    first_page = build_paginated_recall_result(
        matches,
        200,
        snapshot_store=runtime.recall_snapshots,
        snapshot_scope=scope,
        snapshot_ttl_turns=3,
    )
    entry = MagicMock(runtime_data=runtime)
    hass.config_entries.async_get_entry.return_value = entry

    assert first_page["truncated"] is True
    assert first_page["current_page"] == 1
    assert first_page["total_pages"] > 1
    assert first_page["ttl_turns_remaining"] == 3

    result = await BrowseRecalledContextFunction().execute(
        hass,
        {
            "type": "browse_recalled_context",
            "entry_id": "endpoint-1",
            "agent_id": "agent-1",
            "conversation_id": "conversation-1",
        },
        {"snapshot_id": first_page["snapshot_id"], "page": 2},
        None,
        [],
    )

    assert result["current_page"] == 2
    assert result["total_pages"] == first_page["total_pages"]
    assert result["ttl_turns_remaining"] == 3
    assert result["memories"]

    denied = await BrowseRecalledContextFunction().execute(
        hass,
        {
            "type": "browse_recalled_context",
            "entry_id": "endpoint-1",
            "agent_id": "agent-1",
            "conversation_id": "conversation-2",
        },
        {"snapshot_id": first_page["snapshot_id"], "page": 2},
        None,
        [],
    )
    assert denied["error"]["code"] == "snapshot_not_found"


def test_recall_snapshot_ttl_ages_by_turn_and_browse_resets_it():
    """Snapshots survive the configured later turns and browsing renews them."""
    store = RecallSnapshotStore()
    scope = RecallSnapshotScope("agent-1", "conversation-1")
    snapshot_id = store.create(
        scope,
        [[{"record_id": "one"}]],
        1,
        3,
        timedelta(hours=2),
    )

    for _ in range(3):
        store.advance_turn("agent-1")
    assert store.browse(scope, snapshot_id, 1) is not None

    for _ in range(3):
        store.advance_turn("agent-1")
    assert store.browse(scope, snapshot_id, 1) is not None
    for _ in range(4):
        store.advance_turn("agent-1")
    assert store.browse(scope, snapshot_id, 1) is None


def test_recall_snapshot_time_ttl_expires_with_turn_ttl_disabled():
    """Snapshots expire on time even with turn expiry disabled."""
    current_time = [datetime(2026, 1, 1, tzinfo=UTC)]
    store = RecallSnapshotStore(lambda: current_time[0])
    scope = RecallSnapshotScope("agent-1", "conversation-1")
    snapshot_id = store.create(
        scope,
        [[{"record_id": "one"}]],
        1,
        None,
        timedelta(minutes=5),
    )

    for _ in range(10):
        store.advance_turn("agent-1")
    current_time[0] += timedelta(minutes=4)
    assert store.browse(scope, snapshot_id, 1) is not None

    current_time[0] += timedelta(minutes=4)
    store.cleanup_expired()
    assert store.browse(scope, snapshot_id, 1) is not None

    current_time[0] += timedelta(minutes=5)
    assert store.browse(scope, snapshot_id, 1) is None


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
        "current_page": 1,
        "total_pages": 1,
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
