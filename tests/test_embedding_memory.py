"""Tests for file-backed embedding conversation memory."""

from dataclasses import asdict
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from custom_components.extended_openai_conversation.context_management import (
    EMBEDDING_MEMORY_SOURCE_INTRA_TURN,
    EmbeddingMemoryChunk,
    EmbeddingMemoryCounts,
    EmbeddingMemoryMatch,
    EmbeddingMemoryRecord,
    EmbeddingMemoryScope,
    EmbeddingMemoryStore,
    async_embed_text_chunks,
    chunk_embedding_text,
    embedding_history_retention,
    normalize_embedding_history_retention,
    split_close_embedding_matches,
)


def test_embedding_chunks_preserve_text_and_respect_estimated_limit():
    """Chunking retains every character within the configured estimate."""
    text = "alpha beta gamma delta epsilon zeta eta theta"

    chunks = chunk_embedding_text(text, max_estimated_tokens=3)

    assert len(chunks) > 1
    assert "".join(chunks) == text
    assert all(len(chunk) <= 12 for chunk in chunks)


def test_close_embedding_matches_use_lowest_selected_score_as_cutoff():
    """Excluded results inside the configured score tolerance remain available."""
    records = [
        EmbeddingMemoryRecord.create(
            model="embedding-model",
            embedding=[1.0],
            searchable_text=f"memory {index}",
            messages=[{"role": "user", "content": f"memory {index}"}],
            retention=timedelta(days=1),
        )
        for index in range(4)
    ]
    matches = [
        EmbeddingMemoryMatch(record, score)
        for record, score in zip(records, (0.9, 0.8, 0.76, 0.6), strict=True)
    ]

    selected, close = split_close_embedding_matches(matches, 2, 0.05)

    assert selected == matches[:2]
    assert close == matches[2:3]


async def test_embedding_chunks_adapt_to_endpoint_context_rejection():
    """A tokenizer-specific context rejection losslessly subdivides the input."""
    client = MagicMock()
    request_number = 0

    async def create_embedding(**kwargs):
        nonlocal request_number
        request_number += 1
        if request_number == 1:
            raise RuntimeError(
                "request (8453 tokens) exceeds the available context size"
            )
        return SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[float(request_number)])]
        )

    client.embeddings.create = AsyncMock(side_effect=create_embedding)
    text = "dense-json-content-" * 2

    chunks = await async_embed_text_chunks(client, "embedding-model", text, 100)

    assert "".join(chunk.text for chunk in chunks) == text
    assert len(chunks) == 2
    assert [chunk.embedding for chunk in chunks] == [[2.0], [3.0]]


def _configure_path(hass, tmp_path: Path) -> None:
    """Make Home Assistant storage paths resolve beneath the test directory."""
    hass.config.path.side_effect = lambda *parts: str(tmp_path.joinpath(*parts))


def test_embedding_retention_parsing():
    """Retention accepts minutes, hours, days, and defaults to days."""
    assert normalize_embedding_history_retention("10") == "10d"
    assert embedding_history_retention("10m") == timedelta(minutes=10)
    assert embedding_history_retention("5h") == timedelta(hours=5)
    assert embedding_history_retention("50d") == timedelta(days=50)


async def test_store_keeps_structured_messages_and_repeated_turns(hass, tmp_path: Path):
    """A scoped file retains original messages and genuine repeated turns."""
    _configure_path(hass, tmp_path)
    store = EmbeddingMemoryStore(hass, "endpoint-1")
    scope = EmbeddingMemoryScope(
        entry_id="endpoint-1",
        agent_id="agent-1",
        conversation_id="conversation-1",
    )
    record = EmbeddingMemoryRecord.create(
        model="embedding-model",
        embedding=[0.1, -0.2],
        searchable_text='[{"role":"user","content":"Remember alpha"}]',
        messages=[{"role": "user", "content": "Remember alpha"}],
        retention=timedelta(days=30),
    )

    await store.async_append(scope, [record])
    await store.async_append(scope, [record])
    repeated = EmbeddingMemoryRecord.create(
        model="embedding-model",
        embedding=[0.1, -0.2],
        searchable_text=record.searchable_text,
        messages=record.messages,
        retention=timedelta(days=30),
    )
    await store.async_append(scope, [repeated])

    files = list(store.storage_directory(hass, "endpoint-1").glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["scope"] == {
        "entry_id": "endpoint-1",
        "agent_id": "agent-1",
        "conversation_id": "conversation-1",
    }
    assert len(payload["records"]) == 2
    assert payload["records"][0]["messages"] == [
        {"role": "user", "content": "Remember alpha"}
    ]
    assert payload["records"][0]["embedding"] == [0.1, -0.2]


async def test_cleanup_removes_expired_memory_files(hass, tmp_path: Path):
    """Startup cleanup removes conversations whose records have expired."""
    _configure_path(hass, tmp_path)
    store = EmbeddingMemoryStore(hass, "endpoint-1")
    scope = EmbeddingMemoryScope(
        entry_id="endpoint-1",
        agent_id="agent-1",
        conversation_id="conversation-1",
    )
    expired = EmbeddingMemoryRecord.create(
        model="embedding-model",
        embedding=[0.1],
        searchable_text="Expired memory",
        messages=[{"role": "user", "content": "Expired memory"}],
        retention=timedelta(minutes=1),
        now=datetime(2020, 1, 1, tzinfo=UTC),
    )

    await store.async_append(scope, [expired])
    directory = store.storage_directory(hass, "endpoint-1")
    current = EmbeddingMemoryRecord.create(
        model="embedding-model",
        embedding=[0.2],
        searchable_text="Current memory",
        messages=[{"role": "user", "content": "Current memory"}],
        retention=timedelta(days=1),
    )
    await store.async_append(scope, [current])
    path = next(directory.glob("*.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"][0]["expires_at"] = expired.expires_at
    path.write_text(json.dumps(payload), encoding="utf-8")

    await store.async_cleanup_expired()

    assert not list(directory.glob("*.json"))


async def test_search_returns_nearest_compatible_memories(hass, tmp_path: Path):
    """Search ranks matching vectors and skips records from another model size."""
    _configure_path(hass, tmp_path)
    store = EmbeddingMemoryStore(hass, "endpoint-1")
    scope = EmbeddingMemoryScope(
        entry_id="endpoint-1",
        agent_id="agent-1",
        conversation_id="conversation-1",
    )
    records = [
        EmbeddingMemoryRecord.create(
            model="embedding-model",
            embedding=embedding,
            searchable_text=text,
            messages=[{"role": "user", "content": text}],
            retention=timedelta(days=1),
        )
        for embedding, text in (
            ([1.0, 0.0], "closest"),
            ([0.5, 0.5], "second"),
            ([-1.0, 0.0], "opposite"),
            ([1.0, 0.0, 0.0], "incompatible"),
        )
    ]
    await store.async_append(scope, records)

    assert await store.async_has_memories(scope, "embedding-model") is True
    assert await store.async_has_memories(scope, "other-model") is False
    matches = await store.async_search(
        scope, [1.0, 0.0], limit=2, model="embedding-model"
    )

    assert [match.record.searchable_text for match in matches] == [
        "closest",
        "second",
    ]
    assert matches[0].score == 1.0


async def test_literal_search_returns_matching_original_messages_without_vectors(
    hass, tmp_path: Path
):
    """Literal recall works on lossless records without an embeddings model."""
    _configure_path(hass, tmp_path)
    store = EmbeddingMemoryStore(hass, "endpoint-1")
    scope = EmbeddingMemoryScope(
        entry_id="endpoint-1",
        agent_id="agent-1",
        conversation_id="conversation-1",
    )
    record = EmbeddingMemoryRecord.create(
        model=None,
        embedding=[],
        searchable_text="Archived transcript",
        messages=[
            {"role": "user", "content": "Remember Project ALPHA"},
            {"role": "assistant", "content": "Alpha is due on Friday"},
            {"role": "user", "content": "Unrelated message"},
        ],
        retention=timedelta(days=1),
    )
    await store.async_append(scope, [record])

    matches = await store.async_search_literal(scope, "alpha")

    assert [match.record.messages for match in matches] == [
        [{"role": "user", "content": "Remember Project ALPHA"}],
        [{"role": "assistant", "content": "Alpha is due on Friday"}],
    ]
    assert all(match.record.embedding == [] for match in matches)
    assert await store.async_search_literal(
        scope, "alpha", case_sensitive=True
    ) == []


async def test_search_ranks_one_turn_by_its_best_chunk(hass, tmp_path: Path):
    """Multiple chunk vectors produce one deduplicated turn match."""
    _configure_path(hass, tmp_path)
    store = EmbeddingMemoryStore(hass, "endpoint-1")
    scope = EmbeddingMemoryScope(
        entry_id="endpoint-1",
        agent_id="agent-1",
        conversation_id="conversation-1",
    )
    record = EmbeddingMemoryRecord.create(
        model="embedding-model",
        embedding=[0.0, 1.0],
        searchable_text="unrelated first chunk",
        additional_chunks=(
            EmbeddingMemoryChunk([1.0, 0.0], "matching second chunk"),
        ),
        messages=[{"role": "user", "content": "complete original turn"}],
        retention=timedelta(days=1),
    )
    await store.async_append(scope, [record])

    matches = await store.async_search(
        scope, [1.0, 0.0], limit=5, model="embedding-model"
    )

    assert len(matches) == 1
    assert matches[0].record.messages == [
        {"role": "user", "content": "complete original turn"}
    ]
    assert matches[0].score == 1.0
    assert matches[0].matched_text == "matching second chunk"


async def test_search_can_exclude_intra_turn_memories_from_the_current_turn(hass, tmp_path: Path):
    """Automatic recall can omit active-turn archives without hiding older data."""
    _configure_path(hass, tmp_path)
    store = EmbeddingMemoryStore(hass, "endpoint-1")
    scope = EmbeddingMemoryScope(
        entry_id="endpoint-1",
        agent_id="agent-1",
        conversation_id="conversation-1",
    )
    records = [
        EmbeddingMemoryRecord.create(
            model="embedding-model",
            embedding=[1.0, 0.0],
            searchable_text=text,
            messages=[{"role": "tool_result", "content": text}],
            retention=timedelta(days=1),
            source=EMBEDDING_MEMORY_SOURCE_INTRA_TURN,
            turn_id=turn_id,
        )
        for text, turn_id in (
            ("current active turn", "current-turn"),
            ("earlier active turn", "earlier-turn"),
        )
    ]
    await store.async_append_temporary(scope, records)

    matches = await store.async_search(
        scope,
        [1.0, 0.0],
        limit=5,
        model="embedding-model",
        excluded_turn_id="current-turn",
    )

    assert [match.record.searchable_text for match in matches] == [
        "earlier active turn"
    ]


async def test_temporary_intra_turn_memories_are_searchable_without_disk_write(
    hass, tmp_path: Path
):
    """Intra-turn recall remains available without entering persistent storage."""
    _configure_path(hass, tmp_path)
    store = EmbeddingMemoryStore(hass, "endpoint-1")
    scope = EmbeddingMemoryScope(
        entry_id="endpoint-1",
        agent_id="agent-1",
        conversation_id="conversation-1",
    )
    record = EmbeddingMemoryRecord.create(
        model="embedding-model",
        embedding=[1.0, 0.0],
        searchable_text="temporary tool result",
        messages=[{"role": "tool_result", "content": "temporary tool result"}],
        retention=timedelta(days=1),
        source=EMBEDDING_MEMORY_SOURCE_INTRA_TURN,
        turn_id="turn-1",
    )

    await store.async_append_temporary(scope, [record])

    matches = await store.async_search(
        scope, [1.0, 0.0], limit=5, model="embedding-model"
    )
    assert [match.record for match in matches] == [record]
    assert not list(store.storage_directory(hass, "endpoint-1").glob("*.json"))


async def test_clear_temporary_memories_is_scoped_to_the_requested_agent(
    hass, tmp_path: Path
):
    """The manual action clears only one agent's temporary recall data."""
    _configure_path(hass, tmp_path)
    store = EmbeddingMemoryStore(hass, "endpoint-1")
    first_scope = EmbeddingMemoryScope(
        "endpoint-1", "agent-1", "conversation-1"
    )
    second_scope = EmbeddingMemoryScope(
        "endpoint-1", "agent-2", "conversation-2"
    )
    persistent = EmbeddingMemoryRecord.create(
        model="embedding-model",
        embedding=[1.0, 0.0],
        searchable_text="persistent result",
        messages=[{"role": "user", "content": "persistent result"}],
        retention=timedelta(days=1),
    )
    await store.async_append(first_scope, [persistent])
    for scope, text in (
        (first_scope, "first temporary result"),
        (second_scope, "second temporary result"),
    ):
        await store.async_append_temporary(
            scope,
            [
                EmbeddingMemoryRecord.create(
                    model="embedding-model",
                    embedding=[1.0, 0.0],
                    searchable_text=text,
                    messages=[{"role": "tool_result", "content": text}],
                    retention=timedelta(days=1),
                    source=EMBEDDING_MEMORY_SOURCE_INTRA_TURN,
                    turn_id="turn-1",
                )
            ],
        )

    await store.async_clear_temporary("agent-1")

    matches = await store.async_search(
        first_scope, [1.0, 0.0], limit=5, model="embedding-model"
    )
    assert [match.record for match in matches] == [persistent]
    assert await store.async_has_memories(second_scope, "embedding-model") is True


async def test_memory_counts_and_persistent_clear_are_scoped_to_agent(
    hass, tmp_path: Path
):
    """Counts and persistent clearing support agent and endpoint scopes."""
    _configure_path(hass, tmp_path)
    store = EmbeddingMemoryStore(hass, "endpoint-1")
    first_scope = EmbeddingMemoryScope(
        "endpoint-1", "agent-1", "conversation-1"
    )
    second_scope = EmbeddingMemoryScope(
        "endpoint-1", "agent-2", "conversation-2"
    )
    for scope in (first_scope, second_scope):
        await store.async_append(
            scope,
            [
                EmbeddingMemoryRecord.create(
                    model="embedding-model",
                    embedding=[1.0, 0.0],
                    searchable_text=scope.agent_id,
                    messages=[{"role": "user", "content": scope.agent_id}],
                    retention=timedelta(days=1),
                )
            ],
        )
        await store.async_append_temporary(
            scope,
            [
                EmbeddingMemoryRecord.create(
                    model="embedding-model",
                    embedding=[0.0, 1.0],
                    searchable_text=f"temporary {scope.agent_id}",
                    messages=[
                        {
                            "role": "tool_result",
                            "content": f"temporary {scope.agent_id}",
                        }
                    ],
                    retention=timedelta(days=1),
                    source=EMBEDDING_MEMORY_SOURCE_INTRA_TURN,
                    turn_id="turn-1",
                )
            ],
        )

    assert await store.async_counts() == EmbeddingMemoryCounts(2, 2)
    assert await store.async_counts("agent-1") == EmbeddingMemoryCounts(1, 1)

    await store.async_clear_persistent("agent-1")

    assert await store.async_counts() == EmbeddingMemoryCounts(1, 2)
    assert await store.async_counts("agent-1") == EmbeddingMemoryCounts(0, 1)
    assert await store.async_counts("agent-2") == EmbeddingMemoryCounts(1, 1)


async def test_cleanup_removes_legacy_persisted_intra_turn_memories(
    hass, tmp_path: Path
):
    """Persistent stores discard intra-turn records written by older versions."""
    _configure_path(hass, tmp_path)
    store = EmbeddingMemoryStore(hass, "endpoint-1")
    scope = EmbeddingMemoryScope(
        "endpoint-1", "agent-1", "conversation-1"
    )
    persistent = EmbeddingMemoryRecord.create(
        model="embedding-model",
        embedding=[1.0, 0.0],
        searchable_text="persistent conversation memory",
        messages=[{"role": "user", "content": "persistent conversation memory"}],
        retention=timedelta(days=1),
    )
    legacy_temporary = EmbeddingMemoryRecord.create(
        model="embedding-model",
        embedding=[0.0, 1.0],
        searchable_text="legacy intra-turn result",
        messages=[{"role": "tool_result", "content": "legacy intra-turn result"}],
        retention=timedelta(days=1),
        source=EMBEDDING_MEMORY_SOURCE_INTRA_TURN,
        turn_id="turn-1",
    )
    await store.async_append(scope, [persistent])
    path = next(store.storage_directory(hass, "endpoint-1").glob("*.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"].append(asdict(legacy_temporary))
    path.write_text(json.dumps(payload), encoding="utf-8")

    await store.async_cleanup_expired()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert [record["record_id"] for record in payload["records"]] == [
        persistent.record_id
    ]


async def test_clear_endpoint_only_removes_the_requested_endpoint(hass, tmp_path: Path):
    """The explicit wipe action cannot affect another endpoint's memories."""
    _configure_path(hass, tmp_path)
    first = EmbeddingMemoryStore(hass, "endpoint-1")
    second = EmbeddingMemoryStore(hass, "endpoint-2")
    for store, entry_id in ((first, "endpoint-1"), (second, "endpoint-2")):
        await store.async_append(
            EmbeddingMemoryScope(entry_id, "agent-1", "conversation-1"),
            [
                EmbeddingMemoryRecord.create(
                    model="embedding-model",
                    embedding=[0.1],
                    searchable_text=entry_id,
                    messages=[{"role": "user", "content": entry_id}],
                    retention=timedelta(days=1),
                )
            ],
        )

    await EmbeddingMemoryStore.async_clear_endpoint(hass, "endpoint-1")

    assert not first.storage_directory(hass, "endpoint-1").exists()
    assert second.storage_directory(hass, "endpoint-2").exists()
