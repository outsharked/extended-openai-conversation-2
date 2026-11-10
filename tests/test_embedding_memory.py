"""Tests for file-backed embedding conversation memory."""

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

from custom_components.extended_openai_conversation.embedding_memory import (
    EmbeddingMemoryRecord,
    EmbeddingMemoryScope,
    EmbeddingMemoryStore,
    embedding_history_retention,
    normalize_embedding_history_retention,
)


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
