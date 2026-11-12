"""File-backed conversation memory for embedding retrieval."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any
from uuid import uuid4

from homeassistant.core import HomeAssistant

from .const import DOMAIN

_STORAGE_VERSION = 1
_RETENTION_PATTERN = re.compile(r"(?P<value>[1-9]\d*)(?P<unit>[mhd]?)")
_UNIT_SECONDS = {"m": 60, "h": 60 * 60, "d": 24 * 60 * 60}
_ESTIMATED_CHARACTERS_PER_TOKEN = 4

_LOGGER = logging.getLogger(__name__)


def normalize_embedding_history_retention(value: Any) -> str:
    """Validate an embedding retention duration and add the default unit."""
    if not isinstance(value, str):
        raise ValueError
    match = _RETENTION_PATTERN.fullmatch(value.strip())
    if match is None:
        raise ValueError
    return f"{match.group('value')}{match.group('unit') or 'd'}"


def embedding_history_retention(value: Any) -> timedelta:
    """Return a validated embedding retention duration."""
    normalized = normalize_embedding_history_retention(value)
    unit = normalized[-1]
    return timedelta(seconds=int(normalized[:-1]) * _UNIT_SECONDS[unit])


def chunk_embedding_text(text: str, max_estimated_tokens: int) -> list[str]:
    """Split text into bounded embedding inputs without dropping content."""
    if max_estimated_tokens <= 0:
        raise ValueError("max_estimated_tokens must be positive")
    if not text:
        return [text]

    max_characters = max_estimated_tokens * _ESTIMATED_CHARACTERS_PER_TOKEN
    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_characters:
        split_at = max(
            remaining.rfind(separator, 0, max_characters)
            for separator in ("\n", "\t", " ")
        )
        if split_at < max_characters // 2:
            split_at = max_characters
        else:
            split_at += 1
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:]
    if remaining or not chunks:
        chunks.append(remaining)
    return chunks


@dataclass(frozen=True, slots=True)
class EmbeddingMemoryScope:
    """Identify one agent conversation within an AI endpoint."""

    entry_id: str
    agent_id: str
    conversation_id: str


@dataclass(frozen=True, slots=True)
class EmbeddingMemoryChunk:
    """An additional search vector for one archived conversation turn."""

    embedding: list[float]
    searchable_text: str


@dataclass(frozen=True, slots=True)
class EmbeddingMemoryRecord:
    """A lossless archived conversation turn and its search vector."""

    record_id: str
    created_at: str
    expires_at: str
    model: str
    embedding: list[float]
    searchable_text: str
    messages: list[dict[str, Any]]
    additional_chunks: tuple[EmbeddingMemoryChunk, ...] = ()

    @classmethod
    def create(
        cls,
        *,
        model: str,
        embedding: Iterable[float],
        searchable_text: str,
        messages: list[dict[str, Any]],
        retention: timedelta,
        additional_chunks: Iterable[EmbeddingMemoryChunk] = (),
        now: datetime | None = None,
    ) -> EmbeddingMemoryRecord:
        """Create a record without collapsing repeated identical turns."""
        created_at = now or datetime.now(UTC)
        return cls(
            record_id=uuid4().hex,
            created_at=created_at.isoformat(),
            expires_at=(created_at + retention).isoformat(),
            model=model,
            embedding=list(embedding),
            searchable_text=searchable_text,
            messages=messages,
            additional_chunks=tuple(additional_chunks),
        )

    @property
    def chunks(self) -> tuple[EmbeddingMemoryChunk, ...]:
        """Return every searchable chunk associated with this turn."""
        return (
            EmbeddingMemoryChunk(self.embedding, self.searchable_text),
            *self.additional_chunks,
        )

    def is_expired(self, now: datetime) -> bool:
        """Return whether this record has passed its expiry time."""
        return datetime.fromisoformat(self.expires_at) <= now


@dataclass(frozen=True, slots=True)
class EmbeddingMemoryMatch:
    """An archived turn and its similarity to a recall query."""

    record: EmbeddingMemoryRecord
    score: float
    matched_text: str | None = None


def _cosine_similarity(first: list[float], second: list[float]) -> float | None:
    """Return cosine similarity, or None for incompatible zero vectors."""
    if not first or len(first) != len(second):
        return None
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        return None
    return sum(a * b for a, b in zip(first, second, strict=True)) / (
        first_norm * second_norm
    )


class EmbeddingMemoryStore:
    """Persist embedding memories in endpoint-scoped JSON files."""

    def __init__(self, hass: HomeAssistant, entry_id: str) -> None:
        """Initialize the store for an AI endpoint."""
        self._hass = hass
        self._entry_id = entry_id
        self._directory = self.storage_directory(hass, entry_id)
        self._lock = asyncio.Lock()

    @staticmethod
    def storage_directory(hass: HomeAssistant, entry_id: str) -> Path:
        """Return the exact storage directory for an endpoint."""
        safe_entry_id = sha256(entry_id.encode()).hexdigest()
        return Path(
            hass.config.path(".storage", f"{DOMAIN}_embedding_memory", safe_entry_id)
        )

    @classmethod
    async def async_clear_endpoint(cls, hass: HomeAssistant, entry_id: str) -> None:
        """Remove every embedding memory file for one endpoint."""
        directory = cls.storage_directory(hass, entry_id)
        await hass.async_add_executor_job(cls._clear_directory, directory)

    @staticmethod
    def _clear_directory(directory: Path) -> None:
        """Remove one resolved endpoint storage directory."""
        if directory.exists():
            shutil.rmtree(directory)

    def _path(self, scope: EmbeddingMemoryScope) -> Path:
        """Return the hashed file path for a conversation scope."""
        scope_key = f"{scope.agent_id}\0{scope.conversation_id}"
        return self._directory / f"{sha256(scope_key.encode()).hexdigest()}.json"

    async def async_append(
        self,
        scope: EmbeddingMemoryScope,
        records: list[EmbeddingMemoryRecord],
    ) -> None:
        """Append records atomically, removing expired and duplicate records."""
        if not records:
            return
        async with self._lock:
            await self._hass.async_add_executor_job(
                self._append, scope, records, datetime.now(UTC)
            )

    async def async_search(
        self,
        scope: EmbeddingMemoryScope,
        query_embedding: list[float],
        limit: int,
        model: str | None = None,
    ) -> list[EmbeddingMemoryMatch]:
        """Return the nearest unexpired memories within one conversation."""
        if limit <= 0 or not query_embedding:
            return []
        async with self._lock:
            return await self._hass.async_add_executor_job(
                self._search,
                scope,
                query_embedding,
                limit,
                model,
                datetime.now(UTC),
            )

    async def async_has_memories(
        self, scope: EmbeddingMemoryScope, model: str | None = None
    ) -> bool:
        """Return whether a conversation has any unexpired archived turns."""
        async with self._lock:
            return await self._hass.async_add_executor_job(
                self._has_memories, scope, model, datetime.now(UTC)
            )

    def _has_memories(
        self, scope: EmbeddingMemoryScope, model: str | None, now: datetime
    ) -> bool:
        """Synchronously check for unexpired records."""
        return any(
            model is None or record.model == model
            for record in self._active_records(self._path(scope), now)
        )

    def _search(
        self,
        scope: EmbeddingMemoryScope,
        query_embedding: list[float],
        limit: int,
        model: str | None,
        now: datetime,
    ) -> list[EmbeddingMemoryMatch]:
        """Synchronously score memories and remove expired records."""
        records = self._active_records(self._path(scope), now)

        matches: list[EmbeddingMemoryMatch] = []
        for record in records:
            if model is not None and record.model != model:
                continue
            scored_chunks = [
                (score, chunk.searchable_text)
                for chunk in record.chunks
                if (
                    score := _cosine_similarity(query_embedding, chunk.embedding)
                )
                is not None
            ]
            if scored_chunks:
                score, matched_text = max(scored_chunks, key=lambda item: item[0])
                matches.append(EmbeddingMemoryMatch(record, score, matched_text))
        matches.sort(key=lambda match: match.score, reverse=True)
        return matches[:limit]

    def _active_records(self, path: Path, now: datetime) -> list[EmbeddingMemoryRecord]:
        """Read active records and persist expiry cleanup for one scope."""
        payload = self._read_payload(path)
        if payload is None:
            return []
        raw_records = payload.get("records", [])
        records = [
            record
            for record in self._records_from_payload(payload)
            if not record.is_expired(now)
        ]
        if not records:
            path.unlink(missing_ok=True)
            return []
        if not isinstance(raw_records, list) or len(records) != len(raw_records):
            payload["records"] = [asdict(record) for record in records]
            self._atomic_write(path, payload)
        return records

    def _append(
        self,
        scope: EmbeddingMemoryScope,
        records: list[EmbeddingMemoryRecord],
        now: datetime,
    ) -> None:
        """Synchronously append records to their scoped file."""
        path = self._path(scope)
        existing = self._read_records(path)
        records_by_id = {
            record.record_id: record
            for record in existing
            if not record.is_expired(now)
        }
        records_by_id.update(
            (record.record_id, record)
            for record in records
            if not record.is_expired(now)
        )
        if not records_by_id:
            path.unlink(missing_ok=True)
            return
        payload = {
            "version": _STORAGE_VERSION,
            "scope": asdict(scope),
            "records": [asdict(record) for record in records_by_id.values()],
        }
        self._atomic_write(path, payload)

    async def async_cleanup_expired(self) -> None:
        """Delete expired records and empty memory files."""
        async with self._lock:
            await self._hass.async_add_executor_job(
                self._cleanup_expired, datetime.now(UTC)
            )

    def _cleanup_expired(self, now: datetime) -> None:
        """Synchronously remove expired records from every scoped file."""
        if not self._directory.exists():
            return
        for path in self._directory.glob("*.json"):
            payload = self._read_payload(path)
            if payload is None:
                continue
            records = [
                record
                for record in self._records_from_payload(payload)
                if not record.is_expired(now)
            ]
            if not records:
                path.unlink(missing_ok=True)
                continue
            payload["records"] = [asdict(record) for record in records]
            self._atomic_write(path, payload)

    def _read_records(self, path: Path) -> list[EmbeddingMemoryRecord]:
        """Read valid records from a scoped file."""
        payload = self._read_payload(path)
        return [] if payload is None else self._records_from_payload(payload)

    @staticmethod
    def _read_payload(path: Path) -> dict[str, Any] | None:
        """Read a supported memory payload."""
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except OSError, json.JSONDecodeError:
            _LOGGER.warning("Unable to read embedding memory file %s", path)
            return None
        if not isinstance(value, dict) or value.get("version") != _STORAGE_VERSION:
            return None
        return value

    @staticmethod
    def _records_from_payload(
        payload: dict[str, Any],
    ) -> list[EmbeddingMemoryRecord]:
        """Deserialize records from a supported payload."""
        raw_records = payload.get("records", [])
        if not isinstance(raw_records, list):
            return []
        records: list[EmbeddingMemoryRecord] = []
        for record in raw_records:
            if not isinstance(record, dict):
                continue
            try:
                record_data = dict(record)
                raw_chunks = record_data.pop("additional_chunks", [])
                if not isinstance(raw_chunks, list | tuple):
                    raise TypeError
                additional_chunks = tuple(
                    EmbeddingMemoryChunk(**chunk)
                    for chunk in raw_chunks
                    if isinstance(chunk, dict)
                )
                if len(additional_chunks) != len(raw_chunks):
                    raise TypeError
                records.append(
                    EmbeddingMemoryRecord(
                        **record_data,
                        additional_chunks=additional_chunks,
                    )
                )
            except TypeError, ValueError:
                _LOGGER.warning("Ignoring an invalid embedding memory record")
        return records

    @staticmethod
    def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
        """Write JSON through a sibling temporary file and atomic replace."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                ),
                encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
