"""Short-lived in-memory snapshots for paginated conversation recall."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4


def normalize_recall_snapshot_ttl_turns(value: Any) -> int | None:
    """Return a positive turn TTL, or None when turn expiry is disabled."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ValueError
    if isinstance(value, str):
        try:
            value = int(value.strip())
        except ValueError as err:
            raise ValueError from err
    if not isinstance(value, int) or value < 1:
        raise ValueError
    return value


@dataclass(frozen=True, slots=True)
class RecallSnapshotScope:
    """Identify the agent conversation allowed to browse a snapshot."""

    agent_id: str
    conversation_id: str


@dataclass(slots=True)
class RecallSnapshot:
    """Store the pages and remaining lifetime of one recall result."""

    scope: RecallSnapshotScope
    pages: tuple[tuple[dict[str, Any], ...], ...]
    matches_available: int
    ttl_turns: int | None
    turns_until_expiry: int
    ttl_time: timedelta
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class RecallSnapshotPage:
    """Return one page and its current snapshot metadata."""

    memories: tuple[dict[str, Any], ...]
    matches_available: int
    current_page: int
    total_pages: int
    ttl_turns_remaining: int | None


class RecallSnapshotStore:
    """Keep paginated recall results until either configured TTL expires."""

    def __init__(self, now: Callable[[], datetime] | None = None) -> None:
        """Initialize an empty snapshot store."""
        self._snapshots: dict[str, RecallSnapshot] = {}
        self._now = now or (lambda: datetime.now(UTC))

    def cleanup_expired(self) -> None:
        """Discard every snapshot whose time-based TTL has elapsed."""
        now = self._now()
        for snapshot_id, snapshot in tuple(self._snapshots.items()):
            if snapshot.expires_at <= now:
                del self._snapshots[snapshot_id]

    def advance_turn(self, agent_id: str) -> None:
        """Age every snapshot owned by an agent by one top-level turn."""
        self.cleanup_expired()
        expired: list[str] = []
        for snapshot_id, snapshot in self._snapshots.items():
            if snapshot.scope.agent_id != agent_id or snapshot.ttl_turns is None:
                continue
            snapshot.turns_until_expiry -= 1
            if snapshot.turns_until_expiry <= 0:
                expired.append(snapshot_id)
        for snapshot_id in expired:
            del self._snapshots[snapshot_id]

    def create(
        self,
        scope: RecallSnapshotScope,
        pages: list[list[dict[str, Any]]],
        matches_available: int,
        ttl_turns: int | None,
        ttl_time: timedelta,
    ) -> str:
        """Create a snapshot and return its unguessable identifier."""
        ttl_turns = normalize_recall_snapshot_ttl_turns(ttl_turns)
        now = self._now()
        snapshot_id = uuid4().hex
        self._snapshots[snapshot_id] = RecallSnapshot(
            scope=scope,
            pages=tuple(tuple(page) for page in pages),
            matches_available=matches_available,
            ttl_turns=ttl_turns,
            # Aging occurs at the start of a later turn, so the extra internal
            # count lets a TTL of three remain available for three full turns.
            turns_until_expiry=ttl_turns + 1 if ttl_turns is not None else 0,
            ttl_time=ttl_time,
            expires_at=now + ttl_time,
        )
        return snapshot_id

    def browse(
        self,
        scope: RecallSnapshotScope,
        snapshot_id: str,
        page: int,
    ) -> RecallSnapshotPage | None:
        """Return a scoped page and reset the snapshot lifetime."""
        now = self._now()
        snapshot = self._snapshots.get(snapshot_id)
        if (
            snapshot is None
            or snapshot.expires_at <= now
            or snapshot.scope != scope
            or page < 1
            or page > len(snapshot.pages)
        ):
            if snapshot is not None and snapshot.expires_at <= now:
                del self._snapshots[snapshot_id]
            return None
        snapshot.turns_until_expiry = (
            snapshot.ttl_turns + 1 if snapshot.ttl_turns is not None else 0
        )
        snapshot.expires_at = now + snapshot.ttl_time
        return RecallSnapshotPage(
            memories=snapshot.pages[page - 1],
            matches_available=snapshot.matches_available,
            current_page=page,
            total_pages=len(snapshot.pages),
            ttl_turns_remaining=snapshot.ttl_turns,
        )

    def discard(self, scope: RecallSnapshotScope, snapshot_id: str) -> None:
        """Discard a snapshot only when its scope matches."""
        snapshot = self._snapshots.get(snapshot_id)
        if snapshot is not None and snapshot.scope == scope:
            del self._snapshots[snapshot_id]
