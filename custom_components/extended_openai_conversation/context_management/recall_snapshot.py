"""Short-lived in-memory snapshots for paginated conversation recall."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4


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
    ttl_turns: int
    turns_until_expiry: int


@dataclass(frozen=True, slots=True)
class RecallSnapshotPage:
    """Return one page and its current snapshot metadata."""

    memories: tuple[dict[str, Any], ...]
    matches_available: int
    current_page: int
    total_pages: int
    ttl_turns_remaining: int


class RecallSnapshotStore:
    """Keep paginated recall results until their turn-based TTL expires."""

    def __init__(self) -> None:
        """Initialize an empty snapshot store."""
        self._snapshots: dict[str, RecallSnapshot] = {}

    def advance_turn(self, agent_id: str) -> None:
        """Age every snapshot owned by an agent by one top-level turn."""
        expired: list[str] = []
        for snapshot_id, snapshot in self._snapshots.items():
            if snapshot.scope.agent_id != agent_id:
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
        ttl_turns: int,
    ) -> str:
        """Create a snapshot and return its unguessable identifier."""
        snapshot_id = uuid4().hex
        self._snapshots[snapshot_id] = RecallSnapshot(
            scope=scope,
            pages=tuple(tuple(page) for page in pages),
            matches_available=matches_available,
            ttl_turns=ttl_turns,
            # Aging occurs at the start of a later turn, so the extra internal
            # count lets a TTL of three remain available for three full turns.
            turns_until_expiry=ttl_turns + 1,
        )
        return snapshot_id

    def browse(
        self,
        scope: RecallSnapshotScope,
        snapshot_id: str,
        page: int,
    ) -> RecallSnapshotPage | None:
        """Return a scoped page and reset the snapshot lifetime."""
        snapshot = self._snapshots.get(snapshot_id)
        if (
            snapshot is None
            or snapshot.scope != scope
            or page < 1
            or page > len(snapshot.pages)
        ):
            return None
        snapshot.turns_until_expiry = snapshot.ttl_turns + 1
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
