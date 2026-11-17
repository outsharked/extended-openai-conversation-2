"""Bounded model-facing results for recalled conversation history."""

from __future__ import annotations

from datetime import timedelta
import json
from typing import Any

from ..const import (
    DEFAULT_CONTEXT_RECALL_SNAPSHOT_TTL_TIME,
    DEFAULT_CONTEXT_RECALL_SNAPSHOT_TTL_TURNS,
)
from .embedding_memory import EmbeddingMemoryMatch, embedding_history_retention
from .recall_snapshot import RecallSnapshotScope, RecallSnapshotStore

_DEFAULT_SNAPSHOT_TTL_TIME = embedding_history_retention(
    DEFAULT_CONTEXT_RECALL_SNAPSHOT_TTL_TIME
)


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


def build_recall_page_result(
    memories: list[dict[str, Any]],
    matches_available: int,
    *,
    current_page: int,
    total_pages: int,
    snapshot_id: str | None = None,
    ttl_turns_remaining: int | None = None,
    close_matches: dict[str, Any] | None = None,
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
    if close_matches is not None:
        result["close_matches"] = close_matches
    return result


def _memory_item(
    match: EmbeddingMemoryMatch,
    include_relevance: bool,
    close_match: bool,
) -> dict[str, Any]:
    """Return a complete archived memory item."""
    memory: dict[str, Any] = {
        "record_id": match.record.record_id,
        "created_at": match.record.created_at,
        "messages": match.record.messages,
    }
    if include_relevance:
        memory["relevance"] = round(match.score, 6)
    if close_match:
        memory["close_match"] = True
    return memory


def _fragment_memory(
    match: EmbeddingMemoryMatch,
    text: str,
    max_tokens: int,
    include_relevance: bool,
    close_match: bool,
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
        if close_match:
            fragment["close_match"] = True

        low = 0
        high = len(remaining)
        while low < high:
            midpoint = (low + high + 1) // 2
            fragment["excerpt"] = remaining[:midpoint]
            candidate = build_recall_page_result(
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
                **({"close_match": True} if close_match else {}),
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
    close_match: bool = False,
) -> list[list[dict[str, Any]]]:
    """Split selected memories into pages bounded by estimated tokens."""
    items: list[dict[str, Any]] = []
    for match in sorted(matches, key=lambda item: item.record.created_at):
        memory = _memory_item(match, include_relevance, close_match)
        candidate = build_recall_page_result(
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
            if match.matched_text in transcript_chunks:
                transcript_chunks.remove(match.matched_text)
            transcript_chunks.insert(0, match.matched_text)
        transcript = "".join(transcript_chunks)
        items.extend(
            _fragment_memory(
                match, transcript, max_tokens, include_relevance, close_match
            )
        )

    pages: list[list[dict[str, Any]]] = [[]]
    for item in items:
        candidate = build_recall_page_result(
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
    close_matches: list[EmbeddingMemoryMatch] | None = None,
    snapshot_store: RecallSnapshotStore | None = None,
    snapshot_scope: RecallSnapshotScope | None = None,
    snapshot_ttl_turns: int | None = DEFAULT_CONTEXT_RECALL_SNAPSHOT_TTL_TURNS,
    snapshot_ttl_time: timedelta = _DEFAULT_SNAPSHOT_TTL_TIME,
    include_relevance: bool = True,
) -> dict[str, Any]:
    """Return the first bounded page and snapshot later pages when available."""
    selected_pages = _recall_pages(
        matches,
        max_tokens,
        include_relevance=include_relevance,
    )
    close_matches = close_matches or []
    close_pages = (
        _recall_pages(
            close_matches,
            max_tokens,
            include_relevance=include_relevance,
            close_match=True,
        )
        if close_matches
        else []
    )
    pages = [*selected_pages, *close_pages]
    total_pages = len(pages)
    matches_available = len(matches) + len(close_matches)
    snapshot_id: str | None = None
    if total_pages > 1 and snapshot_store is not None and snapshot_scope is not None:
        snapshot_id = snapshot_store.create(
            snapshot_scope,
            pages,
            matches_available,
            snapshot_ttl_turns,
            snapshot_ttl_time,
        )
    close_match_summary = (
        {
            "count": len(close_matches),
            "highest_relevance": round(max(match.score for match in close_matches), 6),
            "lowest_relevance": round(min(match.score for match in close_matches), 6),
            "first_page": len(selected_pages) + 1,
            "last_page": total_pages,
            "message": (
                "Similarly ranked historical records are available on the "
                "indicated snapshot pages. Use browse_recalled_context to "
                "inspect their contents and timestamps."
            ),
        }
        if close_matches
        else None
    )
    return build_recall_page_result(
        pages[0],
        matches_available,
        current_page=1,
        total_pages=total_pages,
        snapshot_id=snapshot_id,
        ttl_turns_remaining=(
            snapshot_ttl_turns
            if snapshot_id is not None and snapshot_ttl_turns is not None
            else None
        ),
        close_matches=close_match_summary,
    )



