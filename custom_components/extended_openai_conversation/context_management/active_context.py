"""Budget recalled conversation data retained in active model context."""

from __future__ import annotations

from typing import Any

from homeassistant.components import conversation

from .compaction import _estimate_serialized_tokens

_RECALL_TOOL_NAMES = {
    "context_recall",
    "context_search",
    "browse_recalled_context",
}


def _tool_result_data(content: conversation.ToolResultContent) -> Any:
    """Return a tool result's data across supported Home Assistant versions."""
    result = getattr(content, "result", None)
    if result is not None:
        return result.data
    return content.tool_result


def _recall_result_payload(value: Any) -> dict[str, Any] | None:
    """Return the paginated recall payload inside a tool result."""
    if not isinstance(value, dict):
        return None
    payload = value.get("result")
    if not isinstance(payload, dict) or not isinstance(payload.get("memories"), list):
        return None
    return payload


def _released_recall_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Replace recalled memories with a compact page reference."""
    released: dict[str, Any] = {"recalled_context_released": True}
    for key in ("snapshot_id", "current_page", "total_pages"):
        if key in payload:
            released[key] = payload[key]
    if "snapshot_id" in released:
        released["message"] = (
            "This page was released from active context. Use "
            "browse_recalled_context to retrieve it again."
        )
    else:
        released["message"] = (
            "This result was released from active context. Repeat the original "
            "recall or search request to retrieve it again."
        )
    return released


def _recalled_memory_tokens(payload: dict[str, Any]) -> int:
    """Estimate only the recalled memory data counted against the budget."""
    memories = payload["memories"]
    return _estimate_serialized_tokens(memories) if memories else 0


def _bound_active_recalled_context(
    contents: list[conversation.Content],
    automatic_recall: dict[str, Any] | None,
    max_tokens: int,
) -> tuple[dict[str, Any] | None, dict[int, Any]]:
    """Bound recalled memories while retaining valid compact tool responses."""
    remaining_tokens = max(0, max_tokens)
    overrides: dict[int, Any] = {}
    latest_user_index = next(
        (
            index
            for index in reversed(range(len(contents)))
            if isinstance(contents[index], conversation.UserContent)
        ),
        -1,
    )
    current_turn: list[tuple[int, dict[str, Any]]] = []
    older: list[tuple[int, dict[str, Any]]] = []
    for index, content in enumerate(contents):
        if (
            not isinstance(content, conversation.ToolResultContent)
            or content.tool_name not in _RECALL_TOOL_NAMES
        ):
            continue
        payload = _recall_result_payload(_tool_result_data(content))
        if payload is None:
            continue
        target = current_turn if index > latest_user_index else older
        target.append((index, payload))

    def retain_or_release(index: int, payload: dict[str, Any]) -> None:
        nonlocal remaining_tokens
        memory_tokens = _recalled_memory_tokens(payload)
        if memory_tokens <= remaining_tokens:
            remaining_tokens -= memory_tokens
            return
        overrides[index] = {"result": _released_recall_payload(payload)}

    for index, payload in reversed(current_turn):
        retain_or_release(index, payload)

    bounded_automatic = automatic_recall
    if automatic_recall is not None and automatic_recall.get("memories"):
        memory_tokens = _recalled_memory_tokens(automatic_recall)
        if memory_tokens <= remaining_tokens:
            remaining_tokens -= memory_tokens
        elif "snapshot_id" in automatic_recall:
            bounded_automatic = _released_recall_payload(automatic_recall)
        else:
            bounded_automatic = None

    for index, payload in reversed(older):
        retain_or_release(index, payload)

    return bounded_automatic, overrides
