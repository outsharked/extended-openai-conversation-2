"""Pure helpers for conversation compaction and context budgeting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from typing import Any

from homeassistant.components import conversation

_INTRA_TURN_SUMMARY_NATIVE_KEY = "eoaic_intra_turn_summary"
_ROLLING_SUMMARY_NATIVE_KEY = "eoaic_rolling_summary"


@dataclass(slots=True)
class RequestTokenUsage:
    """Largest token usage reported across one agent turn."""

    prompt_tokens: int = 0
    total_tokens: int = 0

    def update(self, prompt_tokens: int, total_tokens: int) -> None:
        """Retain the largest reported usage across tool-call iterations."""
        self.prompt_tokens = max(self.prompt_tokens, prompt_tokens)
        self.total_tokens = max(self.total_tokens, total_tokens)


@dataclass(frozen=True, slots=True)
class IntraTurnReduction:
    """Completed active-turn tool exchanges eligible for reduction."""

    summary_groups: tuple[tuple[conversation.Content, ...], ...]
    archive_groups: tuple[tuple[conversation.Content, ...], ...]
    remove_indexes: frozenset[int]
    insert_at: int
    turn_id: str


@dataclass(frozen=True, slots=True)
class CompactionPlan:
    """History boundary and summary budget selected for compaction."""

    history_split_index: int
    recent_turns: int
    max_summary_tokens: int


def _tool_result_data(content: conversation.ToolResultContent) -> Any:
    """Return a tool result's data without using HA's deprecated accessor."""
    result = getattr(content, "result", None)
    if result is not None:
        return result.data
    return content.tool_result


def _content_for_compaction(content: conversation.Content) -> dict[str, Any]:
    """Return a text-only representation of one persisted conversation item."""
    item: dict[str, Any] = {"role": content.role}
    if isinstance(content, conversation.ToolResultContent):
        item["tool_name"] = content.tool_name
        item["result"] = _tool_result_data(content)
        return item

    text = getattr(content, "content", None)
    if text:
        item["content"] = text

    if isinstance(content, conversation.AssistantContent) and content.tool_calls:
        item["tool_calls"] = [
            {
                "name": tool_call.tool_name,
                "arguments": tool_call.tool_args,
            }
            for tool_call in content.tool_calls
        ]
    return item


def _compaction_transcript(contents: list[conversation.Content]) -> str:
    """Serialize conversation history as inert text for the summarizer."""
    return json.dumps(
        [_content_for_compaction(content) for content in contents],
        ensure_ascii=False,
        default=str,
    )


def _estimate_content_tokens(contents: list[conversation.Content]) -> int:
    """Estimate persisted-history tokens for calculating a summary budget."""
    transcript = _compaction_transcript(contents)
    return max(1, (len(transcript) + 3) // 4)


def _estimate_serialized_tokens(value: Any) -> int:
    """Conservatively estimate tokens in JSON-serializable request data."""
    serialized = json.dumps(value, ensure_ascii=False, default=str)
    return max(1, (len(serialized) + 3) // 4)


def _without_embedded_media(value: Any) -> Any:
    """Replace embedded media payloads with markers for text token estimates."""
    if isinstance(value, list):
        return [_without_embedded_media(item) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get("type") == "image_url":
        return {"type": "image_url", "image_url": "<embedded image>"}
    if value.get("type") == "input_audio":
        return {"type": "input_audio", "input_audio": "<embedded audio>"}
    return {key: _without_embedded_media(item) for key, item in value.items()}


def _group_compaction_contents(
    contents: list[conversation.Content],
) -> list[list[conversation.Content]]:
    """Group history into complete turns without splitting tool exchanges."""
    groups: list[list[conversation.Content]] = []
    current: list[conversation.Content] = []
    for content in contents:
        if isinstance(content, conversation.UserContent) and current:
            groups.append(current)
            current = []
        current.append(content)
    if current:
        groups.append(current)
    return groups


def _is_generated_compaction_summary(content: conversation.Content) -> bool:
    """Return whether content is a generated rolling or intra-turn summary."""
    return (
        isinstance(content, conversation.AssistantContent)
        and isinstance(content.native, Mapping)
        and (
            content.native.get(_ROLLING_SUMMARY_NATIVE_KEY) is True
            or content.native.get(_INTRA_TURN_SUMMARY_NATIVE_KEY) is True
        )
    )


def _intra_turn_reduction(
    contents: list[conversation.Content],
) -> IntraTurnReduction | None:
    """Return older completed tool exchanges from the active turn."""
    latest_user_index = next(
        (
            index
            for index in reversed(range(len(contents)))
            if isinstance(contents[index], conversation.UserContent)
        ),
        None,
    )
    if latest_user_index is None:
        return None

    latest_user = contents[latest_user_index]
    turn_id = latest_user.created.isoformat()
    exchange_ranges: list[tuple[int, int]] = []
    index = latest_user_index + 1
    while index < len(contents):
        content = contents[index]
        if (
            not isinstance(content, conversation.AssistantContent)
            or not content.tool_calls
        ):
            index += 1
            continue

        tool_call_ids = {tool_call.id for tool_call in content.tool_calls}
        result_ids: set[str] = set()
        end = index + 1
        while end < len(contents):
            result = contents[end]
            if not isinstance(result, conversation.ToolResultContent):
                break
            result_ids.add(result.tool_call_id)
            end += 1
        if tool_call_ids and tool_call_ids <= result_ids:
            exchange_ranges.append((index, end))
        index = end

    # Keep the newest completed exchange verbatim so the model can continue from it.
    compactable_ranges = exchange_ranges[:-1]
    if not compactable_ranges:
        return None

    groups_by_index: list[tuple[int, tuple[conversation.Content, ...], bool]] = []
    remove_indexes: set[int] = set()
    for index in range(latest_user_index + 1, len(contents)):
        content = contents[index]
        if (
            isinstance(content, conversation.AssistantContent)
            and isinstance(content.native, Mapping)
            and content.native.get(_INTRA_TURN_SUMMARY_NATIVE_KEY) is True
        ):
            groups_by_index.append((index, (content,), False))
            remove_indexes.add(index)

    for start, end in compactable_ranges:
        group = tuple(contents[start:end])
        groups_by_index.append((start, group, True))
        remove_indexes.update(range(start, end))

    groups_by_index.sort(key=lambda item: item[0])
    return IntraTurnReduction(
        summary_groups=tuple(group for _index, group, _archive in groups_by_index),
        archive_groups=tuple(
            group for _index, group, archive in groups_by_index if archive
        ),
        remove_indexes=frozenset(remove_indexes),
        insert_at=min(remove_indexes),
        turn_id=turn_id,
    )
