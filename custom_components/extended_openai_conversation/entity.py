"""Base entity for Extended OpenAI Conversation."""

from __future__ import annotations

import base64
import binascii
from collections.abc import AsyncGenerator, Collection, Mapping
from dataclasses import dataclass
import json
import logging
from mimetypes import guess_file_type
from pathlib import Path
import re
import traceback
from typing import TYPE_CHECKING, Any, Literal

from openai import AsyncClient, AsyncStream
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
import orjson
import probatio as vol

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, TemplateError
from homeassistant.helpers import device_registry as dr, llm, template
from homeassistant.helpers.entity import Entity
from homeassistant.util import slugify

from .const import (
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_CONTEXT_COMPACTION_RECENT_TURNS,
    CONF_CONTEXT_RECALL_MAX_TOKENS,
    CONF_CONTEXT_RECALL_SNAPSHOT_TTL_TIME,
    CONF_CONTEXT_RECALL_SNAPSHOT_TTL_TURNS,
    CONF_CONTEXT_THRESHOLD,
    CONF_CONTEXT_TRUNCATE_STRATEGY,
    CONF_EXTRA_BODY,
    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    CONF_MAX_TOKENS,
    CONF_MCP_AUTO_ADD_DEFAULT_PROMPT,
    CONF_MCP_AUTO_ADD_PERMANENT_TOOLS,
    CONF_MCP_DEFAULT_PROMPT_PLACEMENT,
    CONF_MCP_MEDIA_TYPES,
    CONF_MCP_PERMANENT_TOOLS_PLACEMENT,
    CONF_MCP_SERVERS,
    CONF_REASONING_EFFORT,
    CONF_SERVICE_TIER,
    CONF_SHORTEN_TOOL_CALL_ID,
    CONF_TEMPERATURE,
    CONF_TOOL_ERROR_REPORTING,
    CONF_TOP_P,
    CONTEXT_MANAGEMENT_FUNCTION_NAMES,
    CONTEXT_PRESERVATION_METHOD_EMBEDDINGS,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CONTEXT_COMPACTION_RECENT_TURNS,
    DEFAULT_CONTEXT_RECALL_MAX_TOKENS,
    DEFAULT_CONTEXT_RECALL_SNAPSHOT_TTL_TIME,
    DEFAULT_CONTEXT_RECALL_SNAPSHOT_TTL_TURNS,
    DEFAULT_CONTEXT_THRESHOLD,
    DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
    DEFAULT_EXTRA_BODY,
    DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MCP_AUTO_ADD_DEFAULT_PROMPT,
    DEFAULT_MCP_AUTO_ADD_PERMANENT_TOOLS,
    DEFAULT_MCP_DEFAULT_PROMPT_PLACEMENT,
    DEFAULT_MCP_MEDIA_TYPES,
    DEFAULT_MCP_PERMANENT_TOOLS_PLACEMENT,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_SERVICE_TIER,
    DEFAULT_SHORTEN_TOOL_CALL_ID,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOOL_ERROR_REPORTING,
    DEFAULT_TOP_P,
    DOMAIN,
    MCP_CONTEXT_PLACEMENT_COMBINED_SYSTEM,
    MCP_CONTEXT_PLACEMENT_DEDICATED_SYSTEM,
    MCP_CONTEXT_PLACEMENT_DISABLED,
    MCP_CONTEXT_PLACEMENT_USER,
    MCP_FUNCTION_NAME,
    TOOL_ERROR_REPORTING_DETAILED,
    TOOL_ERROR_REPORTING_RAW,
    TOOL_ERROR_REPORTING_SIMPLE,
)
from .context_management import (
    EmbeddingMemoryStore,
    RecallSnapshotScope,
    context_preservation_methods,
    embedding_history_retention,
    normalize_recall_snapshot_ttl_turns,
)
from .context_management.active_context import _bound_active_recalled_context
from .context_management.compaction import (
    RequestTokenUsage as _RequestTokenUsage,
    _estimate_serialized_tokens,
    _without_embedded_media,
)
from .context_management.manager import ContextManagementMixin
from .context_management.recall import ContextRecallMixin
from .context_management.recall_result import build_paginated_recall_result
from .exceptions import FunctionNotFound, TokenLengthExceededError
from .functions import get_function
from .functions.mcp import build_mcp_function_tool
from .helpers import (
    async_create_chat_completion,
    async_get_embedding_recall_prefix,
    async_get_mcp_default_prompt,
    get_model_config,
)
from .mcp_client import MCPContextRenderer
from .mcp_service_manager import MCPServiceManager
from .runtime import OpenAIServiceRuntime

if TYPE_CHECKING:
    from . import ExtendedOpenAIConfigEntry

_LOGGER = logging.getLogger(__name__)

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 20

_NATIVE_TOOL_CALL_EXTRA_CONTENT_KEY = "tool_call_extra_content_by_id"
_TOOL_ARGUMENT_PARSE_ERROR_KEY = "__eoaic_tool_argument_parse_error"
_MAX_MODEL_ERROR_DETAIL_LENGTH = 1000

_GENERIC_TOOL_ERROR_MESSAGES = {
    "call_service_error": "The Home Assistant service call was invalid.",
    "entity_access_denied": "Access to a requested entity was denied.",
    "entity_not_exposed": "A requested entity is not exposed to this agent.",
    "entity_not_found": "A requested entity was not found.",
    "function_not_found": "The requested tool was not found.",
    "invalid_arguments": "The tool arguments were invalid.",
    "native_not_found": "The requested native function was not found.",
    "service_not_found": "The requested Home Assistant service was not found.",
    "tool_execution_failed": "The tool could not be executed because of an internal error.",
    "tool_result_processing_failed": (
        "The tool ran, but its result could not be processed."
    ),
}

_SUPPORTED_IMAGE_MIME_TYPES = {
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/webp",
}

_SUPPORTED_AUDIO_MIME_TYPES: dict[str, Literal["mp3", "wav"]] = {
    "audio/mp3": "mp3",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
}


@dataclass(frozen=True, slots=True)
class MCPAgentContext:
    """MCP prompt values and automatic context for one agent request."""

    default_prompt: str = ""
    permanent_tools: str = ""
    permanent_tool_messages: tuple[str, ...] = ()


def _place_mcp_context(
    messages: list[ChatCompletionMessageParam],
    content: str,
    placement: str,
) -> None:
    """Place automatic MCP context in the configured message location."""
    if not content or placement == MCP_CONTEXT_PLACEMENT_DISABLED:
        return

    if placement == MCP_CONTEXT_PLACEMENT_DEDICATED_SYSTEM:
        messages.insert(
            0, ChatCompletionSystemMessageParam(role="system", content=content)
        )
        return

    target_role = (
        "system" if placement == MCP_CONTEXT_PLACEMENT_COMBINED_SYSTEM else "user"
    )
    if placement not in (
        MCP_CONTEXT_PLACEMENT_COMBINED_SYSTEM,
        MCP_CONTEXT_PLACEMENT_USER,
    ):
        raise ValueError(f"Unsupported MCP context placement: {placement}")

    for index, message in enumerate(messages):
        if message["role"] != target_role:
            continue
        existing = message.get("content")
        if isinstance(existing, str):
            merged_content = f"{content}\n\n{existing}" if existing else content
            if target_role == "system":
                messages[index] = ChatCompletionSystemMessageParam(
                    role="system", content=merged_content
                )
            else:
                messages[index] = ChatCompletionUserMessageParam(
                    role="user", content=merged_content
                )
            return
        if isinstance(existing, list) and target_role == "user":
            messages[index] = ChatCompletionUserMessageParam(
                role="user",
                content=[
                    ChatCompletionContentPartTextParam(type="text", text=content),
                    *existing,
                ],
            )
            return

    if target_role == "system":
        messages.insert(
            0,
            ChatCompletionSystemMessageParam(role="system", content=content),
        )
    else:
        messages.insert(
            0,
            ChatCompletionUserMessageParam(role="user", content=content),
        )


def _place_mcp_agent_context(
    messages: list[ChatCompletionMessageParam],
    context: MCPAgentContext,
    permanent_tools_placement: str,
    default_prompt_placement: str,
) -> None:
    """Place permanent tools and the default prompt independently."""
    placements = (
        (context.permanent_tools, permanent_tools_placement),
        (
            context.default_prompt,
            default_prompt_placement,
        ),
    )
    for content, placement in sorted(
        placements,
        key=lambda item: item[1] != MCP_CONTEXT_PLACEMENT_COMBINED_SYSTEM,
    ):
        _place_mcp_context(messages, content, placement)


def _shorten_tool_call_id(tool_call_id: str) -> str:
    """Shorten tool call ID to exactly 9 alphanumeric characters as Mistral requires."""
    import hashlib

    return hashlib.sha256(tool_call_id.encode()).hexdigest()[:9]


def _extract_extra_content(value: Any) -> dict[str, Any] | None:
    """Extract opaque provider-specific extra_content from an SDK response model."""
    extra_content = getattr(value, "extra_content", None)
    if isinstance(extra_content, dict):
        return extra_content

    model_extra = getattr(value, "model_extra", None)
    if isinstance(model_extra, dict):
        extra_content = model_extra.get("extra_content")
        if isinstance(extra_content, dict):
            return extra_content

    return None


def _get_tool_call_extra_content(
    native: Any, tool_call_id: str
) -> dict[str, Any] | None:
    """Return provider metadata stored in an AssistantContent native payload."""
    if not isinstance(native, dict):
        return None

    extra_content_by_id = native.get(_NATIVE_TOOL_CALL_EXTRA_CONTENT_KEY)
    if not isinstance(extra_content_by_id, dict):
        return None

    extra_content = extra_content_by_id.get(tool_call_id)
    return extra_content if isinstance(extra_content, dict) else None


def _adjust_schema(schema: dict[str, Any]) -> None:
    """Adjust the schema to be compatible with OpenAI API."""
    if schema["type"] == "object":
        schema.setdefault("strict", True)
        schema.setdefault("additionalProperties", False)
        if "properties" not in schema:
            return

        if "required" not in schema:
            schema["required"] = []

        # Ensure all properties are required
        for prop, prop_info in schema["properties"].items():
            _adjust_schema(prop_info)
            if prop not in schema["required"]:
                prop_info["type"] = [prop_info["type"], "null"]
                schema["required"].append(prop)

    elif schema["type"] == "array":
        if "items" not in schema:
            return

        _adjust_schema(schema["items"])


def _format_structured_output(
    schema: vol.Schema, llm_api: llm.APIInstance | None
) -> dict[str, Any]:
    """Format the schema to be compatible with OpenAI API."""
    result: dict[str, Any] = vol.to_openapi(
        schema,
        custom_serializer=(
            llm_api.custom_serializer if llm_api else llm.selector_serializer
        ),
    )

    _adjust_schema(result)

    return result


def _make_tool_result_content(
    agent_id: str, tool_call_id: str, tool_name: str, data: dict[str, Any]
) -> conversation.ToolResultContent:
    """Build a ToolResultContent across HA versions.

    HA 2026.10 replaced the `tool_result` dict field with `result`, an
    `llm.ToolResult` wrapper; older versions only accept `tool_result`.
    """
    tool_result_cls = getattr(llm, "ToolResult", None)
    payload: dict[str, Any] = (
        {"result": tool_result_cls(data=data)}
        if tool_result_cls is not None
        else {"tool_result": data}
    )
    return conversation.ToolResultContent(
        agent_id=agent_id, tool_call_id=tool_call_id, tool_name=tool_name, **payload
    )


def _tool_result_data(content: conversation.ToolResultContent) -> Any:
    """Return a tool result's data, avoiding HA 2026.10's deprecated accessor."""
    result = getattr(content, "result", None)
    if result is not None:
        return result.data
    return content.tool_result


def _bounded_model_error_detail(value: object) -> str:
    """Return a bounded diagnostic suitable for a model-facing result."""
    text = str(value)
    if len(text) <= _MAX_MODEL_ERROR_DETAIL_LENGTH:
        return text
    return f"{text[:_MAX_MODEL_ERROR_DETAIL_LENGTH]}…"


def _tool_error_identity(error: Exception) -> tuple[str, str]:
    """Return a stable code and generic message for a tool exception."""
    if isinstance(error, KeyError):
        return "invalid_arguments", _GENERIC_TOOL_ERROR_MESSAGES["invalid_arguments"]
    if isinstance(error, (TypeError, ValueError, vol.Invalid)):
        return "invalid_arguments", _GENERIC_TOOL_ERROR_MESSAGES["invalid_arguments"]
    code = re.sub(r"(?<!^)(?=[A-Z])", "_", type(error).__name__).lower()
    return code, _GENERIC_TOOL_ERROR_MESSAGES.get(
        code, "The tool could not be executed."
    )


def _tool_error_reason(error: Exception) -> str:
    """Return a useful bounded reason for a tool exception."""
    if isinstance(error, KeyError):
        missing_value = error.args[0] if error.args else "unknown"
        return _bounded_model_error_detail(
            f"Required value {missing_value!r} was not provided"
        )
    return _bounded_model_error_detail(error)


def _tool_error_payload(
    error: Exception,
    reporting: str,
    *,
    code: str | None = None,
    message: str | None = None,
    details: Mapping[str, Any] | None = None,
    exception_name: str | None = None,
    simple_reason: bool = True,
) -> dict[str, Any]:
    """Build a tool error using the configured model-facing verbosity."""
    if reporting == TOOL_ERROR_REPORTING_RAW:
        raw_traceback = details.get("traceback") if details is not None else None
        if not isinstance(raw_traceback, str):
            raw_traceback = "".join(traceback.format_exception(error))
        return {"error": raw_traceback}

    default_code, default_message = _tool_error_identity(error)
    payload: dict[str, Any] = {
        "code": code or default_code,
        "message": message or default_message,
    }
    reason = _tool_error_reason(error)
    if reporting == TOOL_ERROR_REPORTING_SIMPLE and simple_reason:
        payload["reason"] = reason
    elif reporting == TOOL_ERROR_REPORTING_DETAILED:
        structured_details = {
            key: value for key, value in (details or {}).items() if key != "traceback"
        }
        payload["details"] = {
            "exception": exception_name or type(error).__name__,
            "reason": reason,
            **structured_details,
        }
    return {"error": payload}


def _tool_error_content(
    agent_id: str,
    tool_input: llm.ToolInput,
    error: Exception,
    reporting: str,
    *,
    code: str | None = None,
    message: str | None = None,
    details: Mapping[str, Any] | None = None,
    exception_name: str | None = None,
    simple_reason: bool = True,
    log_traceback: bool = False,
) -> conversation.ToolResultContent:
    """Log and return a tool execution error to the model."""
    if log_traceback:
        _LOGGER.exception("Tool %s failed: %s", tool_input.tool_name, error)
    else:
        _LOGGER.error("Tool %s failed: %s", tool_input.tool_name, error)
    return _make_tool_result_content(
        agent_id=agent_id,
        tool_call_id=tool_input.id,
        tool_name=tool_input.tool_name,
        data=_tool_error_payload(
            error,
            reporting,
            code=code,
            message=message,
            details=details,
            exception_name=exception_name,
            simple_reason=simple_reason,
        ),
    )


def _extract_mcp_media_content(
    result: Any,
    supported_media_types: Collection[str] = DEFAULT_MCP_MEDIA_TYPES,
) -> tuple[
    Any,
    list[
        ChatCompletionContentPartImageParam | ChatCompletionContentPartInputAudioParam
    ],
]:
    """Extract standard MCP image and audio blocks for native model input."""
    if not isinstance(result, dict) or not isinstance(result.get("content"), list):
        return result, []

    enabled_media_types = frozenset(supported_media_types)
    media_parts: list[
        ChatCompletionContentPartImageParam | ChatCompletionContentPartInputAudioParam
    ] = []
    serialized_content: list[Any] = []
    for item in result["content"]:
        if not isinstance(item, dict) or item.get("type") not in {"audio", "image"}:
            serialized_content.append(item)
            continue

        content_type = item["type"]
        mime_type = item.get("mimeType")
        data = item.get("data")
        if content_type == "image" and mime_type == "image/jpg":
            mime_type = "image/jpeg"

        summary = dict(item)
        if content_type not in enabled_media_types:
            summary["media_error"] = {
                "code": "media_not_enabled",
                "message": (
                    "This media type is not enabled for the AI endpoint. The "
                    "original base64 media data is included in this tool result."
                ),
            }
            serialized_content.append(summary)
            continue
        if not isinstance(mime_type, str):
            summary["media_error"] = {
                "code": "unsupported_mime_type",
                "message": (
                    "The media could not be attached because its MIME type is "
                    "missing or invalid. The original media data is included."
                ),
            }
            serialized_content.append(summary)
            continue
        if content_type == "image":
            supported = mime_type in _SUPPORTED_IMAGE_MIME_TYPES
        else:
            supported = mime_type in _SUPPORTED_AUDIO_MIME_TYPES
        if not supported:
            summary["media_error"] = {
                "code": "unsupported_mime_type",
                "message": (
                    "This media type cannot be attached natively. The original "
                    "base64 media data is included in this tool result."
                ),
            }
            serialized_content.append(summary)
            continue
        if not isinstance(data, str):
            summary["media_error"] = {
                "code": "missing_data",
                "message": "The media could not be attached because it has no data.",
            }
            serialized_content.append(summary)
            continue
        try:
            base64.b64decode(data, validate=True)
        except binascii.Error, ValueError:
            summary["media_error"] = {
                "code": "invalid_base64_data",
                "message": (
                    "The media could not be attached because its data is not valid "
                    "base64. The original data is included in this tool result."
                ),
            }
            serialized_content.append(summary)
            continue

        summary.pop("data", None)
        summary["mimeType"] = mime_type
        serialized_content.append(summary)
        if content_type == "image":
            media_parts.append(
                ChatCompletionContentPartImageParam(
                    type="image_url",
                    image_url={"url": f"data:{mime_type};base64,{data}"},
                )
            )
        else:
            media_parts.append(
                ChatCompletionContentPartInputAudioParam(
                    type="input_audio",
                    input_audio={
                        "data": data,
                        "format": _SUPPORTED_AUDIO_MIME_TYPES[mime_type],
                    },
                )
            )

    if not media_parts and serialized_content == result["content"]:
        return result, []

    serialized_result = dict(result)
    serialized_result["content"] = serialized_content
    return serialized_result, media_parts


def _messages_contain_media(messages: list[ChatCompletionMessageParam]) -> bool:
    """Return whether messages contain native image or audio input data."""
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        if any(
            isinstance(part, dict) and part.get("type") in {"image_url", "input_audio"}
            for part in content
        ):
            return True
    return False


def _render_extra_body(
    extra_body_raw: str, hass: HomeAssistant, model: str
) -> dict[str, Any] | None:
    """Render the extra_body template option to a dict, or None if unset/invalid.

    extra_body_raw is a Jinja-templatable JSON string (e.g. from the
    CONF_EXTRA_BODY option); an empty/whitespace-only string means the
    feature is disabled. A template or JSON error is logged as a warning
    and treated as unset, rather than failing the conversation turn.
    """
    if not extra_body_raw.strip():
        return None

    try:
        rendered = template.Template(extra_body_raw, hass).async_render(
            parse_result=False
        )
    except TemplateError as err:
        _LOGGER.warning("Invalid extra_body template for %s, ignoring: %s", model, err)
        return None

    if not rendered.strip():
        return None

    try:
        parsed: dict[str, Any] = json.loads(rendered)
        return parsed
    except json.JSONDecodeError as err:
        _LOGGER.warning("Invalid extra_body JSON for %s, ignoring: %s", model, err)
        return None


def encode_attachments(
    chat_content: list[conversation.Content],
) -> dict[int, list[ChatCompletionContentPartImageParam]]:
    """Read and base64-encode attachments. BLOCKING - run in an executor.

    Returns a map of {index in chat_content: list of image content parts}.
    """
    encoded: dict[int, list[ChatCompletionContentPartImageParam]] = {}

    for index, content in enumerate(chat_content):
        attachments = getattr(content, "attachments", None)
        if content.role != "user" or not attachments:
            continue

        parts: list[ChatCompletionContentPartImageParam] = []
        for attachment in attachments:
            file_path = Path(attachment.path)
            if not file_path.exists():
                raise HomeAssistantError(
                    f"Attachment does not exist: {file_path.as_posix()}"
                )

            mime_type = attachment.mime_type or guess_file_type(file_path)[0]
            if not mime_type or not mime_type.startswith("image/"):
                raise HomeAssistantError(
                    f"Unsupported attachment type {mime_type or 'unknown'} "
                    f"for {file_path.as_posix()}"
                )
            if mime_type == "image/jpg":
                mime_type = "image/jpeg"

            b64 = base64.b64encode(file_path.read_bytes()).decode("utf-8")
            parts.append(
                ChatCompletionContentPartImageParam(
                    type="image_url",
                    image_url={"url": f"data:{mime_type};base64,{b64}"},
                )
            )

        if parts:
            encoded[index] = parts

    return encoded


def _convert_content_to_param(
    chat_content: list[conversation.Content],
    shorten_tool_call_id: bool = False,
    attachment_parts: dict[int, list[ChatCompletionContentPartImageParam]]
    | None = None,
    tool_media_parts: dict[
        str,
        list[
            ChatCompletionContentPartImageParam
            | ChatCompletionContentPartInputAudioParam
        ],
    ]
    | None = None,
    tool_result_overrides: dict[int, Any] | None = None,
) -> list[ChatCompletionMessageParam]:
    """Convert chat log content to OpenAI message format."""
    messages: list[ChatCompletionMessageParam] = []
    attachment_parts = attachment_parts or {}
    tool_media_parts = tool_media_parts or {}
    tool_result_overrides = tool_result_overrides or {}
    pending_tool_media: list[
        ChatCompletionContentPartTextParam
        | ChatCompletionContentPartImageParam
        | ChatCompletionContentPartInputAudioParam
    ] = []

    for index, content in enumerate(chat_content):
        if content.role != "tool_result" and pending_tool_media:
            messages.append({"role": "user", "content": pending_tool_media})
            pending_tool_media = []

        if content.role == "system":
            messages.append({"role": "system", "content": content.content})
        elif content.role == "user":
            if (attachment_image_parts := attachment_parts.get(index)) is None:
                messages.append({"role": "user", "content": content.content})
            else:
                # Multipart form: text first, then one image_url block per
                # attachment.
                multipart: list[
                    ChatCompletionContentPartTextParam
                    | ChatCompletionContentPartImageParam
                ] = []
                if content.content:
                    multipart.append(
                        ChatCompletionContentPartTextParam(
                            type="text", text=content.content
                        )
                    )
                multipart.extend(attachment_image_parts)
                user_msg: ChatCompletionUserMessageParam = {
                    "role": "user",
                    "content": multipart,
                }
                messages.append(user_msg)
        elif content.role == "assistant":
            msg: ChatCompletionAssistantMessageParam = {"role": "assistant"}
            if content.content:
                msg["content"] = content.content
            if content.tool_calls:
                serialized_tool_calls: list[Any] = []
                for tool_call in content.tool_calls:
                    serialized_tool_call: dict[str, Any] = {
                        "id": (
                            _shorten_tool_call_id(tool_call.id)
                            if shorten_tool_call_id
                            else tool_call.id
                        ),
                        "type": "function",
                        "function": {
                            "name": tool_call.tool_name,
                            "arguments": json.dumps(tool_call.tool_args),
                        },
                    }

                    extra_content = _get_tool_call_extra_content(
                        content.native, tool_call.id
                    )
                    if extra_content is not None:
                        serialized_tool_call["extra_content"] = extra_content

                    serialized_tool_calls.append(serialized_tool_call)

                msg["tool_calls"] = serialized_tool_calls
            # Some OpenAI-compatible APIs (like Mistral) reject empty tool_calls arrays
            # Remove tool_calls field if it's an empty array to maintain compatibility
            if msg.get("tool_calls") == []:
                msg.pop("tool_calls", None)
            # Some OpenAI-compatible APIs (vLLM, SGLang, etc.) reject assistant
            # messages missing the "content" key entirely; ensure it's always
            # present, matching the Mistral empty-tool_calls guard above
            if "content" not in msg:
                msg["content"] = None
            messages.append(msg)
        elif content.role == "tool_result":
            tool_result = tool_result_overrides.get(index, _tool_result_data(content))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": _shorten_tool_call_id(content.tool_call_id)
                    if shorten_tool_call_id
                    else content.tool_call_id,
                    "content": orjson.dumps(tool_result).decode(),
                }
            )
            if media_parts := tool_media_parts.get(content.tool_call_id):
                pending_tool_media.append(
                    ChatCompletionContentPartTextParam(
                        type="text",
                        text=f"Media returned by tool {content.tool_name}:",
                    )
                )
                pending_tool_media.extend(media_parts)

    if pending_tool_media:
        messages.append({"role": "user", "content": pending_tool_media})

    return messages


class ExtendedOpenAIBaseLLMEntity(ContextRecallMixin, ContextManagementMixin, Entity):
    """Extended OpenAI base entity."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(
        self, entry: ExtendedOpenAIConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="OpenAI",
            model=subentry.data.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL),
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        # Request parameters the API has rejected, per model, so later
        # requests skip them instead of failing first
        self._unsupported_params: dict[str, set[str]] = {}

    @property
    def _client(self) -> AsyncClient:
        """Return the OpenAI client."""
        runtime = self.entry.runtime_data
        if not isinstance(runtime, OpenAIServiceRuntime):
            raise TypeError("An OpenAI entity requires an OpenAI client")
        return runtime.client

    @property
    def _embedding_runtime(
        self,
    ) -> tuple[AsyncClient, EmbeddingMemoryStore]:
        """Return the configured embeddings client and memory store."""
        runtime = self.entry.runtime_data
        if (
            not isinstance(runtime, OpenAIServiceRuntime)
            or runtime.embedding_client is None
            or runtime.embedding_memory is None
        ):
            raise HomeAssistantError("Embedding retrieval is not configured")
        return runtime.embedding_client, runtime.embedding_memory

    @property
    def _context_memory(self) -> EmbeddingMemoryStore:
        """Return the lossless conversation archive."""
        runtime = self.entry.runtime_data
        if (
            not isinstance(runtime, OpenAIServiceRuntime)
            or runtime.embedding_memory is None
        ):
            raise HomeAssistantError("Conversation retrieval is not configured")
        return runtime.embedding_memory

    def _get_mcp_function_tools(self) -> list[dict[str, Any]]:
        """Return the MCP dispatcher when servers are assigned to this agent."""
        server_ids = self.subentry.data.get(CONF_MCP_SERVERS, []) or []
        if not server_ids:
            return []
        return [build_mcp_function_tool(server_ids)]

    def _get_internal_function_tools(self) -> list[dict[str, Any]]:
        """Return integration-managed tools enabled for this agent."""
        return (
            self._get_mcp_function_tools() + self._get_context_recall_function_tools()
        )

    async def _async_get_mcp_agent_context(self, language: str) -> MCPAgentContext:
        """Build template values and automatic context for assigned MCP services."""
        server_ids = self.subentry.data.get(CONF_MCP_SERVERS, []) or []
        if not server_ids:
            return MCPAgentContext()

        services = MCPServiceManager(self.hass).scope(server_ids).list_loaded()
        if not services:
            return MCPAgentContext()

        permanent_tool_messages: list[str] = []
        for service in services:
            context = MCPContextRenderer.render_permanent_context(
                service.config,
            )
            if context.content:
                permanent_tool_messages.append(context.content)

        default_prompt = await async_get_mcp_default_prompt(self.hass, language)
        return MCPAgentContext(
            default_prompt=default_prompt,
            permanent_tools="\n\n".join(permanent_tool_messages),
            permanent_tool_messages=tuple(permanent_tool_messages),
        )

    def _mcp_context_messages(
        self, context: MCPAgentContext
    ) -> list[ChatCompletionSystemMessageParam]:
        """Return the separately added MCP system messages for a conversation."""
        messages: list[ChatCompletionSystemMessageParam] = []
        if context.default_prompt and self.subentry.data.get(
            CONF_MCP_AUTO_ADD_DEFAULT_PROMPT,
            DEFAULT_MCP_AUTO_ADD_DEFAULT_PROMPT,
        ):
            messages.append(
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=context.default_prompt,
                )
            )
        if self.subentry.data.get(
            CONF_MCP_AUTO_ADD_PERMANENT_TOOLS,
            DEFAULT_MCP_AUTO_ADD_PERMANENT_TOOLS,
        ):
            messages.extend(
                ChatCompletionSystemMessageParam(role="system", content=content)
                for content in context.permanent_tool_messages
            )
        return messages

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        function_tools: list[dict[str, Any]],
        exposed_entities: list[dict[str, Any]],
        llm_context: llm.LLMContext | None = None,
        structure_name: str | None = None,
        structure: vol.Schema | None = None,
        mcp_context: MCPAgentContext | None = None,
    ) -> None:
        """Generate an answer for the chat log with streaming support."""
        options = self.subentry.data
        model = options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        max_function_calls = options.get(
            CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
            DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
        )
        shorten_tool_call_id = options.get(
            CONF_SHORTEN_TOOL_CALL_ID,
            DEFAULT_SHORTEN_TOOL_CALL_ID,
        )

        # Get model-specific configuration
        model_config = get_model_config(model, self.entry.data.get(CONF_BASE_URL))

        language = (
            llm_context.language
            if llm_context is not None and llm_context.language is not None
            else self.hass.config.language
        )
        if mcp_context is None:
            mcp_context = await self._async_get_mcp_agent_context(language)

        permanent_tools_placement = self.subentry.data.get(
            CONF_MCP_PERMANENT_TOOLS_PLACEMENT
        )
        default_prompt_placement = self.subentry.data.get(
            CONF_MCP_DEFAULT_PROMPT_PLACEMENT
        )
        tool_media_parts: dict[
            str,
            list[
                ChatCompletionContentPartImageParam
                | ChatCompletionContentPartInputAudioParam
            ],
        ] = {}
        retrieval_enabled = (
            CONTEXT_PRESERVATION_METHOD_EMBEDDINGS
            in context_preservation_methods(
                options.get(
                    CONF_CONTEXT_TRUNCATE_STRATEGY,
                    DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
                )
            )
        )
        recall_prefix = (
            await async_get_embedding_recall_prefix(self.hass, language)
            if retrieval_enabled
            else ""
        )
        recall_max_tokens = int(
            options.get(
                CONF_CONTEXT_RECALL_MAX_TOKENS,
                DEFAULT_CONTEXT_RECALL_MAX_TOKENS,
            )
        )
        recall_snapshot_ttl_turns = normalize_recall_snapshot_ttl_turns(
            options.get(
                CONF_CONTEXT_RECALL_SNAPSHOT_TTL_TURNS,
                DEFAULT_CONTEXT_RECALL_SNAPSHOT_TTL_TURNS,
            )
        )
        recall_snapshot_ttl_time = embedding_history_retention(
            options.get(
                CONF_CONTEXT_RECALL_SNAPSHOT_TTL_TIME,
                DEFAULT_CONTEXT_RECALL_SNAPSHOT_TTL_TIME,
            )
        )
        runtime = self.entry.runtime_data
        conversation_id = chat_log.conversation_id
        recall_snapshot_scope = (
            RecallSnapshotScope(
                agent_id=self.subentry.subentry_id,
                conversation_id=conversation_id,
            )
            if isinstance(runtime, OpenAIServiceRuntime)
            and isinstance(conversation_id, str)
            and conversation_id
            else None
        )
        if isinstance(runtime, OpenAIServiceRuntime):
            runtime.recall_snapshots.advance_turn(self.subentry.subentry_id)

        (
            recalled_memories,
            close_recalled_memories,
        ) = await self._async_recall_embedding_memories(chat_log)
        recall_result: dict[str, Any] | None = None
        automatic_snapshot_id: str | None = None

        def _refresh_recall_result() -> None:
            """Rebuild automatic recall and discard any superseded snapshot."""
            nonlocal automatic_snapshot_id, recall_result
            if (
                automatic_snapshot_id is not None
                and recall_snapshot_scope is not None
                and isinstance(runtime, OpenAIServiceRuntime)
            ):
                runtime.recall_snapshots.discard(
                    recall_snapshot_scope, automatic_snapshot_id
                )
            automatic_snapshot_id = None
            recall_result = None
            if not recalled_memories:
                return
            recall_result = build_paginated_recall_result(
                recalled_memories,
                recall_max_tokens,
                close_matches=close_recalled_memories,
                snapshot_store=(
                    runtime.recall_snapshots
                    if isinstance(runtime, OpenAIServiceRuntime)
                    else None
                ),
                snapshot_scope=recall_snapshot_scope,
                snapshot_ttl_turns=recall_snapshot_ttl_turns,
                snapshot_ttl_time=recall_snapshot_ttl_time,
                include_relevance=False,
            )
            snapshot_id = recall_result.get("snapshot_id")
            if isinstance(snapshot_id, str):
                automatic_snapshot_id = snapshot_id

        _refresh_recall_result()

        async def _build_request_messages() -> list[ChatCompletionMessageParam]:
            """Build messages again after history has potentially changed."""
            bounded_recall_result, tool_result_overrides = (
                _bound_active_recalled_context(
                    chat_log.content,
                    recall_result,
                    recall_max_tokens,
                )
            )
            attachment_parts = await self.hass.async_add_executor_job(
                encode_attachments, chat_log.content
            )
            converted_messages = _convert_content_to_param(
                chat_log.content,
                shorten_tool_call_id,
                attachment_parts,
                tool_media_parts,
                tool_result_overrides,
            )
            self._place_embedding_recall(
                converted_messages,
                bounded_recall_result,
                recall_prefix,
            )
            if permanent_tools_placement is None and default_prompt_placement is None:
                context_messages = self._mcp_context_messages(mcp_context)
                messages: list[ChatCompletionMessageParam] = [
                    *context_messages,
                    *converted_messages,
                ]
                return messages

            _place_mcp_agent_context(
                converted_messages,
                mcp_context,
                permanent_tools_placement or DEFAULT_MCP_PERMANENT_TOOLS_PLACEMENT,
                default_prompt_placement or DEFAULT_MCP_DEFAULT_PROMPT_PLACEMENT,
            )
            return converted_messages

        messages = await _build_request_messages()

        # Build functions list from custom functions
        tools: list[ChatCompletionToolParam] = [
            ChatCompletionToolParam(
                type="function",
                function=func_spec["spec"],
            )
            for func_spec in function_tools
        ]

        # Build API parameters based on model configuration
        api_kwargs: dict[str, Any] = {
            "model": model,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        # Add token limit parameter based on model support
        max_tokens = options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        if model_config["supports_max_completion_tokens"]:
            api_kwargs["max_completion_tokens"] = max_tokens
        elif model_config["supports_max_tokens"]:
            api_kwargs["max_tokens"] = max_tokens

        # Add top_p if supported
        if model_config["supports_top_p"]:
            api_kwargs["top_p"] = options.get(CONF_TOP_P, DEFAULT_TOP_P)

        # Add temperature if supported
        if model_config["supports_temperature"]:
            api_kwargs["temperature"] = options.get(
                CONF_TEMPERATURE, DEFAULT_TEMPERATURE
            )

        # Add reasoning_effort if supported (o1, o3, o4, gpt-5 models)
        if model_config.get("supports_reasoning_effort"):
            api_kwargs["reasoning_effort"] = options.get(
                CONF_REASONING_EFFORT, DEFAULT_REASONING_EFFORT
            )

        # Add service_tier if supported (o3, o4, gpt-5 models)
        if model_config.get("supports_service_tier"):
            api_kwargs["service_tier"] = options.get(
                CONF_SERVICE_TIER, DEFAULT_SERVICE_TIER
            )

        # Add extra_body if configured - passthrough for OpenAI-compatible
        # backends that accept extra request-body fields (ollama, llama.cpp,
        # vLLM, LM Studio, etc.). E.g. {"chat_template_kwargs":
        # {"enable_thinking": false}} to disable Qwen3 reasoning, or
        # {"cache_prompt": true} for llama.cpp prompt caching.
        extra_body = _render_extra_body(
            options.get(CONF_EXTRA_BODY, DEFAULT_EXTRA_BODY) or "", self.hass, model
        )
        if extra_body is not None:
            api_kwargs["extra_body"] = extra_body

        # Add structured output format if provided
        if structure is not None:
            api_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": slugify(structure_name),
                    "strict": True,
                    "schema": _format_structured_output(structure, chat_log.llm_api),
                },
            }

        # Add tools if available
        tool_kwargs: dict[str, Any] = {}
        if tools:
            tool_kwargs["tools"] = tools
            tool_kwargs["tool_choice"] = "auto"

        # To prevent infinite loops, we limit the number of iterations
        token_usage = _RequestTokenUsage()
        for n_requests in range(MAX_TOOL_ITERATIONS):
            # Update tool_choice based on function call count
            # -1 means unlimited function calls
            if tools and 0 <= max_function_calls <= n_requests:
                tool_kwargs["tool_choice"] = "none"

            threshold = int(
                options.get(CONF_CONTEXT_THRESHOLD, DEFAULT_CONTEXT_THRESHOLD)
            )
            recent_turns_override = max(
                0,
                int(
                    options.get(
                        CONF_CONTEXT_COMPACTION_RECENT_TURNS,
                        DEFAULT_CONTEXT_COMPACTION_RECENT_TURNS,
                    )
                ),
            )
            while True:
                estimated_prompt_tokens = _estimate_serialized_tokens(
                    _without_embedded_media(
                        {"messages": messages, "tools": tool_kwargs.get("tools", [])}
                    )
                )
                if estimated_prompt_tokens + int(max_tokens) <= threshold:
                    break

                if recalled_memories:
                    recalled_memories.pop()
                    _refresh_recall_result()
                    messages = await _build_request_messages()
                    continue

                history_changed = await self._truncate_message_history(
                    chat_log,
                    _RequestTokenUsage(
                        prompt_tokens=estimated_prompt_tokens,
                        total_tokens=estimated_prompt_tokens + int(max_tokens),
                    ),
                    preserve_current_turn=True,
                    recent_turns_override=recent_turns_override,
                    language=language,
                )
                intra_turn_changed = False
                if not history_changed:
                    intra_turn_changed = await self._reduce_intra_turn_history(
                        chat_log,
                        _RequestTokenUsage(
                            prompt_tokens=estimated_prompt_tokens,
                            total_tokens=estimated_prompt_tokens + int(max_tokens),
                        ),
                        language,
                    )
                    history_changed = intra_turn_changed
                if history_changed:
                    if retrieval_enabled and not intra_turn_changed:
                        (
                            recalled_memories,
                            close_recalled_memories,
                        ) = await self._async_recall_embedding_memories(chat_log)
                        _refresh_recall_result()
                    messages = await _build_request_messages()
                    recent_turns_override = max(0, recent_turns_override - 1)
                    continue

                raise HomeAssistantError(
                    "The current conversation turn and request context exceed the "
                    "configured context threshold"
                )

            if _messages_contain_media(messages):
                _LOGGER.info(
                    "Prompt for %s contains %s messages; media data omitted",
                    model,
                    len(messages),
                )
            else:
                _LOGGER.info("Prompt for %s: %s", model, json.dumps(messages))

            stream = await async_create_chat_completion(
                self._client,
                self._unsupported_params.setdefault(model, set()),
                messages=messages,
                **api_kwargs,
                **tool_kwargs,
            )

            # Process stream and collect tool calls
            pending_tool_calls: list[llm.ToolInput] = []

            async for content in chat_log.async_add_delta_content_stream(
                self.entity_id,
                self._transform_stream(chat_log, stream, token_usage),
            ):
                if (
                    isinstance(content, conversation.AssistantContent)
                    and content.tool_calls
                ):
                    pending_tool_calls.extend(content.tool_calls)

            if pending_tool_calls:
                _LOGGER.info("Response Tool Calls %s", pending_tool_calls)

            # Execute custom functions
            for tool_input in pending_tool_calls:
                error_reporting = self.subentry.data.get(
                    CONF_TOOL_ERROR_REPORTING, DEFAULT_TOOL_ERROR_REPORTING
                )
                function_tool = next(
                    (
                        f
                        for f in (function_tools)
                        if f["spec"]["name"] == tool_input.tool_name
                    ),
                    None,
                )

                if function_tool is None:
                    tool_result_content = _tool_error_content(
                        self.entity_id,
                        tool_input,
                        FunctionNotFound(tool_input.tool_name),
                        error_reporting,
                    )
                    chat_log.async_add_assistant_content_without_tools(tool_result_content)
                    continue

                tool_result_content, media_parts = await self._execute_function_tool(
                    function_tool,
                    tool_input,
                    chat_log,
                    llm_context,
                    exposed_entities,
                )

                chat_log.async_add_assistant_content_without_tools(tool_result_content)
                if media_parts:
                    tool_media_parts[tool_input.id] = media_parts

            # Update messages for the next tool-call iteration.
            messages = await _build_request_messages()

            # Check if we need to continue (if there are pending tool results)
            if not chat_log.unresponded_tool_results:
                break

        if token_usage.total_tokens > options.get(
            CONF_CONTEXT_THRESHOLD, DEFAULT_CONTEXT_THRESHOLD
        ):
            await self._truncate_message_history(
                chat_log, token_usage, language=language
            )

    async def _transform_stream(
        self,
        chat_log: conversation.ChatLog,
        result: AsyncStream[ChatCompletionChunk],
        token_usage: _RequestTokenUsage | None = None,
    ) -> AsyncGenerator[
        conversation.AssistantContentDeltaDict | conversation.ToolResultContentDeltaDict
    ]:
        """Transform OpenAI stream to Home Assistant format."""
        current_tool_calls: dict[int, dict[str, Any]] = {}
        # Collect provider metadata across the whole response and attach it
        # once. HA only allows AssistantContent.native to be set a single time
        # per message; yielding native on every tool_calls finish_reason would
        # raise RuntimeError on loosely compatible servers that emit that
        # finish reason more than once.
        tool_call_extra_content_by_id: dict[str, dict[str, Any]] = {}
        first_chunk = True

        async for chunk in result:
            _LOGGER.debug("Received chunk: %s", chunk)

            # Signal new assistant message on first chunk
            if first_chunk:
                yield {"role": "assistant"}
                first_chunk = False

            if not chunk.choices:
                # Track usage from final chunk if available
                if chunk.usage:
                    if token_usage is not None:
                        token_usage.update(
                            chunk.usage.prompt_tokens,
                            chunk.usage.total_tokens,
                        )
                    chat_log.async_trace(
                        {
                            "stats": {
                                "input_tokens": chunk.usage.prompt_tokens,
                                "output_tokens": chunk.usage.completion_tokens,
                            }
                        }
                    )
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if delta.content:
                # Ensure content is a string (Mistral might return unexpected types)
                content_value = delta.content
                if not isinstance(content_value, str):
                    _LOGGER.warning(
                        "Received non-string content from API: %s (type: %s)",
                        content_value,
                        type(content_value),
                    )
                    content_value = str(content_value) if content_value else ""
                if content_value:
                    yield {"content": content_value}

            if delta.tool_calls:
                for tool_call_delta in delta.tool_calls:
                    idx = tool_call_delta.index
                    if idx not in current_tool_calls:
                        current_tool_calls[idx] = {
                            "id": tool_call_delta.id or "",
                            "name": "",
                            "arguments": "",
                        }

                    if tool_call_delta.id:
                        current_tool_calls[idx]["id"] = tool_call_delta.id

                    extra_content = _extract_extra_content(tool_call_delta)
                    if extra_content is not None:
                        current_tool_calls[idx]["extra_content"] = extra_content

                    if tool_call_delta.function:
                        if tool_call_delta.function.name:
                            current_tool_calls[idx]["name"] = (
                                tool_call_delta.function.name
                            )
                        if tool_call_delta.function.arguments:
                            current_tool_calls[idx]["arguments"] += (
                                tool_call_delta.function.arguments
                            )

            if current_tool_calls and (choice.finish_reason in {"tool_calls", "stop"}):
                # Yield all accumulated tool calls (marked as external since we handle them ourselves)
                tool_calls_list = []
                for idx in sorted(current_tool_calls.keys()):
                    tool_call = current_tool_calls[idx]
                    try:
                        args = json.loads(tool_call["arguments"])
                    except json.JSONDecodeError as err:
                        args = {
                            _TOOL_ARGUMENT_PARSE_ERROR_KEY: {
                                "exception": type(err).__name__,
                                "reason": err.msg,
                                "line": err.lineno,
                                "column": err.colno,
                                "position": err.pos,
                                "arguments_preview": _bounded_model_error_detail(
                                    tool_call["arguments"]
                                ),
                                "traceback": "".join(
                                    traceback.format_exception(err)
                                ),
                            }
                        }
                    if not isinstance(args, dict):
                        args = {
                            _TOOL_ARGUMENT_PARSE_ERROR_KEY: {
                                "exception": "TypeError",
                                "reason": "Tool arguments must be a JSON object",
                                "parsed_type": type(args).__name__,
                            }
                        }

                    extra_content = tool_call.get("extra_content")
                    if tool_call["id"] and extra_content is not None:
                        tool_call_extra_content_by_id[tool_call["id"]] = extra_content

                    tool_calls_list.append(
                        llm.ToolInput(
                            id=tool_call["id"],
                            tool_name=tool_call["name"],
                            tool_args=args,
                            external=True,  # Mark as external so ChatLog doesn't try to execute
                        )
                    )
                if tool_calls_list:
                    yield {"tool_calls": tool_calls_list}
                current_tool_calls.clear()
            if choice.finish_reason == "length":
                raise TokenLengthExceededError(
                    self.subentry.data.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
                )

            if choice.finish_reason == "stop":
                break

        if not first_chunk and tool_call_extra_content_by_id:
            yield {
                "native": {
                    _NATIVE_TOOL_CALL_EXTRA_CONTENT_KEY: tool_call_extra_content_by_id
                }
            }

    async def _execute_function_tool(
        self,
        function_tool: dict[str, Any],
        tool_input: llm.ToolInput,
        chat_log: conversation.ChatLog,
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> tuple[
        conversation.ToolResultContent,
        list[
            ChatCompletionContentPartImageParam
            | ChatCompletionContentPartInputAudioParam
        ],
    ]:
        """Execute a custom function."""
        arguments: dict[str, Any] = tool_input.tool_args
        error_reporting = self.subentry.data.get(
            CONF_TOOL_ERROR_REPORTING, DEFAULT_TOOL_ERROR_REPORTING
        )
        parse_error = arguments.pop(_TOOL_ARGUMENT_PARSE_ERROR_KEY, None)
        if isinstance(parse_error, Mapping):
            reason = str(parse_error.get("reason", "Invalid JSON arguments"))
            return (
                _tool_error_content(
                    self.entity_id,
                    tool_input,
                    ValueError(reason),
                    error_reporting,
                    code="invalid_arguments",
                    message=_GENERIC_TOOL_ERROR_MESSAGES["invalid_arguments"],
                    details={
                        key: value
                        for key, value in parse_error.items()
                        if key not in {"exception", "reason"}
                    },
                    exception_name=str(parse_error.get("exception", "ValueError")),
                ),
                [],
            )
        function_config = function_tool["function"]
        if function_config["type"] in CONTEXT_MANAGEMENT_FUNCTION_NAMES:
            function_config = {
                **function_config,
                "conversation_id": chat_log.conversation_id,
            }
        media_parts: list[
            ChatCompletionContentPartImageParam
            | ChatCompletionContentPartInputAudioParam
        ] = []

        try:
            function = get_function(function_config["type"])

            if (
                function_config["type"] not in CONTEXT_MANAGEMENT_FUNCTION_NAMES
                and self.should_run_in_background(arguments)
            ):
                # create a delayed function and execute in background
                function_config = self.get_delayed_function_config(
                    function_config, arguments
                )
                function = get_function(function_config["type"])
                self.entry.async_create_task(
                    self.hass,
                    function.execute(
                        self.hass,
                        function_config,
                        arguments,
                        llm_context,
                        exposed_entities,
                    ),
                )
                result = "Scheduled"
            else:
                result = await function.execute(
                    self.hass,
                    function_config,
                    arguments,
                    llm_context,
                    exposed_entities,
                )
        except (HomeAssistantError, KeyError, TypeError, ValueError, vol.Invalid) as err:
            return (
                _tool_error_content(
                    self.entity_id, tool_input, err, error_reporting
                ),
                media_parts,
            )
        except Exception as err:  # pylint: disable=broad-except
            return (
                _tool_error_content(
                    self.entity_id,
                    tool_input,
                    err,
                    error_reporting,
                    code="tool_execution_failed",
                    message=_GENERIC_TOOL_ERROR_MESSAGES["tool_execution_failed"],
                    simple_reason=False,
                    log_traceback=True,
                ),
                media_parts,
            )

        try:
            if function_config["type"] == MCP_FUNCTION_NAME:
                result, media_parts = _extract_mcp_media_content(
                    result,
                    self.entry.data.get(
                        CONF_MCP_MEDIA_TYPES, DEFAULT_MCP_MEDIA_TYPES
                    ),
                )

            result_data = {
                "result": (
                    result
                    if function_config["type"] in CONTEXT_MANAGEMENT_FUNCTION_NAMES
                    else str(result)
                )
            }
            orjson.dumps(result_data)
            return (
                _make_tool_result_content(
                    agent_id=self.entity_id,
                    tool_call_id=tool_input.id,
                    tool_name=tool_input.tool_name,
                    data=result_data,
                ),
                media_parts,
            )
        except Exception as err:  # pylint: disable=broad-except
            return (
                _tool_error_content(
                    self.entity_id,
                    tool_input,
                    err,
                    error_reporting,
                    code="tool_result_processing_failed",
                    message=_GENERIC_TOOL_ERROR_MESSAGES[
                        "tool_result_processing_failed"
                    ],
                    log_traceback=True,
                ),
                [],
            )

    def should_run_in_background(self, arguments: dict[str, Any]) -> bool:
        """Check if function needs delay."""
        return isinstance(arguments, dict) and arguments.get("delay") is not None

    def get_delayed_function_config(
        self, function_config: dict[str, Any], arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute function with delay."""
        # create a composite function with delay in script function
        return {
            "type": "composite",
            "sequence": [
                {
                    "type": "script",
                    "sequence": [{"delay": arguments["delay"]}],
                },
                function_config,
            ],
        }
