"""Base entity for Extended OpenAI Conversation."""

from __future__ import annotations

import base64
import binascii
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass
import json
import logging
from mimetypes import guess_file_type
from pathlib import Path
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
    CONF_CONTEXT_COMPACTION_PROMPT,
    CONF_CONTEXT_COMPACTION_RECENT_TURNS,
    CONF_CONTEXT_COMPACTION_TARGET,
    CONF_CONTEXT_THRESHOLD,
    CONF_CONTEXT_TRUNCATE_STRATEGY,
    CONF_EMBEDDING_HISTORY_RETENTION,
    CONF_EMBEDDINGS,
    CONF_EMBEDDINGS_MODEL,
    CONF_EXTRA_BODY,
    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    CONF_MAX_TOKENS,
    CONF_MCP_AUTO_ADD_DEFAULT_PROMPT,
    CONF_MCP_AUTO_ADD_PERMANENT_TOOLS,
    CONF_MCP_DEFAULT_PROMPT_PLACEMENT,
    CONF_MCP_PERMANENT_TOOLS_PLACEMENT,
    CONF_MCP_SERVERS,
    CONF_REASONING_EFFORT,
    CONF_SERVICE_TIER,
    CONF_SHORTEN_TOOL_CALL_ID,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONTEXT_TRUNCATE_RETRIEVAL_STRATEGIES,
    CONTEXT_TRUNCATE_STRATEGY_RETRIEVE,
    CONTEXT_TRUNCATE_STRATEGY_RETRIEVE_COMPACT,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CONTEXT_COMPACTION_PROMPT,
    DEFAULT_CONTEXT_COMPACTION_RECENT_TURNS,
    DEFAULT_CONTEXT_COMPACTION_SUMMARY_PREFIX,
    DEFAULT_CONTEXT_COMPACTION_TARGET,
    DEFAULT_CONTEXT_COMPACTION_TOKEN_LIMIT,
    DEFAULT_CONTEXT_THRESHOLD,
    DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
    DEFAULT_EMBEDDING_HISTORY_RETENTION,
    DEFAULT_EMBEDDING_RECALL_RESULTS,
    DEFAULT_EXTRA_BODY,
    DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MCP_AUTO_ADD_DEFAULT_PROMPT,
    DEFAULT_MCP_AUTO_ADD_PERMANENT_TOOLS,
    DEFAULT_MCP_DEFAULT_PROMPT_PLACEMENT,
    DEFAULT_MCP_PERMANENT_TOOLS_PLACEMENT,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_SERVICE_TIER,
    DEFAULT_SHORTEN_TOOL_CALL_ID,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DOMAIN,
    MCP_CONTEXT_PLACEMENT_COMBINED_SYSTEM,
    MCP_CONTEXT_PLACEMENT_DEDICATED_SYSTEM,
    MCP_CONTEXT_PLACEMENT_DISABLED,
    MCP_CONTEXT_PLACEMENT_USER,
)
from .embedding_memory import (
    EmbeddingMemoryMatch,
    EmbeddingMemoryRecord,
    EmbeddingMemoryScope,
    EmbeddingMemoryStore,
    embedding_history_retention,
)
from .exceptions import FunctionNotFound, ParseArgumentsFailed, TokenLengthExceededError
from .functions import get_function
from .functions.context_recall import build_context_recall_function_tool
from .functions.mcp import build_mcp_function_tool
from .helpers import (
    async_create_chat_completion,
    async_get_context_compaction_text,
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


@dataclass(slots=True)
class _RequestTokenUsage:
    """Largest token usage reported across one agent turn."""

    prompt_tokens: int = 0
    total_tokens: int = 0

    def update(self, prompt_tokens: int, total_tokens: int) -> None:
        """Retain the largest reported usage across tool-call iterations."""
        self.prompt_tokens = max(self.prompt_tokens, prompt_tokens)
        self.total_tokens = max(self.total_tokens, total_tokens)


def _place_mcp_context(
    messages: list[ChatCompletionMessageParam],
    content: str,
    placement: str,
) -> None:
    """Place automatic MCP context in the configured message location."""
    if not content or placement == MCP_CONTEXT_PLACEMENT_DISABLED:
        return

    if placement == MCP_CONTEXT_PLACEMENT_DEDICATED_SYSTEM:
        messages.insert(0,ChatCompletionSystemMessageParam(role="system", content=content))
        return

    target_role = ("system" if placement == MCP_CONTEXT_PLACEMENT_COMBINED_SYSTEM else "user")
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


def _extract_mcp_media_content(
    result: Any,
) -> tuple[
    Any,
    list[
        ChatCompletionContentPartImageParam
        | ChatCompletionContentPartInputAudioParam
    ],
]:
    """Extract standard MCP image and audio blocks for native model input."""
    if not isinstance(result, dict) or not isinstance(result.get("content"), list):
        return result, []

    media_parts: list[
        ChatCompletionContentPartImageParam
        | ChatCompletionContentPartInputAudioParam
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
        except (binascii.Error, ValueError):
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
        summary["attached_to_model"] = True
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
            isinstance(part, dict)
            and part.get("type") in {"image_url", "input_audio"}
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
) -> list[ChatCompletionMessageParam]:
    """Convert chat log content to OpenAI message format."""
    messages: list[ChatCompletionMessageParam] = []
    attachment_parts = attachment_parts or {}
    tool_media_parts = tool_media_parts or {}
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
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": _shorten_tool_call_id(content.tool_call_id)
                    if shorten_tool_call_id
                    else content.tool_call_id,
                    "content": orjson.dumps(_tool_result_data(content)).decode(),
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


def _group_compaction_contents(contents: list[conversation.Content]) -> list[list[conversation.Content]]:
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


class ExtendedOpenAIBaseLLMEntity(Entity):
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

    def _get_mcp_function_tools(self) -> list[dict[str, Any]]:
        """Return the MCP dispatcher when servers are assigned to this agent."""
        server_ids = self.subentry.data.get(CONF_MCP_SERVERS, []) or []
        if not server_ids:
            return []
        return [build_mcp_function_tool(server_ids)]

    def _get_context_recall_function_tools(self) -> list[dict[str, Any]]:
        """Return recall when this agent uses embedding retrieval."""
        if (
            self.subentry.data.get(
                CONF_CONTEXT_TRUNCATE_STRATEGY,
                DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
            )
            not in CONTEXT_TRUNCATE_RETRIEVAL_STRATEGIES
            or not isinstance(self.entry.data.get(CONF_EMBEDDINGS), Mapping)
        ):
            return []
        return [
            build_context_recall_function_tool(
                self.entry.entry_id,
                self.subentry.subentry_id,
            )
        ]

    def _get_internal_function_tools(self) -> list[dict[str, Any]]:
        """Return integration-managed tools enabled for this agent."""
        return (self._get_mcp_function_tools() + self._get_context_recall_function_tools())

    async def _async_recall_embedding_memories(self, chat_log: conversation.ChatLog) -> list[EmbeddingMemoryMatch]:
        """Return relevant archived turns for the latest user input."""
        if self.subentry.data.get(
            CONF_CONTEXT_TRUNCATE_STRATEGY,
            DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
        ) not in CONTEXT_TRUNCATE_RETRIEVAL_STRATEGIES:
            return []

        conversation_id = chat_log.conversation_id
        if not isinstance(conversation_id, str) or not conversation_id:
            return []
        latest_user_content = next(
            (
                content
                for content in reversed(chat_log.content)
                if isinstance(content, conversation.UserContent)
            ),
            None,
        )
        if latest_user_content is None:
            return []

        embeddings = self.entry.data.get(CONF_EMBEDDINGS)
        if not isinstance(embeddings, Mapping):
            return []
        model = embeddings.get(CONF_EMBEDDINGS_MODEL)
        if not isinstance(model, str) or not model:
            return []

        try:
            embedding_client, memory_store = self._embedding_runtime
            scope = EmbeddingMemoryScope(
                entry_id=self.entry.entry_id,
                agent_id=self.subentry.subentry_id,
                conversation_id=conversation_id,
            )
            if not await memory_store.async_has_memories(scope, model):
                return []
            response = await embedding_client.embeddings.create(
                model=model,
                input=_compaction_transcript([latest_user_content]),
            )
            if not response.data or not response.data[0].embedding:
                raise HomeAssistantError(
                    "The embeddings endpoint returned no recall vector"
                )
            return await memory_store.async_search(
                scope,
                list(response.data[0].embedding),
                DEFAULT_EMBEDDING_RECALL_RESULTS,
                model,
            )
        except Exception:  # pylint: disable=broad-except
            _LOGGER.exception(
                "Embedding memory retrieval failed; continuing without recalled context"
            )
            return []

    @staticmethod
    def _embedding_recall_message(matches: list[EmbeddingMemoryMatch], prefix: str) -> ChatCompletionUserMessageParam:
        """Render recalled turns as inert historical request context."""
        payload = [
            {
                "created_at": match.record.created_at,
                "messages": match.record.messages,
            }
            for match in sorted(
                matches, key=lambda item: item.record.created_at
            )
        ]
        return ChatCompletionUserMessageParam(
            role="user",
            content=f"{prefix}\n{json.dumps(payload, ensure_ascii=False, default=str)}",
        )

    @classmethod
    def _place_embedding_recall(
        cls,
        messages: list[ChatCompletionMessageParam],
        matches: list[EmbeddingMemoryMatch],
        prefix: str,
    ) -> None:
        """Insert temporary recalled context immediately before the latest user."""
        if not matches:
            return
        latest_user_index = next(
            (
                index
                for index in reversed(range(len(messages)))
                if messages[index]["role"] == "user"
            ),
            len(messages),
        )
        messages.insert(
            latest_user_index,
            cls._embedding_recall_message(matches, prefix),
        )

    async def _async_get_mcp_agent_context(
        self, language: str
    ) -> MCPAgentContext:
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
            options.get(
                CONF_CONTEXT_TRUNCATE_STRATEGY,
                DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
            )
            in CONTEXT_TRUNCATE_RETRIEVAL_STRATEGIES
        )
        recall_prefix = (
            await async_get_embedding_recall_prefix(self.hass, language)
            if retrieval_enabled
            else ""
        )
        recalled_memories = await self._async_recall_embedding_memories(chat_log)

        async def _build_request_messages() -> list[ChatCompletionMessageParam]:
            """Build messages again after history has potentially changed."""
            attachment_parts = await self.hass.async_add_executor_job(
                encode_attachments, chat_log.content
            )
            converted_messages = _convert_content_to_param(
                chat_log.content,
                shorten_tool_call_id,
                attachment_parts,
                tool_media_parts,
            )
            self._place_embedding_recall(
                converted_messages, recalled_memories, recall_prefix
            )
            if (
                permanent_tools_placement is None
                and default_prompt_placement is None
            ):
                context_messages = self._mcp_context_messages(mcp_context)
                messages: list[ChatCompletionMessageParam] = [
                    *context_messages,
                    *converted_messages,
                ]
                return messages

            _place_mcp_agent_context(
                converted_messages,
                mcp_context,
                permanent_tools_placement
                or DEFAULT_MCP_PERMANENT_TOOLS_PLACEMENT,
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
                if history_changed:
                    if retrieval_enabled:
                        recalled_memories = (
                            await self._async_recall_embedding_memories(chat_log)
                        )
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
                function_tool = next(
                    (
                        f
                        for f in (function_tools)
                        if f["spec"]["name"] == tool_input.tool_name
                    ),
                    None,
                )

                if function_tool is None:
                    raise FunctionNotFound(tool_input.tool_name)

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
            await self._truncate_message_history(chat_log, token_usage, language=language)

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
                        raise ParseArgumentsFailed(tool_call["arguments"]) from err

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
        function_config = function_tool["function"]
        if function_config["type"] == "context_recall":
            function_config = {
                **function_config,
                "conversation_id": chat_log.conversation_id,
            }
        function = get_function(function_config["type"])

        if (
            function_config["type"] != "context_recall"
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
                self.hass, function_config, arguments, llm_context, exposed_entities
            )

        media_parts: list[
            ChatCompletionContentPartImageParam
            | ChatCompletionContentPartInputAudioParam
        ] = []
        if function_config["type"] == "mcp":
            result, media_parts = _extract_mcp_media_content(result)

        return (
            _make_tool_result_content(
                agent_id=self.entity_id,
                tool_call_id=tool_input.id,
                tool_name=tool_input.tool_name,
                data={
                    "result": (
                        result
                        if function_config["type"] == "context_recall"
                        else str(result)
                    )
                },
            ),
            media_parts,
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

    def _clear_message_history(self, chat_log: conversation.ChatLog) -> bool:
        """Keep only the system prompt and the latest conversation turn."""
        original_length = len(chat_log.content)
        last_user_message_index = None
        messages = chat_log.content
        for index in reversed(range(len(messages))):
            if isinstance(messages[index], conversation.UserContent):
                last_user_message_index = index
                break

        if last_user_message_index is not None:
            del messages[1:last_user_message_index]
        return len(messages) != original_length

    def _compaction_history_split_index(
        self,
        messages: list[conversation.Content],
        preserve_current_turn: bool,
        recent_turns_override: int | None = None,
    ) -> int | None:
        """Return where the verbatim tail begins for compaction."""
        user_message_indexes = [
            index
            for index, content in enumerate(messages)
            if isinstance(content, conversation.UserContent)
        ]
        recent_turns = (
            max(0, recent_turns_override)
            if recent_turns_override is not None
            else max(
                0,
                int(
                    self.subentry.data.get(
                        CONF_CONTEXT_COMPACTION_RECENT_TURNS,
                        DEFAULT_CONTEXT_COMPACTION_RECENT_TURNS,
                    )
                ),
            )
        )
        turns_to_keep = recent_turns + int(preserve_current_turn)
        if not preserve_current_turn:
            # Never replace the turn that produced the response currently being
            # returned, even when the configured tail is zero.
            turns_to_keep = max(1, turns_to_keep)
        if len(user_message_indexes) <= turns_to_keep:
            return None
        return user_message_indexes[-turns_to_keep] if turns_to_keep else len(messages)

    def _compaction_max_tokens(
        self,
        chat_log: conversation.ChatLog,
        retained_contents: list[conversation.Content],
        prompt_tokens: int
    ) -> int:
        """Return the available summary budget for the configured target."""
        options = self.subentry.data
        threshold = int(
            options.get(CONF_CONTEXT_THRESHOLD, DEFAULT_CONTEXT_THRESHOLD)
        )
        percentage = int(
            options.get(
                CONF_CONTEXT_COMPACTION_TARGET, DEFAULT_CONTEXT_COMPACTION_TARGET
            )
        )
        percentage = max(1, min(100, percentage))
        target_tokens = max(1, threshold * percentage // 100)

        estimated_history_tokens = _estimate_content_tokens(chat_log.content)
        non_history_tokens = max(0, prompt_tokens - estimated_history_tokens)
        retained_tokens = _estimate_content_tokens(retained_contents)
        available_tokens = max(1, target_tokens - non_history_tokens - retained_tokens)

        max_output_tokens = int(options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS))
        return max(1, min(available_tokens, max_output_tokens))

    async def _compact_message_history(
        self,
        chat_log: conversation.ChatLog,
        token_usage: _RequestTokenUsage,
        preserve_current_turn: bool = False,
        recent_turns_override: int | None = None,
        language: str | None = None,
    ) -> bool:
        """Replace older completed turns with a rolling conversation summary."""
        messages = chat_log.content
        history_split_index = self._compaction_history_split_index(
            messages, preserve_current_turn, recent_turns_override
        )
        if history_split_index is None or history_split_index <= 1:
            _LOGGER.info(
                "Context threshold exceeded, but there is no older history to compact"
            )
            return False

        history_to_summarize = messages[1:history_split_index]
        retained_contents = [messages[0], *messages[history_split_index:]]
        max_summary_tokens = self._compaction_max_tokens(
            chat_log, retained_contents, token_usage.prompt_tokens
        )
        threshold = int(
            self.subentry.data.get(CONF_CONTEXT_THRESHOLD, DEFAULT_CONTEXT_THRESHOLD)
        )
        prompt = str(
            self.subentry.data.get(
                CONF_CONTEXT_COMPACTION_PROMPT,
                DEFAULT_CONTEXT_COMPACTION_PROMPT,
            )
        ).strip()
        if not prompt:
            prompt = DEFAULT_CONTEXT_COMPACTION_PROMPT
        summary_prefix = DEFAULT_CONTEXT_COMPACTION_SUMMARY_PREFIX
        token_limit = DEFAULT_CONTEXT_COMPACTION_TOKEN_LIMIT.format(
            max_summary_tokens=max_summary_tokens
        )
        if language is not None:
            summary_prefix, token_limit = await async_get_context_compaction_text(
                self.hass, language, max_summary_tokens
            )
        system_prompt = f"{prompt}\n{token_limit}"
        model = self.subentry.data.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        model_config = get_model_config(model, self.entry.data.get(CONF_BASE_URL))
        groups = _group_compaction_contents(history_to_summarize)
        summary = ""
        while groups:
            selected_groups: list[list[conversation.Content]] = []
            for group in groups:
                candidate_groups = [*selected_groups, group]
                candidate_payload = json.dumps(
                    {
                        "existing_summary": summary or None,
                        "transcript": [
                            _content_for_compaction(content)
                            for candidate_group in candidate_groups
                            for content in candidate_group
                        ],
                    },
                    ensure_ascii=False,
                    default=str,
                )
                candidate_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": candidate_payload},
                ]
                if (
                    _estimate_serialized_tokens(candidate_messages) + max_summary_tokens
                    > threshold
                ):
                    break
                selected_groups = candidate_groups

            if not selected_groups:
                raise HomeAssistantError("A complete conversation turn is too large to compact safely")

            payload = json.dumps(
                {
                    "existing_summary": summary or None,
                    "transcript": [
                        _content_for_compaction(content)
                        for group in selected_groups
                        for content in group
                    ],
                },
                ensure_ascii=False,
                default=str,
            )
            request: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": payload},
                ],
                "stream": False,
            }
            if model_config["supports_max_completion_tokens"]:
                request["max_completion_tokens"] = max_summary_tokens
            elif model_config["supports_max_tokens"]:
                request["max_tokens"] = max_summary_tokens

            response = await async_create_chat_completion(
                self._client,
                self._unsupported_params.setdefault(model, set()),
                **request,
            )
            result = response.choices[0].message.content
            if not isinstance(result, str) or not result.strip():
                raise HomeAssistantError("The conversation compaction returned no summary")
            summary = result.strip()
            del groups[: len(selected_groups)]

        agent_id = next(
            (
                content.agent_id
                for content in reversed(messages)
                if isinstance(content, conversation.AssistantContent)
            ),
            self.entity_id or DOMAIN,
        )
        messages[1:history_split_index] = [
            conversation.AssistantContent(
                agent_id=agent_id,
                content=f"{summary_prefix}\n{summary}",
            )
        ]
        _LOGGER.info(
            "Context threshold exceeded, older conversation history compacted"
        )
        return True

    async def _archive_message_history(
        self,
        chat_log: conversation.ChatLog,
        preserve_current_turn: bool = False,
        recent_turns_override: int | None = None,
        remove_archived_history: bool = True,
    ) -> bool:
        """Embed and archive older complete turns before removing them."""
        messages = chat_log.content
        history_split_index = self._compaction_history_split_index(
            messages, preserve_current_turn, recent_turns_override
        )
        if history_split_index is None or history_split_index <= 1:
            _LOGGER.info(
                "Context threshold exceeded, but there is no older history to archive"
            )
            return False

        conversation_id = chat_log.conversation_id
        if not isinstance(conversation_id, str) or not conversation_id:
            raise HomeAssistantError(
                "Conversation history cannot be archived without a conversation ID"
            )

        embeddings = self.entry.data.get(CONF_EMBEDDINGS)
        if not isinstance(embeddings, Mapping):
            raise HomeAssistantError("Embedding retrieval is not configured")
        model = embeddings.get(CONF_EMBEDDINGS_MODEL)
        if not isinstance(model, str) or not model:
            raise HomeAssistantError("An embedding model is not configured")

        groups = _group_compaction_contents(messages[1:history_split_index])
        searchable_texts = [_compaction_transcript(group) for group in groups]
        embedding_client, memory_store = self._embedding_runtime
        response = await embedding_client.embeddings.create(
            model=model,
            input=searchable_texts,
        )
        vectors_by_index = {
            item.index: item.embedding for item in response.data
        }
        if len(vectors_by_index) != len(groups) or any(
            index not in vectors_by_index or not vectors_by_index[index]
            for index in range(len(groups))
        ):
            raise HomeAssistantError(
                "The embeddings endpoint returned an incomplete response"
            )

        retention = embedding_history_retention(
            self.subentry.data.get(
                CONF_EMBEDDING_HISTORY_RETENTION,
                DEFAULT_EMBEDDING_HISTORY_RETENTION,
            )
        )
        records = [
            EmbeddingMemoryRecord.create(
                model=model,
                embedding=vectors_by_index[index],
                searchable_text=searchable_texts[index],
                messages=[content.as_dict() for content in group],
                retention=retention,
            )
            for index, group in enumerate(groups)
        ]
        await memory_store.async_append(
            EmbeddingMemoryScope(
                entry_id=self.entry.entry_id,
                agent_id=self.subentry.subentry_id,
                conversation_id=conversation_id,
            ),
            records,
        )

        if remove_archived_history:
            del messages[1:history_split_index]
        _LOGGER.info(
            "Context threshold exceeded, older conversation history archived"
        )
        return True

    def _remove_compactable_history(
        self,
        chat_log: conversation.ChatLog,
        preserve_current_turn: bool,
        recent_turns_override: int | None,
    ) -> bool:
        """Remove the older history segment used by compaction strategies."""
        history_split_index = self._compaction_history_split_index(
            chat_log.content, preserve_current_turn, recent_turns_override
        )
        if history_split_index is None or history_split_index <= 1:
            return False
        del chat_log.content[1:history_split_index]
        return True

    async def _retrieve_and_compact_message_history(
        self,
        chat_log: conversation.ChatLog,
        token_usage: _RequestTokenUsage,
        preserve_current_turn: bool,
        recent_turns_override: int | None,
        language: str | None,
    ) -> bool:
        """Archive verbatim history, then replace that history with a summary."""
        archived = False
        try:
            archived = await self._archive_message_history(
                chat_log,
                preserve_current_turn,
                recent_turns_override,
                remove_archived_history=False,
            )
        except Exception:
            _LOGGER.exception(
                "Embedding archival failed; continuing with conversation compaction"
            )

        try:
            return await self._compact_message_history(
                chat_log,
                token_usage,
                preserve_current_turn,
                recent_turns_override,
                language,
            )
        except Exception:
            if archived:
                _LOGGER.exception(
                    "Conversation compaction failed; using the archived history for retrieval"
                )
                return self._remove_compactable_history(
                    chat_log, preserve_current_turn, recent_turns_override
                )
            _LOGGER.exception(
                "Embedding archival and conversation compaction failed, "
                "falling back to clearing history"
            )
            return self._clear_message_history(chat_log)

    async def _truncate_message_history(
        self,
        chat_log: conversation.ChatLog,
        token_usage: _RequestTokenUsage,
        preserve_current_turn: bool = False,
        recent_turns_override: int | None = None,
        language: str | None = None,
    ) -> bool:
        """Truncate message history based on strategy."""
        options = self.subentry.data
        strategy = options.get(
            CONF_CONTEXT_TRUNCATE_STRATEGY, DEFAULT_CONTEXT_TRUNCATE_STRATEGY
        )

        if strategy == "clear":
            _LOGGER.info("Context threshold exceeded, conversation history cleared")
            return self._clear_message_history(chat_log)
        if strategy == "compact":
            try:
                return await self._compact_message_history(
                    chat_log,
                    token_usage,
                    preserve_current_turn,
                    recent_turns_override,
                    language,
                )
            except Exception:
                _LOGGER.exception("Conversation compaction failed; falling back to clearing history")
                return self._clear_message_history(chat_log)
        if strategy == CONTEXT_TRUNCATE_STRATEGY_RETRIEVE:
            try:
                return await self._archive_message_history(
                    chat_log,
                    preserve_current_turn,
                    recent_turns_override,
                )
            except Exception:
                _LOGGER.exception(
                    "Embedding archival failed; falling back to conversation compaction"
                )
                try:
                    return await self._compact_message_history(
                        chat_log,
                        token_usage,
                        preserve_current_turn,
                        recent_turns_override,
                        language,
                    )
                except Exception:
                    _LOGGER.exception(
                        "Conversation compaction failed; falling back to clearing history"
                    )
                    return self._clear_message_history(chat_log)
        if strategy == CONTEXT_TRUNCATE_STRATEGY_RETRIEVE_COMPACT:
            return await self._retrieve_and_compact_message_history(
                chat_log,
                token_usage,
                preserve_current_turn,
                recent_turns_override,
                language,
            )
        return False
