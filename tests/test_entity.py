"""Tests for entity.py message serialization helpers."""

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
)
import probatio as vol

from custom_components.extended_openai_conversation.const import (
    CONF_CONTEXT_COMPACTION_PROMPT,
    CONF_CONTEXT_COMPACTION_RECENT_TURNS,
    CONF_CONTEXT_COMPACTION_TARGET,
    CONF_CONTEXT_THRESHOLD,
    CONF_CONTEXT_TRUNCATE_STRATEGY,
    CONF_EMBEDDING_CHUNK_SIZE,
    CONF_EMBEDDINGS,
    CONF_EMBEDDINGS_MODEL,
    CONF_MAX_TOKENS,
    CONTEXT_TRUNCATE_STRATEGY_RETRIEVE,
    CONTEXT_TRUNCATE_STRATEGY_RETRIEVE_COMPACT,
)
from custom_components.extended_openai_conversation.embedding_memory import (
    EmbeddingMemoryMatch,
    EmbeddingMemoryRecord,
    EmbeddingMemoryScope,
)
from custom_components.extended_openai_conversation.entity import (
    ExtendedOpenAIBaseLLMEntity,
    MCPAgentContext,
    _convert_content_to_param,
    _extract_extra_content,
    _extract_mcp_media_content,
    _format_structured_output,
    _make_tool_result_content,
    _messages_contain_media,
    _place_mcp_agent_context,
    _place_mcp_context,
    _render_extra_body,
    _RequestTokenUsage,
    _tool_result_data,
    encode_attachments,
)
from custom_components.extended_openai_conversation.mcp_client import (
    MCPServerConfig,
    MCPServiceConfig,
    MCPTool,
)
from homeassistant.components import conversation
from homeassistant.helpers import llm


def _stream_chunk(
    *,
    content: str | None = None,
    tool_calls: list[SimpleNamespace] | None = None,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    """Build a minimal ChatCompletionChunk-like object for _transform_stream."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                delta=SimpleNamespace(content=content, tool_calls=tool_calls),
            )
        ],
        usage=None,
    )


def _tool_call_delta(
    index: int,
    *,
    tool_call_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
    extra_content: dict | None = None,
    model_extra: dict | None = None,
) -> SimpleNamespace:
    """Build a tool-call delta, optionally carrying provider extra_content."""
    function = None
    if name is not None or arguments is not None:
        function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(
        index=index,
        id=tool_call_id,
        function=function,
        extra_content=extra_content,
        model_extra=model_extra,
    )


def _entity_for_stream() -> ExtendedOpenAIBaseLLMEntity:
    """Return an entity instance sufficient for exercising _transform_stream."""
    entity = ExtendedOpenAIBaseLLMEntity.__new__(ExtendedOpenAIBaseLLMEntity)
    entity.subentry = MagicMock()
    entity.subentry.data = {}
    return entity


async def test_mcp_context_messages_localize_one_prompt_and_use_saved_catalogue():
    """One localized prompt precedes complete permanent tool context."""
    entity = ExtendedOpenAIBaseLLMEntity.__new__(ExtendedOpenAIBaseLLMEntity)
    entity.hass = MagicMock()
    entity.subentry = MagicMock()
    entity.subentry.data = {"mcp_servers": ["vault", "weather"]}
    tool = MCPTool(
        name="read_secret",
        title="Read secret",
        description="Read a stored secret.",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )
    on_demand_disabled_tool = MCPTool(
        name="delete_secret",
        title="Delete secret",
        description="Delete a stored secret.",
        input_schema={"type": "object"},
        output_schema=None,
    )
    service = MagicMock()
    service.config = MCPServiceConfig(
        server=MCPServerConfig(
            server_id="vault",
            name="Vault MCP",
            url="https://vault.example.com/mcp",
        ),
        permanent_tools=(tool, on_demand_disabled_tool),
        include_permanent_schemas=True,
        disabled_tools=("delete_secret",),
    )
    second_service = MagicMock()
    second_service.config = MCPServiceConfig(
        server=MCPServerConfig(
            server_id="weather",
            name="Weather MCP",
            url="https://weather.example.com/mcp",
        ),
    )
    scope = MagicMock()
    scope.list_loaded.return_value = (service, second_service)
    manager = MagicMock()
    manager.scope.return_value = scope

    with (
        patch(
            "custom_components.extended_openai_conversation.entity.MCPServiceManager",
            return_value=manager,
        ),
        patch(
            "custom_components.extended_openai_conversation.entity.async_get_mcp_default_prompt",
            new=AsyncMock(return_value="Instructions MCP traduites"),
        ) as get_prompt,
    ):
        context = await entity._async_get_mcp_agent_context("fr")
        messages = entity._mcp_context_messages(context)

    manager.scope.assert_called_once_with(["vault", "weather"])
    scope.list_loaded.assert_called_once_with()
    get_prompt.assert_awaited_once_with(entity.hass, "fr")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Instructions MCP traduites"
    assert context.default_prompt == "Instructions MCP traduites"
    assert context.permanent_tools == messages[1]["content"]
    context = json.loads(messages[1]["content"])
    assert context["server_id"] == "vault"
    assert context["tools"] == [
        {
            "description": "Delete a stored secret.",
            "input_schema": {"type": "object"},
            "name": "delete_secret",
        },
        {
            "description": "Read a stored secret.",
            "input_schema": {"type": "object"},
            "name": "read_secret",
            "output_schema": {"type": "object"},
        },
    ]
    assert "instructions" not in context


async def test_mcp_context_automatic_messages_can_be_disabled():
    """Conversation templates retain MCP values when automatic messages are off."""
    from custom_components.extended_openai_conversation.const import (
        CONF_MCP_AUTO_ADD_DEFAULT_PROMPT,
        CONF_MCP_AUTO_ADD_PERMANENT_TOOLS,
    )

    entity = ExtendedOpenAIBaseLLMEntity.__new__(ExtendedOpenAIBaseLLMEntity)
    entity.hass = MagicMock()
    entity.subentry = MagicMock()
    entity.subentry.data = {
        "mcp_servers": ["vault"],
        CONF_MCP_AUTO_ADD_DEFAULT_PROMPT: False,
        CONF_MCP_AUTO_ADD_PERMANENT_TOOLS: False,
    }
    tool = MCPTool(
        name="read_secret",
        title=None,
        description="Read a stored secret.",
        input_schema={"type": "object"},
        output_schema=None,
    )
    service = MagicMock()
    service.config = MCPServiceConfig(
        server=MCPServerConfig(
            server_id="vault",
            name="Vault MCP",
            url="https://vault.example.com/mcp",
        ),
        permanent_tools=(tool,),
    )
    manager = MagicMock()
    manager.scope.return_value.list_loaded.return_value = (service,)

    with (
        patch(
            "custom_components.extended_openai_conversation.entity.MCPServiceManager",
            return_value=manager,
        ),
        patch(
            "custom_components.extended_openai_conversation.entity.async_get_mcp_default_prompt",
            new=AsyncMock(return_value="MCP instructions"),
        ),
    ):
        context = await entity._async_get_mcp_agent_context("en")

    assert context.default_prompt == "MCP instructions"
    assert "read_secret" in context.permanent_tools
    assert entity._mcp_context_messages(context) == []


def test_place_mcp_agent_context_uses_independent_placements():
    """AI Task prompt and permanent tools use independent placements."""
    messages = [
        {"role": "system", "content": "Existing system"},
        {"role": "user", "content": "Do the task"},
    ]
    context = MCPAgentContext(
        default_prompt="MCP instructions",
        permanent_tools="Permanent tools",
    )

    _place_mcp_agent_context(
        messages,
        context,
        permanent_tools_placement="dedicated_system",
        default_prompt_placement="combined_system",
    )

    assert messages == [
        {"role": "system", "content": "Permanent tools"},
        {"role": "system", "content": "MCP instructions\n\nExisting system"},
        {"role": "user", "content": "Do the task"},
    ]


def test_place_mcp_context_uses_dedicated_system_message():
    """AI Task MCP context can use an additional system message."""
    messages = [
        {"role": "system", "content": "Existing system"},
        {"role": "user", "content": "Do the task"},
    ]
    _place_mcp_context(messages, "MCP context", "dedicated_system")
    assert messages == [
        {"role": "system", "content": "MCP context"},
        {"role": "system", "content": "Existing system"},
        {"role": "user", "content": "Do the task"},
    ]


def test_place_mcp_context_merges_into_selected_role():
    """AI Task MCP context can be merged into an existing message."""
    system_messages = [
        {"role": "system", "content": "Existing system"},
        {"role": "user", "content": "Do the task"},
    ]
    _place_mcp_context(system_messages, "MCP context", "combined_system")
    assert system_messages == [
        {"role": "system", "content": "MCP context\n\nExisting system"},
        {"role": "user", "content": "Do the task"},
    ]

    user_messages = [
        {"role": "system", "content": "Existing system"},
        {"role": "user", "content": "Do the task"},
    ]
    _place_mcp_context(user_messages, "MCP context", "user")
    assert user_messages[1] == {
        "role": "user",
        "content": "MCP context\n\nDo the task",
    }

    disabled_messages = [{"role": "user", "content": "Do the task"}]
    _place_mcp_context(disabled_messages, "MCP context", "disabled")
    assert disabled_messages == [{"role": "user", "content": "Do the task"}]


async def test_mcp_context_messages_skip_manager_without_assigned_servers():
    """An agent without assigned MCP servers receives no MCP context."""
    entity = ExtendedOpenAIBaseLLMEntity.__new__(ExtendedOpenAIBaseLLMEntity)
    entity.hass = MagicMock()
    entity.subentry = MagicMock()
    entity.subentry.data = {}

    with patch(
        "custom_components.extended_openai_conversation.entity.MCPServiceManager"
    ) as manager:
        context = await entity._async_get_mcp_agent_context("en")

    assert context.permanent_tools == ""
    assert context.default_prompt == ""

    manager.assert_not_called()


async def _collect_stream_deltas(chunks: list[SimpleNamespace]) -> list[dict]:
    """Feed fake chunks through _transform_stream and return yielded deltas."""

    async def _stream():
        for chunk in chunks:
            yield chunk

    entity = _entity_for_stream()
    chat_log = MagicMock()
    return [delta async for delta in entity._transform_stream(chat_log, _stream())]


def test_encode_attachments_reads_and_base64_encodes_image(tmp_path: Path):
    """A user message with an image attachment is base64-encoded as image_url."""
    image_path = tmp_path / "snapshot.jpg"
    image_path.write_bytes(b"fake-jpeg-bytes")

    content = conversation.UserContent(
        content="What's in this photo?",
        attachments=[
            conversation.Attachment(
                media_content_id="media-source://camera/front_door",
                mime_type="image/jpg",
                path=image_path,
            )
        ],
    )

    encoded = encode_attachments([content])

    assert list(encoded.keys()) == [0]
    parts = encoded[0]
    assert len(parts) == 1
    assert parts[0]["type"] == "image_url"
    # image/jpg is normalized to the standard image/jpeg MIME type
    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_encode_attachments_skips_content_without_attachments():
    """Content with no attachments (or non-user roles) produces no entries."""
    content = conversation.UserContent(content="No photo here")

    assert encode_attachments([content]) == {}


def test_encode_attachments_missing_file_raises(tmp_path: Path):
    """A referenced attachment that no longer exists on disk raises clearly."""
    content = conversation.UserContent(
        content="gone",
        attachments=[
            conversation.Attachment(
                media_content_id="media-source://camera/front_door",
                mime_type="image/jpeg",
                path=tmp_path / "missing.jpg",
            )
        ],
    )

    try:
        encode_attachments([content])
        raise AssertionError("expected HomeAssistantError")
    except Exception as err:
        assert "does not exist" in str(err)


def test_convert_content_to_param_attaches_image_url_block(tmp_path: Path):
    """_convert_content_to_param builds a multipart message when attachments exist."""
    image_path = tmp_path / "photo.png"
    image_path.write_bytes(b"fake-png-bytes")

    content = conversation.UserContent(
        content="Describe this",
        attachments=[
            conversation.Attachment(
                media_content_id="media-source://camera/front_door",
                mime_type="image/png",
                path=image_path,
            )
        ],
    )

    attachment_parts = encode_attachments([content])
    messages = _convert_content_to_param([content], attachment_parts=attachment_parts)

    assert len(messages) == 1
    msg_content = messages[0]["content"]
    assert isinstance(msg_content, list)
    assert msg_content[0] == {"type": "text", "text": "Describe this"}
    assert msg_content[1]["type"] == "image_url"


def test_convert_content_to_param_without_attachments_unchanged():
    """A plain user message (no attachments) keeps its original string content."""
    content = conversation.UserContent(content="Hello there")

    messages = _convert_content_to_param([content])

    assert messages == [{"role": "user", "content": "Hello there"}]


def test_extract_mcp_media_content_builds_native_image_part():
    """Standard MCP ImageContent becomes a native model image input."""
    encoded = base64.b64encode(b"fake-png-bytes").decode()
    result = {
        "server_id": "images",
        "tool_name": "capture",
        "content": [
            {"type": "text", "text": "Captured"},
            {"type": "image", "mimeType": "image/png", "data": encoded},
        ],
        "structured_content": None,
        "is_error": False,
    }

    serialized, media_parts = _extract_mcp_media_content(result)

    assert serialized["content"] == [
        {"type": "text", "text": "Captured"},
        {
            "type": "image",
            "mimeType": "image/png",
            "attached_to_model": True,
        },
    ]
    assert media_parts == [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encoded}"},
        }
    ]
    assert result["content"][1]["data"] == encoded


def test_extract_mcp_media_content_preserves_invalid_image_payload():
    """Invalid MCP image data remains available with a model-facing error."""
    result = {
        "content": [
            {"type": "image", "mimeType": "image/png", "data": "not base64!"}
        ]
    }

    serialized, media_parts = _extract_mcp_media_content(result)

    assert serialized == {
        "content": [
            {
                "type": "image",
                "mimeType": "image/png",
                "data": "not base64!",
                "media_error": {
                    "code": "invalid_base64_data",
                    "message": (
                        "The media could not be attached because its data is not "
                        "valid base64. The original data is included in this tool "
                        "result."
                    ),
                },
            }
        ]
    }
    assert media_parts == []


def test_extract_mcp_media_content_builds_native_audio_part():
    """Standard MCP AudioContent becomes a native model audio input."""
    encoded = base64.b64encode(b"fake-audio-bytes").decode()
    result = {
        "content": [
            {"type": "audio", "mimeType": "audio/mpeg", "data": encoded}
        ]
    }

    serialized, media_parts = _extract_mcp_media_content(result)

    assert serialized == {
        "content": [
            {
                "type": "audio",
                "mimeType": "audio/mpeg",
                "attached_to_model": True,
            }
        ]
    }
    assert media_parts == [
        {
            "type": "input_audio",
            "input_audio": {"data": encoded, "format": "mp3"},
        }
    ]


def test_extract_mcp_media_content_preserves_unsupported_audio_payload():
    """Unsupported MCP audio remains available with a model-facing error."""
    result = {
        "content": [
            {"type": "audio", "mimeType": "audio/ogg", "data": "ZmFrZQ=="}
        ]
    }

    serialized, media_parts = _extract_mcp_media_content(result)

    assert serialized == {
        "content": [
            {
                "type": "audio",
                "mimeType": "audio/ogg",
                "data": "ZmFrZQ==",
                "media_error": {
                    "code": "unsupported_mime_type",
                    "message": (
                        "This media type cannot be attached natively. The original "
                        "base64 media data is included in this tool result."
                    ),
                },
            }
        ]
    }
    assert media_parts == []


def test_convert_content_to_param_places_media_after_all_tool_results():
    """Parallel tool results precede their combined native media message."""
    first = _make_tool_result_content(
        agent_id="conversation.test",
        tool_call_id="call_1",
        tool_name="capture_one",
        data={"result": "image attached"},
    )
    second = _make_tool_result_content(
        agent_id="conversation.test",
        tool_call_id="call_2",
        tool_name="capture_two",
        data={"result": "image attached"},
    )
    media_parts: dict[
        str,
        list[
            ChatCompletionContentPartImageParam
            | ChatCompletionContentPartInputAudioParam
        ],
    ] = {
        "call_1": [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,Zmlyc3Q="},
            }
        ],
        "call_2": [
            {
                "type": "input_audio",
                "input_audio": {"data": "c2Vjb25k", "format": "wav"},
            }
        ],
    }

    messages = _convert_content_to_param(
        [first, second], tool_media_parts=media_parts
    )

    assert [message["role"] for message in messages] == ["tool", "tool", "user"]
    assert messages[2]["content"] == [
        {"type": "text", "text": "Media returned by tool capture_one:"},
        media_parts["call_1"][0],
        {"type": "text", "text": "Media returned by tool capture_two:"},
        media_parts["call_2"][0],
    ]


def test_messages_contain_media_detects_native_media_parts():
    """Media-bearing prompts can be omitted from verbose logging."""
    assert not _messages_contain_media([{"role": "user", "content": "hello"}])
    assert _messages_contain_media(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,ZmFrZQ=="},
                    }
                ],
            }
        ]
    )
    assert _messages_contain_media(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "ZmFrZQ==", "format": "wav"},
                    }
                ],
            }
        ]
    )


def test_assistant_tool_call_without_text_has_content_key():
    """Assistant messages with tool_calls but no text must still include "content"."""
    content = conversation.AssistantContent(
        agent_id="test_agent",
        content=None,
        tool_calls=[
            llm.ToolInput(
                id="call_1",
                tool_name="turn_on_light",
                tool_args={"entity_id": "light.living_room"},
            )
        ],
    )

    messages = _convert_content_to_param([content])

    assert len(messages) == 1
    assert messages[0]["content"] is None


def test_assistant_tool_call_preserves_provider_extra_content():
    """Opaque provider metadata stored in native is re-attached to the tool call."""
    content = conversation.AssistantContent(
        agent_id="test_agent",
        content=None,
        tool_calls=[
            llm.ToolInput(
                id="call_1",
                tool_name="turn_on_light",
                tool_args={"entity_id": "light.living_room"},
            )
        ],
        native={
            "tool_call_extra_content_by_id": {
                "call_1": {
                    "google": {
                        "thought_signature": "encrypted-signature",
                    }
                }
            }
        },
    )

    messages = _convert_content_to_param([content])

    tool_call = messages[0]["tool_calls"][0]
    assert tool_call["extra_content"] == {
        "google": {
            "thought_signature": "encrypted-signature",
        }
    }


def test_assistant_tool_call_without_provider_metadata_unchanged():
    """Standard OpenAI tool calls do not gain provider-specific fields."""
    content = conversation.AssistantContent(
        agent_id="test_agent",
        content=None,
        tool_calls=[
            llm.ToolInput(
                id="call_1",
                tool_name="turn_on_light",
                tool_args={"entity_id": "light.living_room"},
            )
        ],
    )

    messages = _convert_content_to_param([content])

    assert "extra_content" not in messages[0]["tool_calls"][0]


def test_extract_extra_content_from_model_extra():
    """Unknown SDK response fields can be recovered from model_extra."""

    response_part = SimpleNamespace(
        extra_content=None,
        model_extra={
            "extra_content": {
                "google": {
                    "thought_signature": "encrypted-signature",
                }
            }
        },
    )

    assert _extract_extra_content(response_part) == {
        "google": {
            "thought_signature": "encrypted-signature",
        }
    }


async def test_transform_stream_attaches_extra_content_from_separate_chunk():
    """extra_content may arrive on a later chunk than the tool call id/name."""
    signature = {"google": {"thought_signature": "encrypted-signature"}}
    deltas = await _collect_stream_deltas(
        [
            _stream_chunk(
                tool_calls=[
                    _tool_call_delta(
                        0,
                        tool_call_id="call_1",
                        name="turn_on_light",
                        arguments="",
                    )
                ]
            ),
            _stream_chunk(
                tool_calls=[
                    _tool_call_delta(
                        0,
                        arguments='{"entity_id":"light.living_room"}',
                        extra_content=signature,
                    )
                ],
                finish_reason="tool_calls",
            ),
        ]
    )

    assert deltas[0] == {"role": "assistant"}
    assert "tool_calls" in deltas[1]
    assert deltas[1]["tool_calls"][0].id == "call_1"
    assert "native" not in deltas[1]
    assert deltas[2] == {
        "native": {"tool_call_extra_content_by_id": {"call_1": signature}}
    }


async def test_transform_stream_emits_native_once_across_multiple_tool_batches():
    """Multiple tool_calls finish reasons must not yield native more than once."""
    sig_1 = {"google": {"thought_signature": "sig-1"}}
    sig_2 = {"google": {"thought_signature": "sig-2"}}
    deltas = await _collect_stream_deltas(
        [
            _stream_chunk(
                tool_calls=[
                    _tool_call_delta(
                        0,
                        tool_call_id="call_1",
                        name="turn_on_light",
                        arguments='{"entity_id":"light.one"}',
                        extra_content=sig_1,
                    )
                ],
                finish_reason="tool_calls",
            ),
            _stream_chunk(
                tool_calls=[
                    _tool_call_delta(
                        0,
                        tool_call_id="call_2",
                        name="turn_on_light",
                        arguments='{"entity_id":"light.two"}',
                        extra_content=sig_2,
                    )
                ],
                finish_reason="tool_calls",
            ),
        ]
    )

    native_deltas = [delta for delta in deltas if "native" in delta]
    tool_call_deltas = [delta for delta in deltas if "tool_calls" in delta]

    assert len(tool_call_deltas) == 2
    assert len(native_deltas) == 1
    assert native_deltas[0]["native"] == {
        "tool_call_extra_content_by_id": {
            "call_1": sig_1,
            "call_2": sig_2,
        }
    }


async def test_transform_stream_omits_native_without_provider_metadata():
    """Standard OpenAI tool streams should not emit a native payload."""
    deltas = await _collect_stream_deltas(
        [
            _stream_chunk(
                tool_calls=[
                    _tool_call_delta(
                        0,
                        tool_call_id="call_1",
                        name="turn_on_light",
                        arguments='{"entity_id":"light.living_room"}',
                    )
                ],
                finish_reason="tool_calls",
            ),
        ]
    )

    assert all("native" not in delta for delta in deltas)
    assert any("tool_calls" in delta for delta in deltas)


def test_assistant_text_only_content_unchanged():
    """Assistant messages that already carry text keep their content unchanged."""
    content = conversation.AssistantContent(
        agent_id="test_agent",
        content="Hello there",
    )

    messages = _convert_content_to_param([content])

    assert len(messages) == 1
    assert messages[0]["content"] == "Hello there"


def test_render_extra_body_empty_string_returns_none(hass):
    """An empty/unset extra_body option disables the feature."""

    assert _render_extra_body("", hass, "test-model") is None
    assert _render_extra_body("   ", hass, "test-model") is None


def test_render_extra_body_parses_plain_json(hass):
    """A plain (non-templated) JSON string is parsed as-is."""

    result = _render_extra_body(
        '{"chat_template_kwargs": {"enable_thinking": false}}', hass, "test-model"
    )

    assert result == {"chat_template_kwargs": {"enable_thinking": False}}


def test_render_extra_body_renders_jinja_before_parsing(hass):
    """Jinja templating is rendered before the result is parsed as JSON.

    Uses a numeric substitution rather than a Jinja boolean: {{ true }}
    renders to Python's "True" (capitalized), which isn't valid JSON on
    its own (json.loads requires lowercase "true") - a real limitation
    inherited from the reference implementation, not something this test
    should paper over.
    """

    result = _render_extra_body('{"seed": {{ 42 }} }', hass, "test-model")

    assert result == {"seed": 42}


def test_render_extra_body_invalid_json_returns_none_not_raises(hass):
    """Malformed JSON logs a warning and disables the feature for this call,
    rather than failing the conversation turn."""

    assert _render_extra_body("{not valid json", hass, "test-model") is None


def test_render_extra_body_invalid_template_returns_none_not_raises(hass):
    """A broken Jinja template also degrades gracefully rather than raising."""

    # Unclosed {% if %} block - guaranteed Jinja syntax error
    assert _render_extra_body("{% if true %}", hass, "test-model") is None


def test_tool_result_content_round_trips_to_tool_message():
    """A function result is serialized as a tool message on any HA version."""
    content = _make_tool_result_content(
        agent_id="conversation.test",
        tool_call_id="call_1",
        tool_name="get_weather",
        data={"result": "sunny"},
    )

    messages = _convert_content_to_param([content])

    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "call_1"
    assert json.loads(messages[0]["content"]) == {"result": "sunny"}


def test_format_structured_output_converts_probatio_schema():
    """AI Task structures (probatio schemas since HA 2026.9) convert to JSON schema."""
    schema = vol.Schema(
        {
            vol.Required("name", description="The name"): str,
            vol.Optional("count"): int,
        }
    )

    result = _format_structured_output(schema, None)

    assert result["type"] == "object"
    assert result["properties"]["name"] == {
        "type": "string",
        "description": "The name",
    }
    # _adjust_schema makes every property required, optional ones nullable
    assert result["properties"]["count"]["type"] == ["integer", "null"]
    assert sorted(result["required"]) == ["count", "name"]


def _entity_for_compaction(
    strategy: str = "compact",
) -> ExtendedOpenAIBaseLLMEntity:
    """Return an entity configured for history compaction tests."""
    entity = ExtendedOpenAIBaseLLMEntity.__new__(ExtendedOpenAIBaseLLMEntity)
    entity.entry = MagicMock()
    entity.entry.data = {}
    entity.subentry = MagicMock()
    entity.subentry.data = {
        CONF_CONTEXT_THRESHOLD: 1000,
        CONF_CONTEXT_TRUNCATE_STRATEGY: strategy,
        CONF_CONTEXT_COMPACTION_TARGET: 60,
        CONF_CONTEXT_COMPACTION_RECENT_TURNS: 1,
        CONF_MAX_TOKENS: 500,
    }
    entity._unsupported_params = {}
    return entity


def _compaction_chat_log() -> MagicMock:
    """Return a chat log containing one old and one current turn."""
    chat_log = MagicMock()
    chat_log.content = [
        conversation.SystemContent(content="System prompt"),
        conversation.UserContent(content="My preferred temperature is 20 C"),
        conversation.AssistantContent(
            agent_id="conversation.test",
            content="I will remember that preference.",
        ),
        conversation.UserContent(content="What temperature do I prefer?"),
        conversation.AssistantContent(
            agent_id="conversation.test",
            content="You prefer 20 C.",
        ),
    ]
    return chat_log


async def test_embedding_recall_searches_the_current_agent_conversation():
    """Automatic recall embeds the latest user input in the saved scope."""
    entity = _entity_for_compaction(CONTEXT_TRUNCATE_STRATEGY_RETRIEVE)
    entity.entry.entry_id = "endpoint-1"
    entity.entry.data = {
        CONF_EMBEDDINGS: {CONF_EMBEDDINGS_MODEL: "embedding-model"}
    }
    entity.subentry.subentry_id = "agent-1"
    chat_log = _compaction_chat_log()
    chat_log.conversation_id = "conversation-1"
    embedding_client = MagicMock()
    embedding_client.embeddings.create = AsyncMock(
        return_value=SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[1.0, 0.0])]
        )
    )
    record = EmbeddingMemoryRecord(
        record_id="record-1",
        created_at="2026-01-01T00:00:00+00:00",
        expires_at="2027-01-01T00:00:00+00:00",
        model="embedding-model",
        embedding=[1.0, 0.0],
        searchable_text="Earlier preference",
        messages=[{"role": "user", "content": "I prefer 20 C"}],
    )
    match = EmbeddingMemoryMatch(record=record, score=1.0)
    memory_store = MagicMock()
    memory_store.async_has_memories = AsyncMock(return_value=True)
    memory_store.async_search = AsyncMock(return_value=[match])

    with patch.object(
        ExtendedOpenAIBaseLLMEntity,
        "_embedding_runtime",
        new_callable=PropertyMock,
        return_value=(embedding_client, memory_store),
    ):
        matches = await entity._async_recall_embedding_memories(chat_log)

    assert matches == [match]
    scope = EmbeddingMemoryScope(
        entry_id="endpoint-1",
        agent_id="agent-1",
        conversation_id="conversation-1",
    )
    memory_store.async_has_memories.assert_awaited_once_with(
        scope, "embedding-model"
    )
    request = embedding_client.embeddings.create.await_args.kwargs
    assert request["model"] == "embedding-model"
    assert "What temperature do I prefer?" in request["input"]
    memory_store.async_search.assert_awaited_once_with(
        scope, [1.0, 0.0], 3, "embedding-model"
    )

    recall_message = entity._embedding_recall_message(
        matches, "Relevant earlier context:"
    )
    assert recall_message["role"] == "user"
    assert recall_message["content"].startswith("Relevant earlier context:\n")
    assert "I prefer 20 C" in recall_message["content"]

    request_messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Current question"},
    ]
    entity._place_embedding_recall(
        request_messages, matches, "Relevant earlier context:"
    )
    assert [message["role"] for message in request_messages] == [
        "system",
        "user",
        "user",
    ]
    assert "I prefer 20 C" in request_messages[1]["content"]
    assert request_messages[2]["content"] == "Current question"


async def test_embedding_recall_avoids_api_call_without_archived_history():
    """A new conversation performs no unnecessary embeddings request."""
    entity = _entity_for_compaction(CONTEXT_TRUNCATE_STRATEGY_RETRIEVE)
    entity.entry.entry_id = "endpoint-1"
    entity.entry.data = {
        CONF_EMBEDDINGS: {CONF_EMBEDDINGS_MODEL: "embedding-model"}
    }
    entity.subentry.subentry_id = "agent-1"
    chat_log = _compaction_chat_log()
    chat_log.conversation_id = "conversation-1"
    embedding_client = MagicMock()
    embedding_client.embeddings.create = AsyncMock()
    memory_store = MagicMock()
    memory_store.async_has_memories = AsyncMock(return_value=False)

    with patch.object(
        ExtendedOpenAIBaseLLMEntity,
        "_embedding_runtime",
        new_callable=PropertyMock,
        return_value=(embedding_client, memory_store),
    ):
        matches = await entity._async_recall_embedding_memories(chat_log)

    assert matches == []
    embedding_client.embeddings.create.assert_not_awaited()
    memory_store.async_search.assert_not_called()


async def test_embedding_recall_failure_is_non_destructive():
    """A runtime recall failure omits optional context without truncating history."""
    entity = _entity_for_compaction(CONTEXT_TRUNCATE_STRATEGY_RETRIEVE)
    entity.entry.entry_id = "endpoint-1"
    entity.entry.data = {
        CONF_EMBEDDINGS: {CONF_EMBEDDINGS_MODEL: "embedding-model"}
    }
    entity.subentry.subentry_id = "agent-1"
    chat_log = _compaction_chat_log()
    chat_log.conversation_id = "conversation-1"
    original_contents = list(chat_log.content)
    embedding_client = MagicMock()
    embedding_client.embeddings.create = AsyncMock(
        side_effect=RuntimeError("endpoint unavailable")
    )
    memory_store = MagicMock()
    memory_store.async_has_memories = AsyncMock(return_value=True)

    with patch.object(
        ExtendedOpenAIBaseLLMEntity,
        "_embedding_runtime",
        new_callable=PropertyMock,
        return_value=(embedding_client, memory_store),
    ):
        matches = await entity._async_recall_embedding_memories(chat_log)

    assert matches == []
    assert chat_log.content == original_contents


async def test_context_recall_execution_injects_current_conversation_id():
    """The model cannot choose the conversation archive searched by recall."""
    entity = _entity_for_compaction(CONTEXT_TRUNCATE_STRATEGY_RETRIEVE)
    entity.hass = MagicMock()
    entity._attr_entity_id = "conversation.test"
    function = MagicMock()
    function.execute = AsyncMock(
        return_value={"memories": [], "matches_returned": 0}
    )
    function_tool = {
        "spec": {"name": "context_recall"},
        "function": {
            "type": "context_recall",
            "entry_id": "endpoint-1",
            "agent_id": "agent-1",
        },
    }
    tool_input = MagicMock(
        id="tool-call-1",
        tool_name="context_recall",
        tool_args={"query": "earlier preference"},
    )
    chat_log = MagicMock(conversation_id="conversation-1")

    with patch(
        "custom_components.extended_openai_conversation.entity.get_function",
        return_value=function,
    ):
        tool_result, media_parts = await entity._execute_function_tool(
            function_tool,
            tool_input,
            chat_log,
            None,
            [],
        )

    function.execute.assert_awaited_once_with(
        entity.hass,
        {
            "type": "context_recall",
            "entry_id": "endpoint-1",
            "agent_id": "agent-1",
            "conversation_id": "conversation-1",
        },
        {"query": "earlier preference"},
        None,
        [],
    )
    assert media_parts == []
    assert _tool_result_data(tool_result) == {
        "result": {"memories": [], "matches_returned": 0}
    }


async def test_compaction_summarizes_old_history_and_keeps_latest_turn():
    """Compaction stores a rolling summary without replacing the current reply."""
    entity = _entity_for_compaction()
    chat_log = _compaction_chat_log()
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="The user prefers 20 C.")
            )
        ]
    )

    with (
        patch.object(
            ExtendedOpenAIBaseLLMEntity,
            "_client",
            new_callable=PropertyMock,
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.extended_openai_conversation.entity."
            "async_create_chat_completion",
            new=AsyncMock(return_value=response),
        ) as create_completion,
    ):
        await entity._truncate_message_history(
            chat_log,
            _RequestTokenUsage(prompt_tokens=100, total_tokens=1100),
        )

    assert [content.role for content in chat_log.content] == [
        "system",
        "assistant",
        "user",
        "assistant",
    ]
    assert chat_log.content[1].content == (
        "Summary of earlier conversation:\nThe user prefers 20 C."
    )
    assert chat_log.content[-1].content == "You prefer 20 C."

    request = create_completion.await_args.kwargs
    assert request["stream"] is False
    assert request.get("max_tokens", request.get("max_completion_tokens")) == 500
    assert "tools" not in request
    assert [message["role"] for message in request["messages"]] == [
        "system",
        "user",
    ]
    transcript = request["messages"][1]["content"]
    assert "My preferred temperature is 20 C" in transcript
    assert "What temperature do I prefer?" not in transcript


async def test_compaction_uses_configured_prompt_and_preserves_recent_turns():
    """Compaction uses the agent prompt and keeps the configured complete tail."""
    entity = _entity_for_compaction()
    entity.subentry.data.update(
        {
            CONF_CONTEXT_COMPACTION_PROMPT: "Preserve every confirmed identifier.",
            CONF_CONTEXT_COMPACTION_RECENT_TURNS: 2,
        }
    )
    chat_log = _compaction_chat_log()
    chat_log.content[1:1] = [
        conversation.UserContent(content="My device identifier is alpha-1"),
        conversation.AssistantContent(
            agent_id="conversation.test",
            content="I will retain alpha-1.",
        ),
    ]
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Earlier summary"))]
    )

    with (
        patch.object(
            ExtendedOpenAIBaseLLMEntity,
            "_client",
            new_callable=PropertyMock,
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.extended_openai_conversation.entity."
            "async_create_chat_completion",
            new=AsyncMock(return_value=response),
        ) as create_completion,
        patch(
            "custom_components.extended_openai_conversation.entity."
            "async_get_context_compaction_text",
            new=AsyncMock(
                return_value=(
                    "Résumé de la conversation précédente :",
                    "Limitez le résumé à 500 jetons.",
                )
            ),
        ),
    ):
        changed = await entity._compact_message_history(
            chat_log,
            _RequestTokenUsage(prompt_tokens=200, total_tokens=1200),
            language="fr",
        )

    assert changed is True
    assert create_completion.await_args.kwargs["messages"][0]["content"].startswith(
        "Preserve every confirmed identifier."
    )
    assert create_completion.await_args.kwargs["messages"][0]["content"].endswith(
        "Limitez le résumé à 500 jetons."
    )
    assert [content.content for content in chat_log.content] == [
        "System prompt",
        "Résumé de la conversation précédente :\nEarlier summary",
        "My preferred temperature is 20 C",
        "I will remember that preference.",
        "What temperature do I prefer?",
        "You prefer 20 C.",
    ]


async def test_compaction_chunks_oversized_history_by_complete_turn():
    """Large histories are summarized through bounded complete-turn chunks."""
    entity = _entity_for_compaction()
    entity.subentry.data.update(
        {
            CONF_CONTEXT_THRESHOLD: 500,
            CONF_CONTEXT_COMPACTION_TARGET: 20,
            CONF_MAX_TOKENS: 100,
        }
    )
    chat_log = MagicMock()
    chat_log.content = [conversation.SystemContent(content="System prompt")]
    for index in range(4):
        chat_log.content.extend(
            [
                conversation.UserContent(
                    content=f"Old request {index}: " + "x" * 400
                ),
                conversation.AssistantContent(
                    agent_id="conversation.test",
                    content=f"Old response {index}: " + "y" * 400,
                ),
            ]
        )
    chat_log.content.extend(
        [
            conversation.UserContent(content="Current request"),
            conversation.AssistantContent(
                agent_id="conversation.test", content="Current response"
            ),
        ]
    )
    create_completion = AsyncMock(
        side_effect=[
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Chunk one"))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Chunk two"))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Chunk three"))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Final summary"))]
            ),
        ]
    )

    with (
        patch.object(
            ExtendedOpenAIBaseLLMEntity,
            "_client",
            new_callable=PropertyMock,
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.extended_openai_conversation.entity."
            "async_create_chat_completion",
            new=create_completion,
        ),
    ):
        changed = await entity._compact_message_history(
            chat_log,
            _RequestTokenUsage(prompt_tokens=1800, total_tokens=1900),
        )

    assert changed is True
    assert create_completion.await_count > 1
    for call in create_completion.await_args_list:
        request = call.kwargs
        request_tokens = len(json.dumps(request["messages"])) // 4
        output_tokens = request.get(
            "max_tokens", request.get("max_completion_tokens")
        )
        assert request_tokens + output_tokens <= 500
    assert chat_log.content[-2].content == "Current request"
    assert chat_log.content[-1].content == "Current response"


async def test_compaction_failure_falls_back_to_clear_strategy():
    """A failed summary call still leaves a usable latest conversation turn."""
    entity = _entity_for_compaction()
    chat_log = _compaction_chat_log()

    with (
        patch.object(
            ExtendedOpenAIBaseLLMEntity,
            "_client",
            new_callable=PropertyMock,
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.extended_openai_conversation.entity."
            "async_create_chat_completion",
            new=AsyncMock(side_effect=RuntimeError("summary failed")),
        ),
    ):
        await entity._truncate_message_history(
            chat_log,
            _RequestTokenUsage(prompt_tokens=100, total_tokens=1100),
        )

    assert [content.content for content in chat_log.content] == [
        "System prompt",
        "What temperature do I prefer?",
        "You prefer 20 C.",
    ]


async def test_compaction_rolls_previous_summary_into_next_summary():
    """Later compaction summarizes the prior summary plus subsequent old turns."""
    entity = _entity_for_compaction()
    chat_log = _compaction_chat_log()
    responses = [
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Summary one"))]
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Summary two"))]
        ),
    ]

    with (
        patch.object(
            ExtendedOpenAIBaseLLMEntity,
            "_client",
            new_callable=PropertyMock,
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.extended_openai_conversation.entity."
            "async_create_chat_completion",
            new=AsyncMock(side_effect=responses),
        ) as create_completion,
    ):
        usage = _RequestTokenUsage(prompt_tokens=100, total_tokens=1100)
        await entity._truncate_message_history(chat_log, usage)
        chat_log.content.extend(
            [
                conversation.UserContent(content="Remember anything else?"),
                conversation.AssistantContent(
                    agent_id="conversation.test",
                    content="Nothing else yet.",
                ),
            ]
        )
        await entity._truncate_message_history(chat_log, usage)

    second_transcript = create_completion.await_args_list[1].kwargs["messages"][1][
        "content"
    ]
    assert "Summary of earlier conversation:\\nSummary one" in second_transcript
    assert "What temperature do I prefer?" in second_transcript
    assert [content.content for content in chat_log.content] == [
        "System prompt",
        "Summary of earlier conversation:\nSummary two",
        "Remember anything else?",
        "Nothing else yet.",
    ]


async def test_clear_strategy_does_not_call_summarizer():
    """The existing clear strategy remains local and keeps the latest turn."""
    entity = _entity_for_compaction("clear")
    chat_log = _compaction_chat_log()

    with patch(
        "custom_components.extended_openai_conversation.entity."
        "async_create_chat_completion",
        new=AsyncMock(),
    ) as create_completion:
        await entity._truncate_message_history(
            chat_log,
            _RequestTokenUsage(prompt_tokens=100, total_tokens=1100),
        )

    create_completion.assert_not_awaited()
    assert [content.content for content in chat_log.content] == [
        "System prompt",
        "What temperature do I prefer?",
        "You prefer 20 C.",
    ]


async def test_retrieve_compact_archives_and_summarizes_the_same_history():
    """Combined retrieval preserves verbatim turns before summarizing them."""
    entity = _entity_for_compaction(
        CONTEXT_TRUNCATE_STRATEGY_RETRIEVE_COMPACT
    )
    entity.entry.entry_id = "endpoint-1"
    entity.entry.data = {
        CONF_EMBEDDINGS: {CONF_EMBEDDINGS_MODEL: "embedding-model"}
    }
    entity.subentry.subentry_id = "agent-1"
    chat_log = _compaction_chat_log()
    chat_log.conversation_id = "conversation-1"
    embedding_client = MagicMock()
    embedding_client.embeddings.create = AsyncMock(
        return_value=SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[1.0, 0.0])]
        )
    )
    memory_store = MagicMock()
    memory_store.async_append = AsyncMock()
    summary_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="The user prefers 20 C.")
            )
        ]
    )

    with (
        patch.object(
            ExtendedOpenAIBaseLLMEntity,
            "_embedding_runtime",
            new_callable=PropertyMock,
            return_value=(embedding_client, memory_store),
        ),
        patch.object(
            ExtendedOpenAIBaseLLMEntity,
            "_client",
            new_callable=PropertyMock,
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.extended_openai_conversation.entity."
            "async_create_chat_completion",
            new=AsyncMock(return_value=summary_response),
        ) as create_completion,
    ):
        changed = await entity._truncate_message_history(
            chat_log,
            _RequestTokenUsage(prompt_tokens=100, total_tokens=1100),
        )

    assert changed is True
    embedding_input = embedding_client.embeddings.create.await_args.kwargs["input"]
    assert len(embedding_input) == 1
    assert "My preferred temperature is 20 C" in embedding_input[0]
    summary_input = create_completion.await_args.kwargs["messages"][1]["content"]
    assert "My preferred temperature is 20 C" in summary_input
    records = memory_store.async_append.await_args.args[1]
    assert [message["role"] for message in records[0].messages] == [
        "user",
        "assistant",
    ]
    assert records[0].messages[0]["content"] == "My preferred temperature is 20 C"
    assert (
        records[0].messages[1]["content"]
        == "I will remember that preference."
    )
    assert [content.content for content in chat_log.content] == [
        "System prompt",
        "Summary of earlier conversation:\nThe user prefers 20 C.",
        "What temperature do I prefer?",
        "You prefer 20 C.",
    ]


async def test_embedding_archival_chunks_one_turn_and_stores_messages_once():
    """Oversized searchable text uses several vectors for one lossless record."""
    entity = _entity_for_compaction(CONTEXT_TRUNCATE_STRATEGY_RETRIEVE)
    entity.entry.entry_id = "endpoint-1"
    entity.entry.data = {
        CONF_EMBEDDINGS: {
            CONF_EMBEDDINGS_MODEL: "embedding-model",
            CONF_EMBEDDING_CHUNK_SIZE: 10,
        }
    }
    entity.subentry.subentry_id = "agent-1"
    chat_log = _compaction_chat_log()
    chat_log.conversation_id = "conversation-1"
    embedding_client = MagicMock()

    async def create_embeddings(**kwargs):
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[float(index), 1.0])
                for index, _value in enumerate(kwargs["input"])
            ]
        )

    embedding_client.embeddings.create = AsyncMock(side_effect=create_embeddings)
    memory_store = MagicMock()
    memory_store.async_append = AsyncMock()

    with patch.object(
        ExtendedOpenAIBaseLLMEntity,
        "_embedding_runtime",
        new_callable=PropertyMock,
        return_value=(embedding_client, memory_store),
    ):
        changed = await entity._archive_message_history(chat_log)

    assert changed is True
    embedding_inputs = embedding_client.embeddings.create.await_args.kwargs["input"]
    assert len(embedding_inputs) > 1
    assert all(len(value) <= 40 for value in embedding_inputs)
    records = memory_store.async_append.await_args.args[1]
    assert len(records) == 1
    assert len(records[0].additional_chunks) == len(embedding_inputs) - 1
    assert [message["role"] for message in records[0].messages] == [
        "user",
        "assistant",
    ]


async def test_retrieve_compact_uses_archive_when_summarization_fails():
    """A safely archived segment can be removed when summarization fails."""
    entity = _entity_for_compaction(
        CONTEXT_TRUNCATE_STRATEGY_RETRIEVE_COMPACT
    )
    entity._archive_message_history = AsyncMock(return_value=True)
    entity._compact_message_history = AsyncMock(
        side_effect=RuntimeError("summary failed")
    )
    chat_log = _compaction_chat_log()

    changed = await entity._truncate_message_history(
        chat_log,
        _RequestTokenUsage(prompt_tokens=100, total_tokens=1100),
    )

    assert changed is True
    assert [content.content for content in chat_log.content] == [
        "System prompt",
        "What temperature do I prefer?",
        "You prefer 20 C.",
    ]


async def test_retrieve_compact_compacts_when_archival_fails():
    """A retrieval outage does not prevent the configured rolling summary."""
    entity = _entity_for_compaction(
        CONTEXT_TRUNCATE_STRATEGY_RETRIEVE_COMPACT
    )
    entity._archive_message_history = AsyncMock(
        side_effect=RuntimeError("embedding failed")
    )
    entity._compact_message_history = AsyncMock(return_value=True)
    chat_log = _compaction_chat_log()

    changed = await entity._truncate_message_history(
        chat_log,
        _RequestTokenUsage(prompt_tokens=100, total_tokens=1100),
    )

    assert changed is True
    entity._compact_message_history.assert_awaited_once()


async def test_transform_stream_records_usage_without_compacting_mid_stream():
    """Threshold handling is deferred until the completed agent turn."""
    entity = _entity_for_stream()
    entity._truncate_message_history = AsyncMock()
    chat_log = MagicMock()
    usage = _RequestTokenUsage()

    async def _stream():
        yield SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(
                prompt_tokens=900,
                completion_tokens=200,
                total_tokens=1100,
            ),
        )

    assert [
        delta
        async for delta in entity._transform_stream(chat_log, _stream(), usage)
    ] == [{"role": "assistant"}]
    assert usage == _RequestTokenUsage(prompt_tokens=900, total_tokens=1100)
    entity._truncate_message_history.assert_not_awaited()
