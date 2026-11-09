"""Tests for entity.py message serialization helpers."""

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
)
import probatio as vol

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
