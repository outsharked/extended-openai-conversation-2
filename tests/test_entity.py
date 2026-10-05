"""Tests for entity.py message serialization helpers."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import probatio as vol

from custom_components.extended_openai_conversation.entity import (
    ExtendedOpenAIBaseLLMEntity,
    _convert_content_to_param,
    _extract_extra_content,
    _format_structured_output,
    _make_tool_result_content,
    _render_extra_body,
    encode_attachments,
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
