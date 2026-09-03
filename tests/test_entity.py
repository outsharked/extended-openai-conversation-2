"""Tests for entity.py message serialization helpers."""

from pathlib import Path

from custom_components.extended_openai_conversation.entity import (
    _convert_content_to_param,
    _render_extra_body,
    encode_attachments,
)
from homeassistant.components import conversation
from homeassistant.helpers import llm


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
