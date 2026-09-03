"""Tests for entity.py message serialization helpers."""

from pathlib import Path

from custom_components.extended_openai_conversation.entity import (
    _convert_content_to_param,
    encode_attachments,
)
from homeassistant.components import conversation


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
