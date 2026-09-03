"""Tests for the ai_task platform's setup."""

from unittest.mock import MagicMock

from custom_components.extended_openai_conversation.ai_task import (
    ExtendedOpenAITaskEntity,
    async_setup_entry,
)


def _mock_subentry(subentry_type: str, subentry_id: str) -> MagicMock:
    return MagicMock(
        subentry_type=subentry_type,
        subentry_id=subentry_id,
        title=f"{subentry_type} entry",
        data={},
    )


async def test_async_setup_entry_adds_entity_for_ai_task_data_subentry(hass):
    """An ai_task_data subentry gets an ExtendedOpenAITaskEntity added for it."""
    ai_task_subentry = _mock_subentry("ai_task_data", "sub_ai")
    conversation_subentry = _mock_subentry("conversation", "sub_conv")
    config_entry = MagicMock()
    config_entry.subentries = {
        "sub_ai": ai_task_subentry,
        "sub_conv": conversation_subentry,
    }
    async_add_entities = MagicMock()

    await async_setup_entry(hass, config_entry, async_add_entities)

    async_add_entities.assert_called_once()
    (entities,), kwargs = async_add_entities.call_args
    assert len(entities) == 1
    assert isinstance(entities[0], ExtendedOpenAITaskEntity)
    assert kwargs["config_subentry_id"] == "sub_ai"


async def test_async_setup_entry_supports_multiple_ai_task_subentries(hass):
    """Each ai_task_data subentry gets its own async_add_entities call."""
    config_entry = MagicMock()
    config_entry.subentries = {
        "sub_1": _mock_subentry("ai_task_data", "sub_1"),
        "sub_2": _mock_subentry("ai_task_data", "sub_2"),
    }
    async_add_entities = MagicMock()

    await async_setup_entry(hass, config_entry, async_add_entities)

    assert async_add_entities.call_count == 2
    subentry_ids = {
        call.kwargs["config_subentry_id"] for call in async_add_entities.call_args_list
    }
    assert subentry_ids == {"sub_1", "sub_2"}


async def test_async_setup_entry_ignores_non_ai_task_subentries(hass):
    """A config entry with no ai_task_data subentries adds nothing."""
    config_entry = MagicMock()
    config_entry.subentries = {
        "sub_conv": _mock_subentry("conversation", "sub_conv"),
    }
    async_add_entities = MagicMock()

    await async_setup_entry(hass, config_entry, async_add_entities)

    async_add_entities.assert_not_called()


def test_task_entity_supports_generate_data_and_attachments():
    """The entity advertises both GENERATE_DATA and SUPPORT_ATTACHMENTS."""
    from homeassistant.components import ai_task

    entity = ExtendedOpenAITaskEntity(
        MagicMock(), _mock_subentry("ai_task_data", "sub_1")
    )

    assert (
        entity._attr_supported_features
        == ai_task.AITaskEntityFeature.GENERATE_DATA
        | ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS
    )
