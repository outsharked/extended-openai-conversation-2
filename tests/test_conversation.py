"""Tests for the conversation platform's setup."""

from unittest.mock import MagicMock

import pytest

from custom_components.extended_openai_conversation.conversation import (
    ExtendedOpenAIAgentEntity,
    async_setup_entry,
)
from custom_components.extended_openai_conversation.exceptions import FunctionNotFound


def _mock_subentry(subentry_type: str, subentry_id: str) -> MagicMock:
    return MagicMock(
        subentry_type=subentry_type,
        subentry_id=subentry_id,
        title=f"{subentry_type} entry",
        data={},
    )


async def test_async_setup_entry_adds_entity_for_conversation_subentry(hass):
    """A conversation subentry gets an ExtendedOpenAIAgentEntity added for it."""
    conversation_subentry = _mock_subentry("conversation", "sub_conv")
    ai_task_subentry = _mock_subentry("ai_task_data", "sub_ai")
    config_entry = MagicMock()
    config_entry.subentries = {
        "sub_conv": conversation_subentry,
        "sub_ai": ai_task_subentry,
    }
    async_add_entities = MagicMock()

    await async_setup_entry(hass, config_entry, async_add_entities)

    async_add_entities.assert_called_once()
    (entities,), kwargs = async_add_entities.call_args
    assert len(entities) == 1
    assert isinstance(entities[0], ExtendedOpenAIAgentEntity)
    assert kwargs["config_subentry_id"] == "sub_conv"


async def test_async_setup_entry_supports_multiple_conversation_subentries(hass):
    """Each conversation subentry gets its own async_add_entities call."""
    config_entry = MagicMock()
    config_entry.subentries = {
        "sub_1": _mock_subentry("conversation", "sub_1"),
        "sub_2": _mock_subentry("conversation", "sub_2"),
    }
    async_add_entities = MagicMock()

    await async_setup_entry(hass, config_entry, async_add_entities)

    assert async_add_entities.call_count == 2
    subentry_ids = {
        call.kwargs["config_subentry_id"] for call in async_add_entities.call_args_list
    }
    assert subentry_ids == {"sub_1", "sub_2"}


async def test_async_setup_entry_ignores_non_conversation_subentries(hass):
    """A config entry with no conversation subentries adds nothing."""
    config_entry = MagicMock()
    config_entry.subentries = {
        "sub_ai": _mock_subentry("ai_task_data", "sub_ai"),
    }
    async_add_entities = MagicMock()

    await async_setup_entry(hass, config_entry, async_add_entities)

    async_add_entities.assert_not_called()


def test_agent_entity_supports_control_and_streaming():
    """The entity advertises CONTROL support and streaming."""
    from homeassistant.components.conversation import ConversationEntityFeature

    entity = ExtendedOpenAIAgentEntity(
        MagicMock(), _mock_subentry("conversation", "sub_1")
    )

    assert entity._attr_supported_features == ConversationEntityFeature.CONTROL
    assert entity._attr_supports_streaming is True


def test_agent_entity_supports_all_languages():
    """supported_languages returns MATCH_ALL (all languages)."""
    from homeassistant.const import MATCH_ALL

    entity = ExtendedOpenAIAgentEntity(
        MagicMock(), _mock_subentry("conversation", "sub_1")
    )

    assert entity.supported_languages == MATCH_ALL


def test_get_function_tools_returns_default_when_unconfigured():
    """With no function_tools option set, the built-in default toolset is used."""
    from custom_components.extended_openai_conversation.const import (
        DEFAULT_CONF_FUNCTION_TOOLS,
    )

    entity = ExtendedOpenAIAgentEntity(
        MagicMock(), _mock_subentry("conversation", "sub_1")
    )

    tools = entity._get_function_tools()

    assert tools == DEFAULT_CONF_FUNCTION_TOOLS


def test_get_function_tools_parses_custom_yaml():
    """A custom function_tools YAML string is parsed and validated."""
    from custom_components.extended_openai_conversation.const import CONF_FUNCTION_TOOLS

    custom_yaml = """
- spec:
    name: test_function
    description: A test function
    parameters:
      type: object
      properties: {}
  function:
    type: template
    value_template: "hello"
"""
    subentry = _mock_subentry("conversation", "sub_1")
    subentry.data = {CONF_FUNCTION_TOOLS: custom_yaml}
    entity = ExtendedOpenAIAgentEntity(MagicMock(), subentry)

    tools = entity._get_function_tools()

    assert len(tools) == 1
    assert tools[0]["spec"]["name"] == "test_function"


def test_get_function_tools_unknown_function_type_raises():
    """An unrecognized function type raises FunctionNotFound."""
    from custom_components.extended_openai_conversation.const import CONF_FUNCTION_TOOLS

    custom_yaml = """
- spec:
    name: bad_function
    description: A bad function
    parameters:
      type: object
      properties: {}
  function:
    type: does_not_exist
"""
    subentry = _mock_subentry("conversation", "sub_1")
    subentry.data = {CONF_FUNCTION_TOOLS: custom_yaml}
    entity = ExtendedOpenAIAgentEntity(MagicMock(), subentry)

    with pytest.raises(FunctionNotFound):
        entity._get_function_tools()


def test_build_system_prompt_renders_template_with_context(hass):
    """The system prompt template renders with ha_name/exposed_entities/etc."""
    from custom_components.extended_openai_conversation.const import CONF_PROMPT
    from custom_components.extended_openai_conversation.skills import SkillManager
    from homeassistant.helpers.template import TemplateEnvironment

    hass.config.location_name = "Test Home"
    subentry = _mock_subentry("conversation", "sub_1")
    subentry.data = {CONF_PROMPT: "You manage {{ ha_name }}."}
    entity = ExtendedOpenAIAgentEntity(MagicMock(), subentry)
    entity.hass = hass
    entity.skill_manager = MagicMock(spec=SkillManager)
    entity.skill_manager.get_all_skills.return_value = []
    hass.data["template.environment"] = TemplateEnvironment(hass)

    llm_context = MagicMock(device_id=None)
    user_input = MagicMock()

    result = entity._build_system_prompt([], llm_context, user_input)

    assert result == "You manage Test Home."
