"""Tests for config_flow.py.

Note: these test validate_input() and the schema-building logic directly,
rather than driving the full flow through Home Assistant's flow manager
(hass.config_entries.flow.async_init(...)) - that needs the real
pytest-homeassistant-custom-component `hass` fixture, which this project's
conftest.py currently shadows project-wide with a lightweight mock for fast
unit testing. Actually exercising the flow steps end-to-end is tracked as a
follow-up (see issue #7) rather than attempted here with a mocked hass that
doesn't behave like the real flow manager.
"""

from unittest.mock import AsyncMock, patch

import pytest

from custom_components.extended_openai_conversation.config_flow import (
    ExtendedOpenAISubentryFlowHandler,
    validate_input,
)
from custom_components.extended_openai_conversation.const import (
    CONF_API_PROVIDER,
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_FUNCTION_TOOLS,
    CONF_PROMPT,
    CONF_SKILLS,
    DEFAULT_CONF_BASE_URL,
)
from homeassistant.const import CONF_API_KEY
from homeassistant.exceptions import HomeAssistantError


def _schema_keys(schema: dict) -> set[str]:
    """Extract the plain string keys from a dict of voluptuous Marker -> Selector."""
    return {getattr(key, "schema", key) for key in schema}


async def test_validate_input_calls_get_authenticated_client(hass):
    """A valid config is passed through to get_authenticated_client as-is."""
    with patch(
        "custom_components.extended_openai_conversation.config_flow.get_authenticated_client",
        new=AsyncMock(),
    ) as mock_get_client:
        data = {
            CONF_API_KEY: "sk-test",
            CONF_BASE_URL: "https://my-proxy.example.com",
        }
        await validate_input(hass, data)

    mock_get_client.assert_called_once()
    assert mock_get_client.call_args.kwargs["api_key"] == "sk-test"
    assert (
        mock_get_client.call_args.kwargs["base_url"] == "https://my-proxy.example.com"
    )


async def test_validate_input_clears_default_base_url(hass):
    """The default (OpenAI's own) base_url is popped rather than sent through,
    so future changes to OpenAI's own default URL don't require reconfiguring
    existing entries."""
    with patch(
        "custom_components.extended_openai_conversation.config_flow.get_authenticated_client",
        new=AsyncMock(),
    ) as mock_get_client:
        data = {CONF_API_KEY: "sk-test", CONF_BASE_URL: DEFAULT_CONF_BASE_URL}
        await validate_input(hass, data)

    assert CONF_BASE_URL not in data
    assert mock_get_client.call_args.kwargs["base_url"] is None


async def test_validate_input_azure_without_base_url_raises(hass):
    """Azure OpenAI requires a custom base URL - a missing one is a clear error,
    not a confusing failure from the OpenAI client itself."""
    data = {CONF_API_KEY: "sk-test", CONF_API_PROVIDER: "azure"}

    with pytest.raises(HomeAssistantError, match="Azure OpenAI requires"):
        await validate_input(hass, data)


def test_openai_config_option_schema_includes_core_options():
    """The shared subentry schema includes the expected core option keys,
    given at least one skill is available (see the dedicated test below for
    what happens when none are)."""
    handler = ExtendedOpenAISubentryFlowHandler()
    handler.context = {"source": "user"}

    schema = handler.openai_config_option_schema({}, skills=[{"name": "skill_a"}])

    keys = _schema_keys(schema)
    assert CONF_PROMPT in keys
    assert CONF_CHAT_MODEL in keys
    assert CONF_FUNCTION_TOOLS in keys
    assert CONF_SKILLS in keys


def test_openai_config_option_schema_omits_skills_field_when_none_available():
    """With no skills loaded, the skills selector is dropped from the schema
    entirely (rather than shown as an always-empty dropdown)."""
    handler = ExtendedOpenAISubentryFlowHandler()
    handler.context = {"source": "user"}

    schema = handler.openai_config_option_schema({}, skills=[])

    assert CONF_SKILLS not in _schema_keys(schema)
    # everything else is still there
    assert CONF_PROMPT in _schema_keys(schema)


def test_openai_config_option_schema_defaults_skills_for_new_subentry():
    """A brand-new subentry with no skills option defaults to all loaded skills."""
    handler = ExtendedOpenAISubentryFlowHandler()
    handler.context = {"source": "user"}  # _is_new == True
    skills = [{"name": "skill_a"}, {"name": "skill_b"}]

    schema = handler.openai_config_option_schema({}, skills=skills)

    skills_key = next(k for k in schema if getattr(k, "schema", k) == CONF_SKILLS)
    assert skills_key.default() == ["skill_a", "skill_b"]


def test_openai_config_option_schema_preserves_existing_skills_on_reconfigure():
    """An existing subentry's already-chosen skills aren't silently reset to
    "all skills" on reconfigure."""
    handler = ExtendedOpenAISubentryFlowHandler()
    handler.context = {"source": "reconfigure"}  # _is_new == False
    skills = [{"name": "skill_a"}, {"name": "skill_b"}]

    schema = handler.openai_config_option_schema(
        {CONF_SKILLS: ["skill_a"]}, skills=skills
    )

    skills_key = next(k for k in schema if getattr(k, "schema", k) == CONF_SKILLS)
    assert skills_key.default() == ["skill_a"]
