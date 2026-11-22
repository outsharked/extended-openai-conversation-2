"""Tests for integration-level Home Assistant actions."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from custom_components.extended_openai_conversation.const import (
    CONF_SERVICE_TYPE,
    DOMAIN,
    SERVICE_QUERY_IMAGE,
    SERVICE_TYPE_MCP,
)
from custom_components.extended_openai_conversation.services import async_setup_services
from homeassistant.exceptions import HomeAssistantError


async def _registered_handlers(hass) -> dict[str, Any]:
    """Set up integration actions and return their registered handlers."""
    await async_setup_services(hass, {})
    return {
        call.args[1]: call.args[2]
        for call in hass.services.async_register.call_args_list
    }


def _mcp_entry() -> MagicMock:
    """Return an MCP config entry that must not be used as an AI endpoint."""
    entry = MagicMock()
    entry.domain = DOMAIN
    entry.data = {CONF_SERVICE_TYPE: SERVICE_TYPE_MCP}
    return entry


async def test_query_image_rejects_mcp_entry(hass):
    """The image-query action cannot treat an MCP client as an AI client."""
    handlers = await _registered_handlers(hass)
    hass.config_entries.async_get_entry.return_value = _mcp_entry()
    call = MagicMock()
    call.data = {
        "config_entry": "mcp-entry",
        "model": "vision-model",
        "prompt": "Describe this image",
        "images": [],
        "max_tokens": 300,
    }

    with pytest.raises(HomeAssistantError, match="is not an AI endpoint"):
        await handlers[SERVICE_QUERY_IMAGE](call)


async def test_change_config_rejects_mcp_entry(hass):
    """The endpoint-update action cannot modify an MCP config entry."""
    handlers = await _registered_handlers(hass)
    hass.config_entries.async_get_entry.return_value = _mcp_entry()
    call = MagicMock()
    call.data = {
        "config_entry": "mcp-entry",
        "base_url": "https://ai.example.com/v1",
    }

    with pytest.raises(HomeAssistantError, match="is not an AI endpoint"):
        await handlers["change_config"](call)

    hass.config_entries.async_update_entry.assert_not_called()
