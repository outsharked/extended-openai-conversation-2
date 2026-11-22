"""Tests for shared template-function lifecycle."""

from unittest.mock import AsyncMock, MagicMock

from custom_components.extended_openai_conversation.const import (
    CONF_SERVICE_TYPE,
    DOMAIN,
    SERVICE_TYPE_AI_ENDPOINT,
    SERVICE_TYPE_MCP,
)
from custom_components.extended_openai_conversation.template import (
    DATA_TEMPLATE_MANAGER,
    async_unload_templates,
)


async def test_unload_templates_ignores_mcp_entries(hass):
    """The last AI endpoint unloads templates even when MCP entries remain."""
    ai_entry = MagicMock()
    ai_entry.data = {CONF_SERVICE_TYPE: SERVICE_TYPE_AI_ENDPOINT}
    mcp_entry = MagicMock()
    mcp_entry.data = {CONF_SERVICE_TYPE: SERVICE_TYPE_MCP}
    hass.config_entries.async_entries.return_value = [ai_entry, mcp_entry]
    manager = MagicMock()
    manager.async_on_unload = AsyncMock()
    hass.data = {DOMAIN: {DATA_TEMPLATE_MANAGER: manager}}

    assert await async_unload_templates(hass)

    manager.async_on_unload.assert_awaited_once_with()
    assert DATA_TEMPLATE_MANAGER not in hass.data[DOMAIN]
