"""Tests for Home Assistant MCP service runtime lookup."""

from unittest.mock import MagicMock

import pytest

from custom_components.extended_openai_conversation.const import (
    CONF_SERVICE_TYPE,
    SERVICE_TYPE_AI_ENDPOINT,
    SERVICE_TYPE_MCP,
)
from custom_components.extended_openai_conversation.mcp_client import (
    MCPClient,
    MCPServerConfig,
    MCPServiceConfig,
)
from custom_components.extended_openai_conversation.mcp_service_manager import (
    MCPServiceManager,
)
from homeassistant.config_entries import ConfigEntryState


def _mcp_entry(
    server_id: str,
    *,
    state: ConfigEntryState = ConfigEntryState.LOADED,
    runtime_data: object | None = None,
) -> MagicMock:
    config = MCPServiceConfig(
        server=MCPServerConfig(
            server_id=server_id,
            name=f"{server_id} MCP",
            url=f"https://{server_id}.example.com/mcp",
        )
    )
    entry = MagicMock()
    entry.data = {**config.to_dict(), CONF_SERVICE_TYPE: SERVICE_TYPE_MCP}
    entry.state = state
    entry.runtime_data = runtime_data or MCPClient(config.server)
    return entry


def test_scope_lists_only_allowed_loaded_mcp_services(hass):
    """Enumeration excludes unselected, unavailable, and non-MCP entries."""
    loaded = _mcp_entry("vault")
    unselected = _mcp_entry("weather")
    unavailable = _mcp_entry("calendar", state=ConfigEntryState.NOT_LOADED)
    invalid_runtime = _mcp_entry("invalid")
    invalid_runtime.runtime_data = MagicMock()
    ai_entry = MagicMock()
    ai_entry.data = {CONF_SERVICE_TYPE: SERVICE_TYPE_AI_ENDPOINT}
    hass.config_entries.async_entries.return_value = [
        ai_entry,
        loaded,
        unselected,
        unavailable,
        invalid_runtime,
    ]

    services = MCPServiceManager(hass).scope(
        {"vault", "calendar", "invalid"}
    ).list_loaded()

    assert len(services) == 1
    assert services[0].config.server.server_id == "vault"
    assert services[0].client is loaded.runtime_data
    hass.config_entries.async_entries.assert_called_once_with(
        "extended_openai_conversation"
    )

def test_empty_scope_lists_no_services(hass):
    """An agent with no selected server IDs cannot enumerate MCP services."""
    hass.config_entries.async_entries.return_value = [_mcp_entry("vault")]

    assert MCPServiceManager(hass).scope(set()).list_loaded() == ()
    hass.config_entries.async_entries.assert_not_called()

@pytest.mark.parametrize("server_ids", ["vault", {"vault", 1}])
def test_scope_rejects_malformed_server_id_collection(hass, server_ids):
    """Malformed agent options cannot accidentally broaden the scope."""
    with pytest.raises(TypeError, match="server_ids"):
        MCPServiceManager(hass).scope(server_ids)
