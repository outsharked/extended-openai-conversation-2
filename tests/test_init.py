"""Tests for integration setup and service-type routing."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.extended_openai_conversation import (
    async_setup_entry,
    async_unload_entry,
)
from custom_components.extended_openai_conversation.const import (
    CONF_EMBEDDINGS,
    CONF_RERANKER,
    CONF_RERANKER_MODEL,
    CONF_RERANKER_URL,
    CONF_SERVICE_TYPE,
    SERVICE_TYPE_MCP,
)
from custom_components.extended_openai_conversation.context_management import (
    RerankerClient,
)
from custom_components.extended_openai_conversation.mcp_client import (
    CONF_MCP_AUTHORIZATION_URL,
    CONF_MCP_SERVER_ID,
    CONF_MCP_SERVER_URL,
    CONF_MCP_TOKEN_URL,
    CONF_MCP_USE_OAUTH,
    MCPClient,
)
from homeassistant.const import CONF_API_KEY, CONF_TOKEN
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady


async def test_mcp_entry_does_not_create_openai_client(hass):
    """Setting up an MCP service must not run AI endpoint authentication."""
    entry = MagicMock()
    entry.data = {
        CONF_SERVICE_TYPE: SERVICE_TYPE_MCP,
        CONF_MCP_SERVER_ID: "vault",
        CONF_MCP_SERVER_URL: "https://mcp.example.com/api",
    }
    entry.title = "Vault MCP"
    entry.add_update_listener.return_value = MagicMock()

    with patch(
        "custom_components.extended_openai_conversation.get_authenticated_client"
    ) as get_authenticated_client:
        assert await async_setup_entry(hass, entry)

    get_authenticated_client.assert_not_called()
    hass.config_entries.async_forward_entry_setups.assert_not_called()
    entry.async_on_unload.assert_called_once()
    assert isinstance(entry.runtime_data, MCPClient)
    assert entry.runtime_data.config.server_id == "vault"


async def test_mcp_oauth_entry_injects_persisted_auth_provider(hass):
    """OAuth MCP entries restore SDK auth without starting a browser flow."""
    entry = MagicMock()
    entry.data = {
        CONF_SERVICE_TYPE: SERVICE_TYPE_MCP,
        CONF_MCP_SERVER_ID: "vault",
        CONF_MCP_SERVER_URL: "https://mcp.example.com/api",
        CONF_MCP_USE_OAUTH: True,
        CONF_MCP_AUTHORIZATION_URL: "https://auth.example.com/authorize",
        CONF_MCP_TOKEN_URL: "https://auth.example.com/token",
        CONF_TOKEN: {"access_token": "access-token"},
    }
    entry.title = "Vault MCP"
    entry.add_update_listener.return_value = MagicMock()
    auth = MagicMock()

    with patch(
        "custom_components.extended_openai_conversation.async_get_mcp_oauth_auth",
        new=AsyncMock(return_value=auth),
    ) as get_auth:
        assert await async_setup_entry(hass, entry)

    get_auth.assert_awaited_once_with(hass, entry)
    assert entry.runtime_data._auth is auth


async def test_mcp_oauth_token_refresh_does_not_reload_entry(hass):
    """Refreshing an existing OAuth token preserves the active MCP client."""
    hass.config_entries.async_reload = AsyncMock()
    entry = MagicMock()
    entry.entry_id = "mcp-entry"
    entry.data = {
        CONF_SERVICE_TYPE: SERVICE_TYPE_MCP,
        CONF_MCP_SERVER_ID: "vault",
        CONF_MCP_SERVER_URL: "https://mcp.example.com/api",
        CONF_MCP_USE_OAUTH: True,
        CONF_MCP_AUTHORIZATION_URL: "https://auth.example.com/authorize",
        CONF_MCP_TOKEN_URL: "https://auth.example.com/token",
        CONF_TOKEN: {"access_token": "old-token"},
    }
    entry.title = "Vault MCP"
    entry.add_update_listener.return_value = MagicMock()

    with patch(
        "custom_components.extended_openai_conversation.async_get_mcp_oauth_auth",
        new=AsyncMock(return_value=MagicMock()),
    ):
        assert await async_setup_entry(hass, entry)

    client = entry.runtime_data
    listener = entry.add_update_listener.call_args.args[0]
    entry.data = {
        **entry.data,
        CONF_TOKEN: {"access_token": "refreshed-token"},
    }

    await listener(hass, entry)

    hass.config_entries.async_reload.assert_not_awaited()
    assert entry.runtime_data is client


async def test_mcp_oauth_non_token_update_reloads_entry(hass):
    """Changing MCP connection configuration still reloads the entry."""
    hass.config_entries.async_reload = AsyncMock()
    entry = MagicMock()
    entry.entry_id = "mcp-entry"
    entry.data = {
        CONF_SERVICE_TYPE: SERVICE_TYPE_MCP,
        CONF_MCP_SERVER_ID: "vault",
        CONF_MCP_SERVER_URL: "https://mcp.example.com/api",
        CONF_MCP_USE_OAUTH: True,
        CONF_MCP_AUTHORIZATION_URL: "https://auth.example.com/authorize",
        CONF_MCP_TOKEN_URL: "https://auth.example.com/token",
        CONF_TOKEN: {"access_token": "access-token"},
    }
    entry.title = "Vault MCP"
    entry.add_update_listener.return_value = MagicMock()

    with patch(
        "custom_components.extended_openai_conversation.async_get_mcp_oauth_auth",
        new=AsyncMock(return_value=MagicMock()),
    ):
        assert await async_setup_entry(hass, entry)

    listener = entry.add_update_listener.call_args.args[0]
    entry.data = {
        **entry.data,
        CONF_MCP_SERVER_URL: "https://new-mcp.example.com/api",
    }

    await listener(hass, entry)

    hass.config_entries.async_reload.assert_awaited_once_with("mcp-entry")


async def test_mcp_oauth_entry_without_token_still_loads(hass):
    """A deliberately saved incomplete OAuth entry does not crash setup."""
    entry = MagicMock()
    entry.data = {
        CONF_SERVICE_TYPE: SERVICE_TYPE_MCP,
        CONF_MCP_SERVER_ID: "vault",
        CONF_MCP_SERVER_URL: "https://mcp.example.com/api",
        CONF_MCP_USE_OAUTH: True,
    }
    entry.title = "Vault MCP"
    entry.add_update_listener.return_value = MagicMock()

    with patch(
        "custom_components.extended_openai_conversation.async_get_mcp_oauth_auth",
        new=AsyncMock(),
    ) as get_auth:
        assert await async_setup_entry(hass, entry)

    get_auth.assert_not_awaited()
    assert entry.runtime_data._auth is None


async def test_mcp_oauth_setup_preserves_home_assistant_failure_types(hass):
    """Native OAuth retry and reauth exceptions retain their HA semantics."""
    entry = MagicMock()
    entry.data = {
        CONF_SERVICE_TYPE: SERVICE_TYPE_MCP,
        CONF_MCP_SERVER_ID: "vault",
        CONF_MCP_SERVER_URL: "https://mcp.example.com/api",
        CONF_MCP_USE_OAUTH: True,
        CONF_TOKEN: {"access_token": "access-token"},
    }
    entry.title = "Vault MCP"

    for error in (
        ConfigEntryAuthFailed("reauth required"),
        ConfigEntryNotReady("provider unavailable"),
    ):
        with (
            patch(
                "custom_components.extended_openai_conversation."
                "async_get_mcp_oauth_auth",
                new=AsyncMock(side_effect=error),
            ),
            pytest.raises(type(error)),
        ):
            await async_setup_entry(hass, entry)


async def test_mcp_oauth_setup_maps_invalid_stored_data_to_reauth(hass):
    """Malformed persisted OAuth data starts repair instead of crashing setup."""
    entry = MagicMock()
    entry.data = {
        CONF_SERVICE_TYPE: SERVICE_TYPE_MCP,
        CONF_MCP_SERVER_ID: "vault",
        CONF_MCP_SERVER_URL: "https://mcp.example.com/api",
        CONF_MCP_USE_OAUTH: True,
        CONF_TOKEN: {"access_token": "access-token"},
    }
    entry.title = "Vault MCP"

    with (
        patch(
            "custom_components.extended_openai_conversation."
            "async_get_mcp_oauth_auth",
            new=AsyncMock(side_effect=KeyError("authorization_url")),
        ),
        pytest.raises(ConfigEntryAuthFailed, match="configuration is invalid"),
    ):
        await async_setup_entry(hass, entry)


async def test_mcp_entry_unloads_without_ai_platforms(hass):
    """Unloading an MCP service must not unload AI endpoint platforms."""
    entry = MagicMock()
    entry.data = {CONF_SERVICE_TYPE: SERVICE_TYPE_MCP}
    entry.runtime_data = MagicMock(aclose=AsyncMock())

    assert await async_unload_entry(hass, entry)

    entry.runtime_data.aclose.assert_awaited_once_with()
    hass.config_entries.async_unload_platforms.assert_not_called()


async def test_ai_entry_creates_configured_reranker_client(hass):
    """AI endpoint setup retains an isolated reranker runtime client."""
    entry = MagicMock(entry_id="endpoint-1")
    entry.data = {
        CONF_API_KEY: "chat-key",
        CONF_RERANKER: {
            CONF_API_KEY: "reranker-key",
            CONF_RERANKER_URL: "https://reranker.example.com/v1/rerank",
            CONF_RERANKER_MODEL: "reranker-model",
        },
    }
    entry.add_update_listener.return_value = MagicMock()
    hass.config_entries.async_forward_entry_setups = AsyncMock()
    session = MagicMock()

    with (
        patch(
            "custom_components.extended_openai_conversation."
            "get_authenticated_client",
            new=AsyncMock(return_value=MagicMock()),
        ),
        patch(
            "custom_components.extended_openai_conversation."
            "async_get_clientsession",
            return_value=session,
        ),
        patch(
            "custom_components.extended_openai_conversation."
            "async_setup_templates",
            new=AsyncMock(),
        ),
    ):
        assert await async_setup_entry(hass, entry)

    assert isinstance(entry.runtime_data.reranker_client, RerankerClient)
    assert entry.runtime_data.reranker_client.endpoint_url == (
        "https://reranker.example.com/v1/rerank"
    )
    assert entry.runtime_data.reranker_client.model == "reranker-model"


async def test_ai_entry_periodically_cleans_context_memory(hass):
    """AI endpoints periodically reclaim expired context data."""
    entry = MagicMock()
    entry.data = {
        CONF_API_KEY: "chat-key",
        CONF_EMBEDDINGS: {CONF_API_KEY: "embedding-key"},
    }
    entry.add_update_listener.return_value = MagicMock()
    hass.config_entries.async_forward_entry_setups = AsyncMock()
    memory_store = MagicMock()
    memory_store.async_cleanup_expired = AsyncMock()
    cancel_interval = MagicMock()

    with (
        patch(
            "custom_components.extended_openai_conversation."
            "get_authenticated_client",
            new=AsyncMock(return_value=MagicMock()),
        ),
        patch(
            "custom_components.extended_openai_conversation."
            "EmbeddingMemoryStore",
            return_value=memory_store,
        ),
        patch(
            "custom_components.extended_openai_conversation."
            "async_track_time_interval",
            return_value=cancel_interval,
        ) as track_interval,
        patch(
            "custom_components.extended_openai_conversation."
            "async_setup_templates",
            new=AsyncMock(),
        ),
    ):
        assert await async_setup_entry(hass, entry)

    cleanup_snapshots = MagicMock()
    entry.runtime_data.recall_snapshots.cleanup_expired = cleanup_snapshots
    cleanup_callback = track_interval.call_args.args[1]
    await cleanup_callback(datetime(2026, 1, 1, tzinfo=UTC))

    cleanup_snapshots.assert_called_once_with()
    assert memory_store.async_cleanup_expired.await_count == 2
    entry.async_on_unload.assert_any_call(cancel_interval)
