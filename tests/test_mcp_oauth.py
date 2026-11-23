"""Tests for the Home Assistant MCP OAuth adapter."""

from unittest.mock import AsyncMock, MagicMock

import httpx2 as httpx
import pytest

from custom_components.extended_openai_conversation.mcp_oauth import (
    HomeAssistantOAuthAuth,
    StaticOAuthAuth,
)


async def test_home_assistant_oauth_auth_refreshes_and_authorizes_request():
    """Each request uses Home Assistant's current OAuth access token."""
    session = MagicMock()
    session.async_ensure_token_valid = AsyncMock()
    session.token = {"access_token": "current-token"}
    auth = HomeAssistantOAuthAuth(session)
    request = httpx.Request("POST", "https://mcp.example.com/api")

    flow = auth.async_auth_flow(request)
    authorized = await anext(flow)

    session.async_ensure_token_valid.assert_awaited_once_with()
    assert authorized.headers["Authorization"] == "Bearer current-token"


async def test_home_assistant_oauth_auth_starts_reauth_after_401():
    """A runtime rejection starts reauthentication for the affected entry."""
    session = MagicMock()
    session.async_ensure_token_valid = AsyncMock()
    session.token = {"access_token": "rejected-token"}
    auth = HomeAssistantOAuthAuth(session)
    request = httpx.Request("POST", "https://mcp.example.com/api")
    response = httpx.Response(401, request=request)

    flow = auth.async_auth_flow(request)
    await anext(flow)
    with pytest.raises(StopAsyncIteration):
        await flow.asend(response)

    session.config_entry.async_start_reauth_if_available.assert_called_once_with(
        session.hass
    )


async def test_home_assistant_oauth_auth_does_not_reauth_after_other_status():
    """Non-authentication responses do not open a repair flow."""
    session = MagicMock()
    session.async_ensure_token_valid = AsyncMock()
    session.token = {"access_token": "current-token"}
    auth = HomeAssistantOAuthAuth(session)
    request = httpx.Request("POST", "https://mcp.example.com/api")
    response = httpx.Response(403, request=request)

    flow = auth.async_auth_flow(request)
    await anext(flow)
    with pytest.raises(StopAsyncIteration):
        await flow.asend(response)

    session.config_entry.async_start_reauth_if_available.assert_not_called()


def test_static_oauth_auth_authorizes_config_flow_request():
    """Fresh config-flow tokens can authorize connection and discovery checks."""
    request = httpx.Request("POST", "https://mcp.example.com/api")

    authorized = next(StaticOAuthAuth("setup-token").auth_flow(request))

    assert authorized.headers["Authorization"] == "Bearer setup-token"
