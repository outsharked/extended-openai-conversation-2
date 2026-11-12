"""Tests for the Home Assistant MCP OAuth adapter."""

from unittest.mock import AsyncMock, MagicMock

import httpx2 as httpx

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


def test_static_oauth_auth_authorizes_config_flow_request():
    """Fresh config-flow tokens can authorize connection and discovery checks."""
    request = httpx.Request("POST", "https://mcp.example.com/api")

    authorized = next(StaticOAuthAuth("setup-token").auth_flow(request))

    assert authorized.headers["Authorization"] == "Bearer setup-token"
