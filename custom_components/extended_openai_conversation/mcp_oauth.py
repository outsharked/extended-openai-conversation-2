"""Home Assistant OAuth adapter for MCP connections."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from typing import Any, cast, override

import httpx2 as httpx

from homeassistant.components.application_credentials import AuthorizationServer
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_ACCESS_TOKEN, CONF_TOKEN
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_entry_oauth2_flow

from .application_credentials import (
    MCPOAuthApplicationContext,
    oauth_application_context,
)
from .mcp_client import (
    CONF_MCP_AUTHORIZATION_URL,
    CONF_MCP_SERVER_URL,
    CONF_MCP_TOKEN_URL,
)


class HomeAssistantOAuthAuth(httpx.Auth):
    """Add a Home Assistant-managed OAuth token to each MCP request."""

    requires_request_body = True

    def __init__(self, session: config_entry_oauth2_flow.OAuth2Session) -> None:
        """Initialize the HTTPX authentication adapter."""
        self._session = session

    @override
    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        """Refresh the token when needed and authorize the request."""
        await self._session.async_ensure_token_valid()
        request.headers["Authorization"] = (
            f"Bearer {self._session.token[CONF_ACCESS_TOKEN]}"
        )
        yield request


class StaticOAuthAuth(httpx.Auth):
    """Authorize config-flow requests with a freshly obtained access token."""

    def __init__(self, access_token: str) -> None:
        """Initialize the static bearer-token adapter."""
        self._access_token = access_token

    @override
    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response]:
        """Authorize one HTTPX request."""
        request.headers["Authorization"] = f"Bearer {self._access_token}"
        yield request


def oauth_application_context_from_data(
    data: dict[str, Any],
) -> MCPOAuthApplicationContext:
    """Build the dynamic Application Credentials context from entry data."""
    return MCPOAuthApplicationContext(
        authorization_server=AuthorizationServer(
            authorize_url=data[CONF_MCP_AUTHORIZATION_URL],
            token_url=data[CONF_MCP_TOKEN_URL],
        ),
        resource=data[CONF_MCP_SERVER_URL],
    )


async def async_get_mcp_oauth_auth(
    hass: HomeAssistant, entry: ConfigEntry
) -> HomeAssistantOAuthAuth:
    """Return HTTPX authentication backed by Home Assistant OAuth storage."""
    context = oauth_application_context_from_data(dict(entry.data))
    with oauth_application_context(context):
        implementation = (
            await config_entry_oauth2_flow.async_get_config_entry_implementation(
                hass, entry
            )
        )
    session = config_entry_oauth2_flow.OAuth2Session(hass, entry, implementation)
    return HomeAssistantOAuthAuth(session)


def static_oauth_auth(data: dict[str, Any]) -> StaticOAuthAuth:
    """Return HTTPX authentication for OAuth data held by a config flow."""
    token = cast(dict[str, Any], data[CONF_TOKEN])
    return StaticOAuthAuth(cast(str, token[CONF_ACCESS_TOKEN]))
