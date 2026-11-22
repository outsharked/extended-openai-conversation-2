"""Application Credentials platform for MCP OAuth."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
import contextvars
from dataclasses import dataclass
from typing import override

from homeassistant.components.application_credentials import (
    AuthorizationServer,
    ClientCredential,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_entry_oauth2_flow


@dataclass(frozen=True, slots=True)
class MCPOAuthApplicationContext:
    """Per-server data needed to construct an OAuth implementation."""

    authorization_server: AuthorizationServer
    resource: str


_OAUTH_CONTEXT: contextvars.ContextVar[MCPOAuthApplicationContext] = (
    contextvars.ContextVar("extended_openai_conversation_mcp_oauth_context")
)


@contextmanager
def oauth_application_context(
    context: MCPOAuthApplicationContext,
) -> Generator[None]:
    """Expose one MCP server while HA resolves Application Credentials."""
    token = _OAUTH_CONTEXT.set(context)
    try:
        yield
    finally:
        _OAUTH_CONTEXT.reset(token)


class MCPOAuthImplementation(
    config_entry_oauth2_flow.LocalOAuth2ImplementationWithPkce
):
    """PKCE OAuth implementation for one MCP server."""

    def __init__(
        self,
        hass: HomeAssistant,
        auth_domain: str,
        credential: ClientCredential,
        context: MCPOAuthApplicationContext,
    ) -> None:
        """Initialize an MCP OAuth implementation."""
        super().__init__(
            hass,
            auth_domain,
            credential.client_id,
            context.authorization_server.authorize_url,
            context.authorization_server.token_url,
            credential.client_secret,
        )
        self._name: str | None = credential.name
        self._resource = context.resource

    @property
    @override
    def name(self) -> str:
        """Return the user-visible credential name."""
        return self._name or self.client_id

    @property
    @override
    def extra_authorize_data(self) -> dict:
        """Include the RFC 8707 resource in the authorization request."""
        return {**super().extra_authorize_data, "resource": self._resource}

    @property
    @override
    def extra_token_resolve_data(self) -> dict:
        """Include PKCE and the RFC 8707 resource in the token request."""
        return {**super().extra_token_resolve_data, "resource": self._resource}


async def async_get_auth_implementation(
    hass: HomeAssistant,
    auth_domain: str,
    credential: ClientCredential,
) -> config_entry_oauth2_flow.AbstractOAuth2Implementation:
    """Build an implementation using the active MCP server's endpoints."""
    return MCPOAuthImplementation(
        hass,
        auth_domain,
        credential,
        _OAUTH_CONTEXT.get(),
    )


async def async_get_authorization_server(
    hass: HomeAssistant,
) -> AuthorizationServer:
    """Return the active MCP server's dynamically discovered endpoints."""
    return _OAUTH_CONTEXT.get().authorization_server
