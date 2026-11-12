"""Tests for framework-independent MCP OAuth discovery and registration."""

import httpx2 as httpx
import pytest

from custom_components.extended_openai_conversation.mcp_client import (
    MCPAuthenticationError,
    MCPOAuthClientRegistrar,
    MCPOAuthConfiguration,
)


async def test_oauth_discovery_retains_issuer_and_registration_endpoint():
    """OAuth discovery retains the identity needed for credential reuse."""

    async def handler(request: httpx.Request) -> httpx.Response:
        if "oauth-protected-resource" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "resource": "https://mcp.example.com/api",
                    "authorization_servers": ["https://auth.example.com/"],
                    "scopes_supported": ["mcp"],
                },
            )
        if "oauth-authorization-server" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "issuer": "https://auth.example.com/",
                    "authorization_endpoint": "https://auth.example.com/authorize",
                    "token_endpoint": "https://auth.example.com/token",
                    "registration_endpoint": "https://auth.example.com/register",
                    "response_types_supported": ["code"],
                },
            )
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await MCPOAuthClientRegistrar.async_discover_mcp_oauth(
            "https://mcp.example.com/api", http_client=client
        )

    assert result.issuer == "https://auth.example.com/"
    assert result.registration_url == "https://auth.example.com/register"
    assert result.scopes == ("mcp",)


async def test_dynamic_client_registration_returns_credentials():
    """A valid registration response is normalized for the host adapter."""

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://auth.example.com/register"
        return httpx.Response(
            201,
            json={
                "client_id": "generated-client",
                "client_secret": "generated-secret",
                "redirect_uris": ["https://ha.example.com/auth/external/callback"],
                "token_endpoint_auth_method": "client_secret_post",
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await MCPOAuthClientRegistrar.async_register_mcp_oauth_client(
            MCPOAuthConfiguration(
                authorization_url="https://auth.example.com/authorize",
                token_url="https://auth.example.com/token",
                resource="https://mcp.example.com/api",
                issuer="https://auth.example.com/",
                registration_url="https://auth.example.com/register",
                scopes=("mcp",),
            ),
            "https://ha.example.com/auth/external/callback",
            client_name="Vault MCP",
            http_client=client,
        )

    assert result.client_id == "generated-client"
    assert result.client_secret == "generated-secret"


async def test_dynamic_client_registration_requires_advertised_endpoint():
    """The client does not guess or write to an unadvertised endpoint."""
    with pytest.raises(MCPAuthenticationError):
        await MCPOAuthClientRegistrar.async_register_mcp_oauth_client(
            MCPOAuthConfiguration(
                authorization_url="https://auth.example.com/authorize",
                token_url="https://auth.example.com/token",
                resource="https://mcp.example.com/api",
                issuer="https://auth.example.com/",
            ),
            "https://ha.example.com/auth/external/callback",
            client_name="Vault MCP",
        )


async def test_oauth_discovery_preserves_http_failure_cause():
    """Hosts can classify the HTTP reason when metadata discovery fails."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MCPAuthenticationError) as caught:
            await MCPOAuthClientRegistrar.async_discover_mcp_oauth(
                "https://mcp.example.com/api", http_client=client
            )

    assert isinstance(caught.value.__cause__, httpx.HTTPStatusError)
    assert caught.value.__cause__.response.status_code == 404
