"""Framework-independent MCP OAuth metadata discovery."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urljoin

import httpx2 as httpx
from mcp.client.auth import OAuthRegistrationError
from mcp.client.auth.utils import (
    build_oauth_authorization_server_metadata_discovery_urls,
    build_protected_resource_metadata_discovery_urls,
    handle_registration_response,
)
from mcp.shared.auth import (
    OAuthClientMetadata,
    OAuthMetadata,
    ProtectedResourceMetadata,
)
from pydantic import ValidationError

from .mcp_errors import MCPAuthenticationError


@dataclass(frozen=True, slots=True)
class MCPOAuthConfiguration:
    """OAuth endpoints and scopes advertised for an MCP resource."""

    authorization_url: str
    token_url: str
    resource: str
    issuer: str
    registration_url: str | None = None
    scopes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MCPOAuthClientRegistration:
    """Credentials returned by dynamic OAuth client registration."""

    client_id: str
    client_secret: str = ""

class MCPOAuthClientRegistrar:

    @staticmethod
    async def async_register_mcp_oauth_client(
        oauth_config: MCPOAuthConfiguration,
        redirect_uri: str,
        *,
        client_name: str,
        http_client: httpx.AsyncClient | None = None,
    ) -> MCPOAuthClientRegistration:
        """Dynamically register a public PKCE client with an MCP auth server."""
        if oauth_config.registration_url is None:
            raise MCPAuthenticationError("The MCP authorization server does not advertise client registration")

        metadata = OAuthClientMetadata.model_validate(
            {
                "redirect_uris": [redirect_uri],
                "token_endpoint_auth_method": "none",
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "scope": " ".join(oauth_config.scopes) or None,
                "client_name": client_name,
            }
        )
        owns_client = http_client is None
        client = http_client or httpx.AsyncClient(follow_redirects=False)
        try:
            response = await client.post(
                oauth_config.registration_url,
                json=metadata.model_dump(mode="json", exclude_none=True),
                headers={"Content-Type": "application/json"},
            )
            client_info = await handle_registration_response(response)
        except (httpx.HTTPError, OAuthRegistrationError, ValueError) as err:
            raise MCPAuthenticationError(
                "Unable to register an OAuth client with the MCP authorization server"
            ) from err
        finally:
            if owns_client:
                await client.aclose()

        if client_info.token_endpoint_auth_method not in (
            None,
            "none",
            "client_secret_post",
        ):
            raise MCPAuthenticationError(
                "The MCP authorization server selected an unsupported client authentication method"
            )
        if not client_info.client_id:
            raise MCPAuthenticationError(
                "The MCP authorization server did not return a client ID"
            )
        return MCPOAuthClientRegistration(
            client_id=client_info.client_id,
            client_secret=client_info.client_secret or "",
        )

    @staticmethod
    async def async_discover_mcp_oauth(
        server_url: str,
        *,
        resource_metadata_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> MCPOAuthConfiguration:
        """Discover OAuth metadata using the MCP authorization specification."""
        owns_client = http_client is None
        client = http_client or httpx.AsyncClient(follow_redirects=False)
        try:
            resource_metadata = await MCPOAuthClientRegistrar._async_discover_resource_metadata(
                client, server_url, resource_metadata_url
            )
            authorization_server = (
                str(resource_metadata.authorization_servers[0])
                if resource_metadata is not None and resource_metadata.authorization_servers
                else None
            )
            oauth_metadata = await MCPOAuthClientRegistrar._async_discover_authorization_metadata(
                client, server_url, authorization_server
            )
        finally:
            if owns_client:
                await client.aclose()

        scopes = (
            resource_metadata.scopes_supported
            if resource_metadata is not None
            and resource_metadata.scopes_supported is not None
            else oauth_metadata.scopes_supported or []
        )
        endpoint_base = authorization_server or server_url
        return MCPOAuthConfiguration(
            authorization_url=urljoin(
                endpoint_base, str(oauth_metadata.authorization_endpoint)
            ),
            token_url=urljoin(endpoint_base, str(oauth_metadata.token_endpoint)),
            resource=server_url,
            issuer=str(oauth_metadata.issuer),
            registration_url=(
                str(oauth_metadata.registration_endpoint)
                if oauth_metadata.registration_endpoint is not None
                else None
            ),
            scopes=tuple(scopes),
        )

    @staticmethod
    async def _async_discover_resource_metadata(
        client: httpx.AsyncClient,
        server_url: str,
        resource_metadata_url: str | None,
    ) -> ProtectedResourceMetadata | None:
        """Return the first valid protected-resource metadata response, if any."""
        urls = build_protected_resource_metadata_discovery_urls(
            resource_metadata_url, server_url
        )
        for url in urls:
            try:
                response = await client.get(url)
            except httpx.HTTPError:
                continue
            if response.status_code != 200:
                continue
            try:
                return ProtectedResourceMetadata.model_validate_json(response.content)
            except ValidationError:
                continue
        return None

    @staticmethod
    async def _async_discover_authorization_metadata(
        client: httpx.AsyncClient,
        server_url: str,
        authorization_server: str | None,
    ) -> OAuthMetadata:
        """Return authorization-server metadata or raise a useful public error."""
        urls = build_oauth_authorization_server_metadata_discovery_urls(
            authorization_server, server_url
        )
        failures: list[httpx.HTTPError] = []
        for url in urls:
            try:
                response = await client.get(url)
            except httpx.HTTPError as err:
                failures.append(err)
                continue
            if response.status_code != 200:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as err:
                    failures.append(err)
                continue
            try:
                return OAuthMetadata.model_validate_json(response.content)
            except ValidationError:
                continue
        error = MCPAuthenticationError(
            "Unable to discover the MCP authorization server"
        )
        if failures:
            raise error from MCPOAuthClientRegistrar._preferred_discovery_failure(failures)
        raise error

    @staticmethod
    def _preferred_discovery_failure(failures: list[httpx.HTTPError]) -> httpx.HTTPError:
        """Choose the most useful failed discovery response for the host UI."""
        priorities = {401: 0, 403: 1, 500: 2, 404: 3}

        def priority(error: httpx.HTTPError) -> int:
            if not isinstance(error, httpx.HTTPStatusError):
                return 4
            status = error.response.status_code
            if 500 <= status <= 599:
                status = 500
            return priorities.get(status, 5)

        return min(failures, key=priority)
