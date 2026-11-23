"""MCP server configuration for the integration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import re
from typing import Any
from urllib.parse import urlparse

from .mcp_errors import MCPConfigError

_SERVER_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_HEADER_NAME_PATTERN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")


def _validate_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Return a validated copy of HTTP headers."""
    if not isinstance(headers, Mapping):
        raise MCPConfigError("headers", "invalid_headers")

    validated: dict[str, str] = {}
    for name, value in headers.items():
        if (
            not isinstance(name, str)
            or not _HEADER_NAME_PATTERN.fullmatch(name)
            or not isinstance(value, str)
            or "\r" in value
            or "\n" in value
        ):
            raise MCPConfigError("headers", "invalid_headers")
        validated[name] = value
    return validated


@dataclass(frozen=True, slots=True)
class MCPServerConfig:
    """Validated configuration for one Streamable HTTP MCP server."""

    server_id: str
    name: str
    url: str
    description: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)
    use_oauth: bool = False

    def __post_init__(self) -> None:
        """Validate and normalise configuration supplied by a host application."""
        server_id = self.server_id.strip()
        name = self.name.strip()
        url = self.url.strip()
        description = self.description.strip()

        if not _SERVER_ID_PATTERN.fullmatch(server_id):
            raise MCPConfigError("server_id", "invalid_server_id")
        if not name:
            raise MCPConfigError("name", "invalid_name")

        parsed_url = urlparse(url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise MCPConfigError("url", "invalid_url")

        headers = _validate_headers(self.headers)
        if not isinstance(self.use_oauth, bool):
            raise MCPConfigError("use_oauth", "invalid_oauth_setting")

        object.__setattr__(self, "server_id", server_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "url", url)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "headers", headers)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MCPServerConfig:
        """Build a configuration from the library's public mapping format."""
        return cls(
            server_id=data["server_id"],
            name=data["name"],
            url=data["url"],
            description=data.get("description", ""),
            headers=data.get("headers", {}),
            use_oauth=data.get("use_oauth", False),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the library's public, serialisable mapping format."""
        return {
            "server_id": self.server_id,
            "name": self.name,
            "url": self.url,
            "description": self.description,
            "headers": dict(self.headers),
            "use_oauth": self.use_oauth,
        }
