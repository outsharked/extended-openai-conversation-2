"""Framework-independent MCP client configuration flow processing."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

from .config import MCPServerConfig
from .mcp_client import MCPClient
from .mcp_datatypes import MCPDiscoveryResult, MCPServerInformation
from .mcp_errors import MCPConfigError

DEFAULT_MCP_SERVER_NAME = "MCP Server"

_REQUIRED_TEXT_FIELDS = {
    "name": "invalid_name",
    "server_id": "invalid_server_id",
    "url": "invalid_url",
}


@dataclass(frozen=True, slots=True)
class MCPClientConfigFlowResult:
    """Result for a host application to render or persist."""

    values: dict[str, Any]
    errors: dict[str, str]
    config: MCPServerConfig | None = None

    @property
    def is_complete(self) -> bool:
        """Return whether validation produced a configuration."""
        return self.config is not None


class MCPClientConfigFlow:
    """Validate MCP configuration without depending on a UI framework."""

    def __init__(self, existing_server_ids: Collection[str] = ()) -> None:
        """Initialize processing with the server IDs reserved by the host."""
        self._existing_server_ids = frozenset(existing_server_ids)

    def _generate_server_id(self, base_id: str = "mcp_server") -> str:
        """Generate a readable server ID that does not collide with existing IDs."""
        candidate = base_id
        suffix = 2
        while candidate in self._existing_server_ids:
            candidate = f"{base_id}_{suffix}"
            suffix += 1
        return candidate

    def process(self, user_input: Mapping[str, Any] | None = None) -> MCPClientConfigFlowResult:
        """Return initial form values or process submitted connection data."""
        if user_input is None:
            return MCPClientConfigFlowResult(
                values={
                    "name": DEFAULT_MCP_SERVER_NAME,
                    "server_id": self._generate_server_id(),
                    "description": "",
                    "headers": {},
                    "use_oauth": False,
                },
                errors={},
            )

        values = dict(user_input)
        for field, error_code in _REQUIRED_TEXT_FIELDS.items():
            if not isinstance(values.get(field), str):
                return MCPClientConfigFlowResult(
                    values=values,
                    errors={field: error_code},
                )

        try:
            config = MCPServerConfig.from_dict(values)
        except MCPConfigError as err:
            return MCPClientConfigFlowResult(
                values=values,
                errors={err.field: err.code},
            )

        if config.server_id in self._existing_server_ids:
            return MCPClientConfigFlowResult(
                values=values,
                errors={"server_id": "duplicate_server_id"},
            )

        return MCPClientConfigFlowResult(
            values=config.to_dict(),
            errors={},
            config=config,
        )

    async def discover(self, config: MCPServerConfig, *, auth: Any | None = None) -> MCPDiscoveryResult:
        """Retrieve the server and tools catalogue."""
        async with MCPClient(config, auth=auth) as client:
            return await client.discover()

    async def check_connection(self, config: MCPServerConfig, *, auth: Any | None = None) -> MCPServerInformation:
        """Initialize the server without retrieving its tool catalogue."""
        async with MCPClient(config, auth=auth) as client:
            return await client.check_connection()
