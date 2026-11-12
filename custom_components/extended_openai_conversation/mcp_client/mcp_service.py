
from dataclasses import dataclass

from .mcp_client import MCPClient
from .service_config import MCPServiceConfig


@dataclass(frozen=True, slots=True)
class MCPService:
    """One loaded MCP service and its host-independent configuration."""

    config: MCPServiceConfig
    client: MCPClient
