"""Errors raised by the MCP Clients"""
class MCPClientError(Exception):
    """Base exception raised by the public MCP client API."""

class MCPClientClosedError(MCPClientError):
    """The client has already been closed."""

class MCPProtocolError(MCPClientError):
    """The remote server returned an invalid MCP protocol sequence."""

class MCPAuthenticationError(MCPClientError):
    """The MCP server rejected authentication."""

class MCPConfigError(ValueError):
    """An invalid MCP server configuration value."""

    def __init__(self, field: str, code: str) -> None:
        super().__init__(code)
        self.field = field
        self.code = code
