"""Streamable HTTP MCP client lifecycle and operations."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import httpx2 as httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import (
    CallToolResult,
    InitializeResult,
    ListToolsResult,
    PaginatedRequestParams,
    Tool,
)

from .config import MCPServerConfig
from .mcp_datatypes import (
    MCPDiscoveryResult,
    MCPServerInformation,
    MCPTool,
    MCPToolResult,
)
from .mcp_errors import MCPClientClosedError, MCPProtocolError

DEFAULT_CONNECT_TIMEOUT = 30.0
DEFAULT_READ_TIMEOUT = 300.0


class MCPClient:
    """Access one Streamable HTTP MCP server.

    MCP protocol sessions are short lived and initialized once per operation.
    The underlying HTTP client is reused across operations for connection pooling.
    """

    def __init__(
        self,
        config: MCPServerConfig,
        *,
        http_client: httpx.AsyncClient | None = None,
        auth: httpx.Auth | None = None,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = DEFAULT_READ_TIMEOUT,
    ) -> None:
        """Initialize an MCP client without opening a remote connection."""
        if connect_timeout <= 0 or read_timeout <= 0:
            raise ValueError("MCP client timeouts must be greater than zero")

        self.config: MCPServerConfig = config
        self._http_client: httpx.AsyncClient | None = http_client
        self._auth: httpx.Auth | None = auth
        self._owns_http_client: bool = http_client is None
        self._http_timeout: httpx.Timeout = httpx.Timeout(
            connect_timeout,
            connect=connect_timeout,
            read=read_timeout,
            write=connect_timeout,
            pool=connect_timeout,
        )
        self._session_read_timeout: float = read_timeout
        self._operation_lock: asyncio.Lock = asyncio.Lock()
        self._closed: bool = False

    async def __aenter__(self) -> MCPClient:
        self._ensure_open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        await self.aclose()

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[tuple[ClientSession, InitializeResult]]:
        """Open and initialize one short-lived MCP protocol session."""
        http_client: httpx.AsyncClient = self._get_http_client()

        async with (
            streamable_http_client(url=self.config.url, http_client=http_client) as streams,
            ClientSession(streams[0], streams[1], read_timeout_seconds=self._session_read_timeout) as session
        ):
            initialization = await session.initialize()
            yield session, initialization

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                headers=dict(self.config.headers),
                auth=self._auth,
                timeout=self._http_timeout,
                follow_redirects=False,
            )
        return self._http_client

    def _ensure_open(self) -> None:
        """Reject operations after the client has been closed."""
        if self._closed:
            raise MCPClientClosedError("MCP client is closed")

    def _server_information(self, result: InitializeResult) -> MCPServerInformation:
        """Convert an SDK initialization result to the public model."""
        return MCPServerInformation(
            name=result.server_info.name,
            version=result.server_info.version,
            protocol_version=str(result.protocol_version),
            instructions=result.instructions,
            capabilities=result.capabilities.model_dump(
                mode="json", by_alias=True, exclude_none=True
            ),
        )

    def _tool_definition(self, tool: Tool) -> MCPTool:
        """Return serializable tool definitions"""
        annotations = (
            tool.annotations.model_dump(mode="json", by_alias=True, exclude_none=True)
            if tool.annotations is not None
            else None
        )
        return MCPTool(
            name=tool.name,
            title=tool.title,
            description=tool.description,
            input_schema=dict(tool.input_schema),
            output_schema=(
                dict(tool.output_schema) if tool.output_schema is not None else None
            ),
            annotations=annotations or None,
            task_support=(
                tool.execution.task_support if tool.execution is not None else None
            ),
        )

    def _tool_result(self, result: CallToolResult) -> MCPToolResult:
        """Convert an SDK tool result to the public serializable model."""
        return MCPToolResult(
            content=tuple(
                item.model_dump(mode="json", by_alias=True, exclude_none=True)
                for item in result.content
            ),
            structured_content=(
                dict(result.structured_content)
                if isinstance(result.structured_content, Mapping)
                else None
            ),
            is_error=result.is_error,
        )

    async def _list_session_tools(self, session: ClientSession) -> tuple[MCPTool, ...]:
        """Return the complete advertised tool catalogue across all SDK pages."""
        tools: list[MCPTool] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            params = (
                PaginatedRequestParams(cursor=cursor) if cursor is not None else None
            )
            result: ListToolsResult = await session.list_tools(params=params)
            tools.extend(self._tool_definition(tool) for tool in result.tools)

            next_cursor: str | None = result.next_cursor
            if next_cursor is None:
                return tuple(tools)
            if next_cursor in seen_cursors:
                raise MCPProtocolError("MCP server repeated a tools/list pagination cursor")

            seen_cursors.add(next_cursor)
            cursor = next_cursor

    async def discover(self) -> MCPDiscoveryResult:
        """Initialize once and return server metadata and its complete catalogue."""
        async with self._operation_lock:
            self._ensure_open()
            async with self._session() as (session, initialization):
                return MCPDiscoveryResult(
                    server=self._server_information(initialization),
                    tools=await self._list_session_tools(session),
                )

    async def check_connection(self) -> MCPServerInformation:
        """Connect, initialize once, and return information about the server."""
        async with self._operation_lock:
            self._ensure_open()
            async with self._session() as (_, initialization):
                return self._server_information(initialization)

    async def list_tools(self) -> tuple[MCPTool, ...]:
        """Return the complete advertised tool catalogue."""
        async with self._operation_lock:
            self._ensure_open()
            async with self._session() as (session, _):
                return await self._list_session_tools(session)

    async def call_tool(self, tool_name: str, arguments: Mapping[str, Any] | None = None) -> MCPToolResult:
        """Invoke a tool once without automatically retrying side effects."""
        async with self._operation_lock:
            self._ensure_open()
            async with self._session() as (session, _):
                result = await session.call_tool(
                    tool_name,
                    dict(arguments) if arguments is not None else {},
                    read_timeout_seconds=self._session_read_timeout,
                )
                return self._tool_result(result)

    async def aclose(self) -> None:
        """Close the owned HTTP client after any active operation completes."""
        async with self._operation_lock:
            if self._closed:
                return
            self._closed = True

            if self._owns_http_client and self._http_client is not None:
                await self._http_client.aclose()
            self._http_client = None
