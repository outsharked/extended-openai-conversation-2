"""Tests for the Streamable HTTP MCP client public API."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, call, patch

from mcp.types import (
    CallToolResult,
    Implementation,
    InitializeResult,
    ListToolsResult,
    PaginatedRequestParams,
    ServerCapabilities,
    TextContent,
    Tool,
    ToolAnnotations,
    ToolExecution,
)
import pytest

from custom_components.extended_openai_conversation.mcp_client import (
    MCPClient,
    MCPClientClosedError,
    MCPProtocolError,
    MCPServerConfig,
)

MODULE = "custom_components.extended_openai_conversation.mcp_client.mcp_client"


def _config() -> MCPServerConfig:
    """Return a valid server configuration."""
    return MCPServerConfig(
        server_id="storage",
        name="Storage",
        url="https://mcp.example.com/mcp",
        headers={"X-API-Key": "secret"},
    )


def _initialization() -> InitializeResult:
    """Return an SDK initialization result."""
    return InitializeResult(
        protocolVersion="2025-03-26",
        capabilities=ServerCapabilities(),
        serverInfo=Implementation(name="Example MCP", version="1.2.3"),
        instructions="Use tools carefully.",
    )


@asynccontextmanager
async def _transport(**kwargs):
    """Yield fake transport streams."""
    yield (MagicMock(), MagicMock(), MagicMock())


class _Session:
    """Controllable async MCP session double."""

    def __init__(self) -> None:
        self.initialize = AsyncMock(return_value=_initialization())
        self.list_tools = AsyncMock()
        self.call_tool = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None


async def test_check_connection_initializes_once_and_returns_public_model():
    """Connection checks initialize one session and hide SDK result types."""
    session = _Session()
    http_client = MagicMock()

    with (
        patch(f"{MODULE}.streamable_http_client", _transport),
        patch(f"{MODULE}.ClientSession", return_value=session),
    ):
        client = MCPClient(_config(), http_client=http_client)
        information = await client.check_connection()

    session.initialize.assert_awaited_once_with()
    assert information.name == "Example MCP"
    assert information.version == "1.2.3"
    assert information.protocol_version == "2025-03-26"
    assert information.instructions == "Use tools carefully."


async def test_operations_can_run_concurrently():
    """One slow MCP operation does not block another on the same server."""
    both_entered = asyncio.Event()
    release = asyncio.Event()
    entered = 0

    async def initialize():
        nonlocal entered
        entered += 1
        if entered == 2:
            both_entered.set()
        await release.wait()
        return _initialization()

    first_session = _Session()
    first_session.initialize.side_effect = initialize
    second_session = _Session()
    second_session.initialize.side_effect = initialize

    with (
        patch(f"{MODULE}.streamable_http_client", _transport),
        patch(
            f"{MODULE}.ClientSession",
            side_effect=[first_session, second_session],
        ),
    ):
        client = MCPClient(_config(), http_client=MagicMock())
        operations = [
            asyncio.create_task(client.check_connection()),
            asyncio.create_task(client.check_connection()),
        ]
        await asyncio.wait_for(both_entered.wait(), timeout=1)
        release.set()
        await asyncio.gather(*operations)


async def test_list_tools_exhausts_sdk_cursor_pages():
    """Catalogue discovery returns every SDK page through one initialized session."""
    session = _Session()
    session.list_tools.side_effect = [
        ListToolsResult(
            tools=[
                Tool(
                    name="upload_file",
                    title="Upload file",
                    description="Upload a file.",
                    inputSchema={"type": "object"},
                    outputSchema={"type": "object"},
                    annotations=ToolAnnotations(
                        readOnlyHint=False,
                        destructiveHint=True,
                        idempotentHint=False,
                        openWorldHint=True,
                    ),
                    execution=ToolExecution(taskSupport="optional"),
                )
            ],
            nextCursor="page-2",
        ),
        ListToolsResult(
            tools=[
                Tool(
                    name="download_file",
                    description="Download a file.",
                    inputSchema={"type": "object"},
                )
            ]
        ),
    ]

    with (
        patch(f"{MODULE}.streamable_http_client", _transport),
        patch(f"{MODULE}.ClientSession", return_value=session),
    ):
        tools = await MCPClient(_config(), http_client=MagicMock()).list_tools()

    session.initialize.assert_awaited_once_with()
    assert session.list_tools.await_args_list == [
        call(params=None),
        call(params=PaginatedRequestParams(cursor="page-2")),
    ]
    assert [tool.name for tool in tools] == ["upload_file", "download_file"]
    assert tools[0].input_schema == {"type": "object"}
    assert tools[0].output_schema == {"type": "object"}
    assert tools[0].annotations == {
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": True,
    }
    assert tools[0].task_support == "optional"


async def test_list_tools_rejects_repeated_cursor():
    """A broken server cannot trap catalogue discovery in an infinite loop."""
    session = _Session()
    session.list_tools.side_effect = [
        ListToolsResult(tools=[], nextCursor="repeat"),
        ListToolsResult(tools=[], nextCursor="repeat"),
    ]

    with (
        patch(f"{MODULE}.streamable_http_client", _transport),
        patch(f"{MODULE}.ClientSession", return_value=session),
        pytest.raises(MCPProtocolError),
    ):
        await MCPClient(_config(), http_client=MagicMock()).list_tools()


async def test_call_tool_returns_serializable_public_result():
    """Tool invocation copies arguments and converts SDK content to dictionaries."""
    session = _Session()
    session.call_tool.return_value = CallToolResult(
        content=[TextContent(type="text", text="uploaded")],
        structuredContent={"success": True},
        isError=False,
    )
    arguments = {"path": "/tmp/report.pdf"}

    with (
        patch(f"{MODULE}.streamable_http_client", _transport),
        patch(f"{MODULE}.ClientSession", return_value=session),
    ):
        result = await MCPClient(_config(), http_client=MagicMock()).call_tool(
            "upload_file", arguments
        )

    arguments["path"] = "changed"
    session.initialize.assert_awaited_once_with()
    assert session.call_tool.await_args.args == (
        "upload_file",
        {"path": "/tmp/report.pdf"},
    )
    assert result.content == ({"type": "text", "text": "uploaded"},)
    assert result.structured_content == {"success": True}
    assert not result.is_error


async def test_discover_and_call_tool_reuses_one_initialized_session():
    """Live authorization lookup and invocation share one MCP session."""
    session = _Session()
    session.list_tools.return_value = ListToolsResult(
        tools=[
            Tool(
                name="upload_file",
                description="Upload a file.",
                inputSchema={"type": "object"},
            )
        ]
    )
    session.call_tool.return_value = CallToolResult(
        content=[TextContent(type="text", text="uploaded")],
        isError=False,
    )

    with (
        patch(f"{MODULE}.streamable_http_client", _transport),
        patch(f"{MODULE}.ClientSession", return_value=session),
    ):
        result = await MCPClient(
            _config(), http_client=MagicMock()
        ).discover_and_call_tool(
            "upload_file",
            {"path": "/tmp/report.pdf"},
            tool_filter=lambda tools: tools,
        )

    session.initialize.assert_awaited_once_with()
    session.list_tools.assert_awaited_once_with(params=None)
    session.call_tool.assert_awaited_once_with(
        "upload_file",
        {"path": "/tmp/report.pdf"},
        read_timeout_seconds=300.0,
    )
    assert result is not None
    assert result.content == ({"type": "text", "text": "uploaded"},)


async def test_discover_and_call_tool_does_not_invoke_filtered_tool():
    """A tool rejected by host policy is not invoked after discovery."""
    session = _Session()
    session.list_tools.return_value = ListToolsResult(
        tools=[
            Tool(
                name="delete_file",
                description="Delete a file.",
                inputSchema={"type": "object"},
            )
        ]
    )

    with (
        patch(f"{MODULE}.streamable_http_client", _transport),
        patch(f"{MODULE}.ClientSession", return_value=session),
    ):
        result = await MCPClient(
            _config(), http_client=MagicMock()
        ).discover_and_call_tool(
            "delete_file",
            tool_filter=lambda tools: (),
        )

    assert result is None
    session.list_tools.assert_awaited_once_with(params=None)
    session.call_tool.assert_not_awaited()


async def test_close_does_not_close_injected_http_client():
    """The caller retains ownership of an injected HTTP client."""
    http_client = MagicMock(aclose=AsyncMock())
    client = MCPClient(_config(), http_client=http_client)

    await client.aclose()
    await client.aclose()

    http_client.aclose.assert_not_awaited()
    with pytest.raises(MCPClientClosedError):
        await client.check_connection()


async def test_close_closes_lazily_created_http_client():
    """The MCP client closes an HTTP client it created itself."""
    session = _Session()
    http_client = MagicMock(aclose=AsyncMock())

    with (
        patch(f"{MODULE}.httpx.AsyncClient", return_value=http_client),
        patch(f"{MODULE}.streamable_http_client", _transport),
        patch(f"{MODULE}.ClientSession", return_value=session),
    ):
        client = MCPClient(_config())
        await client.check_connection()
        await client.aclose()

    http_client.aclose.assert_awaited_once_with()


async def test_created_http_client_receives_injected_auth_provider():
    """Host OAuth providers are attached without coupling the client to its host."""
    session = _Session()
    http_client = MagicMock(aclose=AsyncMock())
    auth = MagicMock()

    with (
        patch(f"{MODULE}.httpx.AsyncClient", return_value=http_client) as client_class,
        patch(f"{MODULE}.streamable_http_client", _transport),
        patch(f"{MODULE}.ClientSession", return_value=session),
    ):
        await MCPClient(_config(), auth=auth).check_connection()

    assert client_class.call_args.kwargs["auth"] is auth
    assert client_class.call_args.kwargs["follow_redirects"] is True
