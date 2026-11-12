"""Tests for host-independent MCP server listing."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from custom_components.extended_openai_conversation.mcp_client import (
    MCPClient,
    MCPServerConfig,
    MCPServerInformation,
    MCPService,
    MCPServiceConfig,
    MCPTool,
    MCPToolManager,
    MCPToolResult,
)


def _tool(
    name: str,
    description: str,
    *,
    output_schema: dict[str, Any] | None = None,
    annotations: dict[str, Any] | None = None,
    task_support: str | None = None,
) -> MCPTool:
    return MCPTool(
        name=name,
        title=None,
        description=description,
        input_schema={
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
        },
        output_schema=output_schema,
        annotations=annotations,
        task_support=task_support,
    )


def _service(
    server_id: str,
    *,
    description: str = "",
    permanent_tools: tuple[MCPTool, ...] = (),
    on_demand_discovery: bool = True,
    disabled_tools: tuple[str, ...] = (),
    tool_name_pattern: str = "",
    tools: tuple[MCPTool, ...] = (),
    server_information: MCPServerInformation | None = None,
) -> MCPService:
    config = MCPServiceConfig(
        server=MCPServerConfig(
            server_id=server_id,
            name=f"{server_id.title()} Server",
            url=f"https://{server_id}.example.com/mcp",
            description=description,
        ),
        permanent_tools=permanent_tools,
        on_demand_discovery=on_demand_discovery,
        disabled_tools=disabled_tools,
        tool_name_pattern=tool_name_pattern,
        server_information=server_information,
    )
    client = MagicMock(spec=MCPClient)
    client.list_tools = AsyncMock(return_value=tools)
    return MCPService(
        config=config,
        client=client,
    )


async def test_list_servers_without_tool_fields_avoids_server_calls():
    """A metadata-only listing uses saved configuration without discovery."""
    storage = _service("storage", description="File storage")
    calendar = _service("calendar", description="Appointments")

    result = await MCPToolManager.list_servers(
        [storage, calendar],
        query="storage",
        search_fields=["server_id", "server_name"],
    )

    assert result == {
        "servers": [
            {
                "id": "storage",
                "name": "Storage Server",
                "description": "File storage",
            }
        ],
        "max_matches": 1,
        "current_page": 1,
        "max_pages": 1,
        "entries_per_page": 20,
    }
    storage.client.list_tools.assert_not_awaited()
    calendar.client.list_tools.assert_not_awaited()


async def test_list_servers_returns_cached_server_information_without_calling_server():
    """Cached initialization data is returned without a live connection."""
    cached = MCPServerInformation(
        name="Storage MCP",
        version="1.0.0",
        protocol_version="2025-06-18",
        instructions="Use storage tools carefully.",
        capabilities={"tools": {"listChanged": True}},
    )
    storage = _service("storage", server_information=cached)

    result = await MCPToolManager.list_servers(
        [storage], server_info="cached"
    )

    assert result["servers"][0]["server_info"] == cached.to_dict()
    assert result["servers"][0]["server_info_source"] == "cached"
    storage.client.check_connection.assert_not_awaited()


async def test_list_servers_latest_compares_only_servers_on_returned_page():
    """Live initialization checks occur after server-result pagination."""
    cached = MCPServerInformation(
        name="Storage MCP",
        version="1.0.0",
        protocol_version="2025-06-18",
        instructions=None,
        capabilities={"tools": {}},
    )
    latest = MCPServerInformation(
        name="Storage MCP",
        version="2.0.0",
        protocol_version="2025-06-18",
        instructions=None,
        capabilities={"tools": {}},
    )
    storage = _service("storage", server_information=cached)
    calendar = _service("calendar")
    storage.client.check_connection = AsyncMock(return_value=latest)

    result = await MCPToolManager.list_servers(
        [storage, calendar],
        server_info="latest",
        entries_per_page=1,
    )

    server = result["servers"][0]
    assert server["server_info"] == latest.to_dict()
    assert server["server_info_source"] == "latest"
    assert server["server_info_changed"] is True
    assert server["changed_fields"] == ["version"]
    storage.client.check_connection.assert_awaited_once_with()
    calendar.client.check_connection.assert_not_awaited()


async def test_list_servers_latest_retains_server_when_information_check_fails():
    """A live metadata failure is local to that server result."""
    storage = _service("storage")
    storage.client.check_connection = AsyncMock(side_effect=RuntimeError("offline"))

    result = await MCPToolManager.list_servers(
        [storage], server_info="latest"
    )

    assert result["servers"][0]["id"] == "storage"
    assert result["servers"][0]["server_info_error"] == {
        "code": "server_info_unavailable",
        "message": "Latest server information could not be retrieved",
    }


async def test_list_servers_rejects_invalid_server_information_mode():
    """Invalid server information modes fail without contacting servers."""
    storage = _service("storage")

    result = await MCPToolManager.list_servers(
        [storage], server_info="refresh"
    )

    assert result["error"]["code"] == "invalid_server_info"
    storage.client.check_connection.assert_not_awaited()


async def test_list_servers_searches_only_discoverable_non_permanent_tools():
    """Tool matching applies service policy and omits permanent tools."""
    upload = _tool("upload_file", "Upload a file")
    permanent = _tool("download_file", "Download a file")
    storage = _service(
        "storage",
        permanent_tools=(permanent,),
        tools=(upload, permanent),
    )

    result = await MCPToolManager.list_servers(
        [storage],
        query="file",
        search_fields=["tool_name"],
        include_tools=True,
    )

    assert result["servers"][0]["tools"] == [
        {"tool_name": "upload_file", "description": "Upload a file"}
    ]
    storage.client.list_tools.assert_awaited_once_with()


async def test_list_servers_controls_tool_descriptions_and_schemas():
    """Tool descriptions and schemas are independently optional."""
    output_schema = {
        "type": "object",
        "properties": {"success": {"type": "boolean"}},
    }
    storage = _service(
        "storage",
        tools=(
            _tool(
                "upload_file",
                "Upload a file",
                output_schema=output_schema,
            ),
        ),
    )

    result = await MCPToolManager.list_servers(
        [storage],
        include_tools=True,
        include_description=False,
        include_schema=True,
    )

    assert result["servers"][0]["tools"] == [
        {
            "tool_name": "upload_file",
            "input_schema": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
            },
            "output_schema": output_schema,
        }
    ]


async def test_list_servers_limits_nested_tool_results():
    """One server cannot bypass the response limit with a large tool list."""
    storage = _service(
        "storage",
        tools=tuple(_tool(f"tool_{index}", "Tool") for index in range(3)),
    )

    result = await MCPToolManager.list_servers(
        [storage], include_tools=True, entries_per_page=2
    )

    server = result["servers"][0]
    assert [tool["tool_name"] for tool in server["tools"]] == ["tool_0", "tool_1"]
    assert server["tool_matches"] == 3
    assert server["tools_returned"] == 2
    assert server["tools_truncated"] is True


async def test_list_servers_all_tools_implies_include_tools():
    """Requesting all tools enables a bounded nested tool preview."""
    storage = _service(
        "storage",
        tools=tuple(_tool(f"tool_{index}", "Tool") for index in range(3)),
    )

    result = await MCPToolManager.list_servers(
        [storage],
        query="storage",
        search_fields=["server_id"],
        all_tools=True,
        entries_per_page=2,
    )

    server = result["servers"][0]
    assert [tool["tool_name"] for tool in server["tools"]] == ["tool_0", "tool_1"]
    assert server["tool_matches"] == 3
    assert server["tools_returned"] == 2
    assert server["tools_truncated"] is True
    storage.client.list_tools.assert_awaited_once_with()


async def test_list_servers_reports_metadata_match_without_tool_match():
    """A server metadata match remains visible with an explanatory empty list."""
    storage = _service("storage", tools=(_tool("upload_file", "Upload a file"),))

    result = await MCPToolManager.list_servers(
        [storage], query="storage", include_tools=True
    )

    server = result["servers"][0]
    assert server["tools"] == []
    assert "no discoverable tools matched" in server["tool_match_message"]


async def test_list_servers_paginates_with_a_hard_page_size_limit():
    """Pagination reports totals and rejects more than 100 entries per page."""
    services = [_service(f"server-{index}") for index in range(3)]

    result = await MCPToolManager.list_servers(
        services, current_page=2, entries_per_page=2
    )
    invalid = await MCPToolManager.list_servers(services, entries_per_page=101)

    assert [server["id"] for server in result["servers"]] == ["server-2"]
    assert result["max_matches"] == 3
    assert result["max_pages"] == 2
    assert invalid["error"]["code"] == "invalid_entries_per_page"


async def test_list_servers_regex_searches_schema_values():
    """Regex search inspects meaningful schema fields using search semantics."""
    storage = _service(
        "storage", tools=(_tool("upload_file", "Upload a file"),)
    )

    result = await MCPToolManager.list_servers(
        [storage],
        query="file_pa.*",
        search_fields=["schema"],
        search_regex=True,
        include_tools=True,
    )

    assert result["servers"][0]["id"] == "storage"
    assert result["servers"][0]["tools"][0]["tool_name"] == "upload_file"


async def test_list_tools_scopes_query_to_selected_authorized_servers():
    """A scoped tool search contacts only selected authorized services."""
    storage = _service(
        "storage", tools=(_tool("upload_file", "Upload a file"),)
    )
    calendar = _service(
        "calendar", tools=(_tool("create_event", "Create an event"),)
    )

    result = await MCPToolManager.list_tools(
        [storage, calendar],
        server_ids=["storage", "not-assigned"],
        query="upload",
        search_fields=["tool_name"],
        include_description=False,
        include_schema=True,
    )

    assert result["tools"] == [
        {
            "tool_name": "upload_file",
            "input_schema": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
            },
            "server_id": "storage",
        }
    ]
    storage.client.list_tools.assert_awaited_once_with()
    calendar.client.list_tools.assert_not_awaited()


async def test_list_tools_returns_empty_for_unassigned_server():
    """An unknown or unassigned server is indistinguishable from no match."""
    storage = _service(
        "storage", tools=(_tool("upload_file", "Upload a file"),)
    )

    result = await MCPToolManager.list_tools(
        [storage], server_ids=["not-assigned"]
    )

    assert result == {
        "tools": [],
        "max_matches": 0,
        "current_page": 1,
        "max_pages": 0,
        "entries_per_page": 20,
    }
    storage.client.list_tools.assert_not_awaited()


async def test_list_tools_invalid_regex_avoids_server_calls():
    """An invalid request regex is rejected before live discovery."""
    storage = _service("storage")

    result = await MCPToolManager.list_tools(
        [storage], query="(", search_regex=True
    )

    assert result["error"]["code"] == "invalid_regex"
    storage.client.list_tools.assert_not_awaited()


async def test_list_tools_excludes_permanent_tools_and_paginates_globally():
    """Discovery omits permanent tools before paginating the flattened result."""
    permanent = _tool("permanent_tool", "Permanent")
    storage = _service(
        "storage",
        permanent_tools=(permanent,),
        tools=(permanent, _tool("tool_one", "One"), _tool("tool_two", "Two")),
    )

    result = await MCPToolManager.list_tools(
        [storage], current_page=2, entries_per_page=1
    )

    assert result["tools"][0]["tool_name"] == "tool_two"
    assert result["max_matches"] == 2
    assert result["max_pages"] == 2


async def test_get_tool_details_returns_permanent_details_without_server_call():
    """Permanent tool details come from saved context without discovery."""
    output_schema = {
        "type": "object",
        "properties": {"result": {"type": "string"}},
    }
    permanent = _tool(
        "permanent_tool",
        "Permanent",
        output_schema=output_schema,
        annotations={"readOnlyHint": True, "destructiveHint": False},
        task_support="optional",
    )
    storage = _service(
        "storage",
        permanent_tools=(permanent,),
        on_demand_discovery=False,
        disabled_tools=("permanent_tool",),
        tool_name_pattern="different_tool",
    )

    result = await MCPToolManager.get_tool_details(
        [storage], server_id="storage", tool_name="permanent_tool"
    )

    assert result == {
        "server_id": "storage",
        "tool_name": "permanent_tool",
        "description": "Permanent",
        "input_schema": permanent.input_schema,
        "output_schema": output_schema,
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
        },
        "execution": {"task_support": "optional"},
    }
    storage.client.list_tools.assert_not_awaited()


async def test_get_tool_details_discovers_non_permanent_tool():
    """A discoverable tool's details are retrieved from the live catalogue."""
    upload = _tool("upload_file", "Upload a file")
    storage = _service("storage", tools=(upload,))

    result = await MCPToolManager.get_tool_details(
        [storage], server_id="storage", tool_name="upload_file"
    )

    assert result["input_schema"] == upload.input_schema
    assert "title" not in result
    assert "output_schema" not in result
    assert "annotations" not in result
    assert "execution" not in result
    storage.client.list_tools.assert_awaited_once_with()


async def test_get_tool_details_hides_unassigned_server():
    """An unassigned server returns the same result as an unknown tool."""
    storage = _service("storage")

    result = await MCPToolManager.get_tool_details(
        [storage], server_id="not-assigned", tool_name="upload_file"
    )

    assert result == {
        "error": {"code": "tool_not_found", "message": "Tool was not found"}
    }
    storage.client.list_tools.assert_not_awaited()


async def test_get_tool_details_hides_tool_when_discovery_is_disabled():
    """Non-permanent tools cannot be retrieved without on-demand discovery."""
    storage = _service(
        "storage",
        on_demand_discovery=False,
        tools=(_tool("upload_file", "Upload a file"),),
    )

    result = await MCPToolManager.get_tool_details(
        [storage], server_id="storage", tool_name="upload_file"
    )

    assert result["error"]["code"] == "tool_not_found"
    storage.client.list_tools.assert_not_awaited()


async def test_get_tool_details_validates_identifiers():
    """Invalid identifiers are rejected before examining services."""
    assert (
        await MCPToolManager.get_tool_details([], server_id="", tool_name="tool")
    )["error"]["code"] == "invalid_server_id"
    assert (
        await MCPToolManager.get_tool_details([], server_id="server", tool_name="")
    )["error"]["code"] == "invalid_tool_name"


async def test_execute_tool_runs_permanent_tool_without_discovery():
    """A permanent tool executes directly from its saved definition."""
    permanent = _tool("permanent_tool", "Permanent")
    storage = _service(
        "storage",
        permanent_tools=(permanent,),
        on_demand_discovery=False,
    )
    storage.client.call_tool.return_value = MCPToolResult(
        content=({"type": "text", "text": "done"},),
        structured_content={"success": True},
        is_error=False,
    )

    result = await MCPToolManager.execute_tool(
        [storage],
        server_id="storage",
        tool_name="permanent_tool",
        arguments={"value": 1},
    )

    assert result == {
        "server_id": "storage",
        "tool_name": "permanent_tool",
        "content": [{"type": "text", "text": "done"}],
        "structured_content": {"success": True},
        "is_error": False,
    }
    storage.client.list_tools.assert_not_awaited()
    storage.client.call_tool.assert_awaited_once_with(
        "permanent_tool", {"value": 1}
    )


async def test_execute_tool_discovers_non_permanent_tool():
    """A non-permanent tool must pass live discovery before execution."""
    upload = _tool("upload_file", "Upload a file")
    storage = _service("storage", tools=(upload,))
    storage.client.call_tool.return_value = MCPToolResult((), None, False)

    result = await MCPToolManager.execute_tool(
        [storage], server_id="storage", tool_name="upload_file"
    )

    assert result["is_error"] is False
    storage.client.list_tools.assert_awaited_once_with()
    storage.client.call_tool.assert_awaited_once_with("upload_file", None)


async def test_execute_tool_hides_unassigned_server():
    """Execution does not reveal whether an unassigned server exists."""
    storage = _service("storage")

    result = await MCPToolManager.execute_tool(
        [storage], server_id="not-assigned", tool_name="upload_file"
    )

    assert result["error"]["code"] == "tool_not_found"
    storage.client.list_tools.assert_not_awaited()
    storage.client.call_tool.assert_not_awaited()


async def test_execute_tool_rejects_undiscoverable_tool():
    """A non-permanent tool cannot execute when discovery is disabled."""
    storage = _service(
        "storage",
        on_demand_discovery=False,
        tools=(_tool("upload_file", "Upload a file"),),
    )

    result = await MCPToolManager.execute_tool(
        [storage], server_id="storage", tool_name="upload_file"
    )

    assert result["error"]["code"] == "tool_not_found"
    storage.client.list_tools.assert_not_awaited()
    storage.client.call_tool.assert_not_awaited()


async def test_execute_tool_validates_arguments():
    """Tool arguments must be an object with string keys."""
    storage = _service("storage")

    result = await MCPToolManager.execute_tool(
        [storage],
        server_id="storage",
        tool_name="upload_file",
        arguments=MagicMock(),
    )

    assert result["error"]["code"] == "invalid_arguments"
    storage.client.list_tools.assert_not_awaited()
    storage.client.call_tool.assert_not_awaited()
