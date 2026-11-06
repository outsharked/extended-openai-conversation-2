"""Tests for the MCP function dispatcher."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx2 as httpx
import pytest

from custom_components.extended_openai_conversation.functions import MCPFunction
from custom_components.extended_openai_conversation.functions.mcp import (
    build_mcp_function_tool,
)
from custom_components.extended_openai_conversation.mcp_client import MCPToolManager


def test_mcp_tool_schema_documents_actions_search_and_defaults():
    """The shared dispatcher explains action-specific inputs to the model."""
    tool = build_mcp_function_tool(["weather"])
    spec = tool["spec"]
    properties = spec["parameters"]["properties"]

    assert "get_tool_details to retrieve" in spec["description"]
    assert "require server_id and tool_name" in properties["action"]["description"]
    assert "cannot expand" in properties["server_ids"]["description"]
    assert "specific operation" in properties["server_id"]["description"]
    assert "including a tool omitted from discovery results" in properties[
        "tool_name"
    ]["description"]
    assert "May be omitted" in properties["arguments"]["description"]
    assert properties["search_fields"]["items"]["enum"] == [
        "server_id",
        "server_name",
        "tool_name",
        "description",
        "schema",
    ]
    assert "list_tools accepts only" in properties["search_fields"]["description"]
    assert "Defaults to false" in properties["search_regex"]["description"]
    assert "Defaults to true" in properties["include_description"]["description"]
    assert "Defaults to 1" in properties["current_page"]["description"]
    assert "Defaults to 20" in properties["entries_per_page"]["description"]
    assert "hard maximum of 100" in properties["entries_per_page"]["description"]
    assert "Implies include_tools=true" in properties["all_tools"]["description"]
    assert "per-server result limit" in properties["all_tools"]["description"]
    assert "paginate further" in properties["all_tools"]["description"]
    assert properties["server_info"]["enum"] == ["cached", "latest"]
    assert "returned page" in properties["server_info"]["description"]


@pytest.mark.parametrize(
    ("action", "arguments", "method_name", "expected_arguments"),
    [
        (
            "list_servers",
            {
                "action": "list_servers",
                "query": "weather",
                "include_tools": True,
                "server_info": "cached",
                "ignored": True,
            },
            "list_servers",
            {
                "query": "weather",
                "include_tools": True,
                "server_info": "cached",
            },
        ),
        (
            "list_tools",
            {
                "action": "list_tools",
                "server_ids": ["weather"],
                "query": "forecast",
                "include_schema": True,
                "ignored": True,
            },
            "list_tools",
            {
                "server_ids": ["weather"],
                "query": "forecast",
                "include_schema": True,
            },
        ),
        (
            "get_tool_details",
            {
                "action": "get_tool_details",
                "server_id": "weather",
                "tool_name": "forecast",
                "ignored": True,
            },
            "get_tool_details",
            {"server_id": "weather", "tool_name": "forecast"},
        ),
        (
            "execute_tool",
            {
                "action": "execute_tool",
                "server_id": "weather",
                "tool_name": "forecast",
                "arguments": {"location": "London"},
                "ignored": True,
            },
            "execute_tool",
            {
                "server_id": "weather",
                "tool_name": "forecast",
                "arguments": {"location": "London"},
            },
        ),
    ],
)
async def test_execute_routes_action(
    hass,
    llm_context,
    exposed_entities,
    action,
    arguments,
    method_name,
    expected_arguments,
):
    """Test actions are scoped and forwarded to the matching operation."""
    services = [MagicMock()]
    scope = MagicMock()
    scope.list_loaded.return_value = services
    manager = MagicMock()
    manager.scope.return_value = scope
    operation = AsyncMock(return_value={"action": action})

    with (
        patch(
            "custom_components.extended_openai_conversation.functions.mcp.MCPServiceManager",
            return_value=manager,
        ),
        patch.object(MCPToolManager, method_name, operation),
    ):
        result = await MCPFunction().execute(
            hass,
            {"type": "mcp", "server_ids": ["weather", "home"]},
            arguments,
            llm_context,
            exposed_entities,
        )

    assert result == {"action": action}
    manager.scope.assert_called_once_with(["weather", "home"])
    scope.list_loaded.assert_called_once_with()
    operation.assert_awaited_once_with(services, **expected_arguments)


@pytest.mark.parametrize("arguments", [{}, {"action": "unknown"}])
async def test_execute_rejects_invalid_action(
    hass, llm_context, exposed_entities, arguments
):
    """Test missing and unsupported actions return a structured error."""
    with patch(
        "custom_components.extended_openai_conversation.functions.mcp.MCPServiceManager"
    ) as manager:
        result = await MCPFunction().execute(
            hass,
            {"type": "mcp", "server_ids": ["weather"]},
            arguments,
            llm_context,
            exposed_entities,
        )

    assert result == {
        "error": {
            "code": "invalid_action",
            "message": "Unsupported MCP action",
        }
    }
    manager.assert_not_called()


@pytest.mark.parametrize(
    ("status", "code", "message"),
    [
        (
            401,
            "authentication_failed",
            "The MCP server rejected the supplied credentials",
        ),
        (403, "access_denied", "The MCP server denied access"),
    ],
)
async def test_execute_returns_nested_http_authentication_failures(
    hass, llm_context, exposed_entities, status, code, message
):
    """Transport authentication failures remain ordinary tool results."""
    request = httpx.Request("POST", "https://mcp.example.com/api")
    response = httpx.Response(status, request=request)
    transport_error = httpx.HTTPStatusError(
        "request failed", request=request, response=response
    )
    operation = AsyncMock(
        side_effect=ExceptionGroup("MCP transport failed", [transport_error])
    )
    scope = MagicMock()
    scope.list_loaded.return_value = [MagicMock()]
    manager = MagicMock()
    manager.scope.return_value = scope

    with (
        patch(
            "custom_components.extended_openai_conversation.functions.mcp."
            "MCPServiceManager",
            return_value=manager,
        ),
        patch.object(MCPToolManager, "list_tools", operation),
    ):
        result = await MCPFunction().execute(
            hass,
            {"type": "mcp", "server_ids": ["weather"]},
            {"action": "list_tools"},
            llm_context,
            exposed_entities,
        )

    assert result == {"error": {"code": code, "message": message}}
