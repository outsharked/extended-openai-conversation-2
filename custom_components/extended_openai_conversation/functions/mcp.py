"""LLM-callable MCP function dispatcher."""

from __future__ import annotations

import logging
from typing import Any

import httpx2 as httpx
import probatio as vol

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from ..mcp_client import MCPToolManager
from ..mcp_service_manager import MCPServiceManager
from .base import Function

_LOGGER = logging.getLogger(__name__)

_LIST_SERVERS_ARGUMENTS = (
    "query",
    "search_fields",
    "search_regex",
    "search_case_sensitive",
    "include_tools",
    "all_tools",
    "include_description",
    "include_schema",
    "server_info",
    "current_page",
    "entries_per_page",
)
_LIST_TOOLS_ARGUMENTS = (
    "server_ids",
    "query",
    "search_fields",
    "search_regex",
    "search_case_sensitive",
    "include_description",
    "include_schema",
    "current_page",
    "entries_per_page",
)


def _mcp_runtime_error(error: BaseException) -> dict[str, Any]:
    """Convert nested MCP transport failures into a model-facing result."""
    pending = [error]
    seen: set[int] = set()
    exceptions: list[BaseException] = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        exceptions.append(current)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)

    statuses = {
        current.response.status_code
        for current in exceptions
        if isinstance(current, httpx.HTTPStatusError)
    }
    if 401 in statuses:
        code = "authentication_failed"
        message = "The MCP server rejected the supplied credentials"
    elif 403 in statuses:
        code = "access_denied"
        message = "The MCP server denied access"
    elif 404 in statuses:
        code = "server_not_found"
        message = "The MCP server endpoint was not found"
    elif any(500 <= status <= 599 for status in statuses):
        code = "server_error"
        message = "The MCP server failed to process the request"
    elif any(
        isinstance(current, (httpx.NetworkError, httpx.TimeoutException))
        for current in exceptions
    ):
        code = "cannot_connect"
        message = "Unable to connect to the MCP server"
    else:
        code = "mcp_request_failed"
        message = "The MCP request failed"
    return {"error": {"code": code, "message": message}}


def build_mcp_function_tool(server_ids: list[str]) -> dict[str, Any]:
    """Build the model-facing MCP dispatcher definition for an agent."""
    return {
        "spec": {
            "name": "mcp",
            "description": (
                "Use list_servers to discover servers, list_tools to search "
                "discoverable tools, get_tool_details to retrieve all available "
                "details for a specific authorised tool, or execute_tool to invoke "
                "it. Use discovery "
                "before execution when the required server, tool, or arguments are "
                "unknown."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "list_servers",
                            "list_tools",
                            "get_tool_details",
                            "execute_tool",
                        ],
                        "description": (
                            "The MCP operation to perform. get_tool_details and "
                            "execute_tool require server_id and tool_name."
                        ),
                    },
                    "server_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "For list_tools only, exactly filter discovery to these "
                            "server IDs. This cannot expand the agent's assigned "
                            "server scope; inaccessible IDs return no results."
                        ),
                    },
                    "server_id": {
                        "type": "string",
                        "description": (
                            "The single target server ID required by get_tool_details and "
                            "execute_tool. Unlike server_ids, this identifies the "
                            "target of a specific operation."
                        ),
                    },
                    "tool_name": {
                        "type": "string",
                        "description": (
                            "The tool name required by get_tool_details and "
                            "execute_tool. get_tool_details works for any authorised "
                            "tool, including a tool omitted from discovery results. It "
                            "returns the input schema and any available title, "
                            "description, output schema, MCP annotations (such as "
                            "read-only, destructive, idempotent, and open-world "
                            "hints), and task-support status. Details the server does "
                            "not provide are omitted."
                        ),
                    },
                    "arguments": {
                        "type": "object",
                        "description": (
                            "Tool arguments for execute_tool, following the tool's "
                            "input schema. May be omitted for a tool that accepts no "
                            "arguments."
                        ),
                        "additionalProperties": True,
                    },
                    "query": {
                        "type": "string",
                        "description": (
                            "Optional server or tool search query. When omitted, all "
                            "eligible discovery results match."
                        ),
                    },
                    "search_fields": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "server_id",
                                "server_name",
                                "tool_name",
                                "description",
                                "schema",
                            ],
                        },
                        "description": (
                            "Optional fields in which to search. list_servers accepts "
                            "server_id, server_name, tool_name, description, and "
                            "schema. list_tools accepts only tool_name, description, "
                            "and schema. Defaults to every field supported by the "
                            "selected action."
                        ),
                    },
                    "search_regex": {
                        "type": "boolean",
                        "description": (
                            "Interpret query as a regular expression. Defaults to "
                            "false."
                        ),
                    },
                    "search_case_sensitive": {
                        "type": "boolean",
                        "description": (
                            "Perform a case-sensitive search. Defaults to false."
                        ),
                    },
                    "include_tools": {
                        "type": "boolean",
                        "description": (
                            "For list_servers, include matching tool summaries as a "
                            "bounded preview. Defaults to false."
                        ),
                    },
                    "all_tools": {
                        "type": "boolean",
                        "description": (
                            "For list_servers, include all discoverable tools from "
                            "each matching server rather than only tools matching "
                            "the query, subject to the per-server result limit. "
                            "Implies include_tools=true. Use list_tools with the "
                            "returned server ID to paginate further. Defaults to "
                            "false."
                        ),
                    },
                    "include_description": {
                        "type": "boolean",
                        "description": (
                            "Include tool descriptions in results. Server descriptions "
                            "are always included by list_servers. Defaults to true."
                        ),
                    },
                    "include_schema": {
                        "type": "boolean",
                        "description": (
                            "Include tool input schemas and advertised output schemas "
                            "in results. For list_servers, this applies only to tool "
                            "summaries included in the bounded preview. For list_tools, "
                            "this applies to each returned tool. Defaults to false."
                        ),
                    },
                    "server_info": {
                        "type": "string",
                        "enum": ["cached", "latest"],
                        "description": (
                            "For list_servers only, include MCP initialization "
                            "information. cached uses the last successful setup or "
                            "reconfiguration check without contacting the server. "
                            "latest contacts each server on the returned page and "
                            "compares its current information with the cached value. "
                            "Omit this parameter to exclude server information."
                        ),
                    },
                    "current_page": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Result page to return. Defaults to 1.",
                    },
                    "entries_per_page": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "description": (
                            "Results per page, from 1 through the hard maximum of 100. "
                            "Defaults to 20. For list_servers, this also bounds each "
                            "server's included tool preview."
                        ),
                    },
                },
                "required": ["action"],
            },
        },
        "function": {"type": "mcp", "server_ids": list(server_ids)},
    }


class MCPFunction(Function):
    """Dispatch MCP actions over the servers assigned to an agent."""

    def __init__(self) -> None:
        """Initialize the function configuration schema."""
        super().__init__(vol.Schema({vol.Required("server_ids"): [str]}))

    async def execute(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute an MCP action against the configured server scope."""
        action = arguments.get("action")
        if action not in {
            "list_servers",
            "list_tools",
            "get_tool_details",
            "execute_tool",
        }:
            return {
                "error": {
                    "code": "invalid_action",
                    "message": "Unsupported MCP action",
                }
            }

        try:
            services = MCPServiceManager(hass).scope(
                function_config["server_ids"]
            ).list_loaded()

            if action == "list_servers":
                return await MCPToolManager.list_servers(
                    services,
                    **{
                        key: arguments[key]
                        for key in _LIST_SERVERS_ARGUMENTS
                        if key in arguments
                    },
                )

            if action == "list_tools":
                return await MCPToolManager.list_tools(
                    services,
                    **{
                        key: arguments[key]
                        for key in _LIST_TOOLS_ARGUMENTS
                        if key in arguments
                    },
                )

            if action == "get_tool_details":
                return await MCPToolManager.get_tool_details(
                    services,
                    server_id=arguments.get("server_id", ""),
                    tool_name=arguments.get("tool_name", ""),
                )

            return await MCPToolManager.execute_tool(
                services,
                server_id=arguments.get("server_id", ""),
                tool_name=arguments.get("tool_name", ""),
                arguments=arguments.get("arguments"),
            )
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.exception("MCP %s action failed", action)
            return _mcp_runtime_error(err)
