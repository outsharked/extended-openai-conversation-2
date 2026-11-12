"""Search MCP servers without host-framework dependencies."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any

import regex

from .mcp_datatypes import MCPTool, MCPToolResult
from .mcp_service import MCPService
from .service_config import MCPServiceConfig

_DEFAULT_ENTRIES_PER_PAGE = 20
_MAX_ENTRIES_PER_PAGE = 100
_SERVER_INFO_MODES = frozenset({"cached", "latest"})
_SERVER_SEARCH_FIELDS = frozenset(
    {"server_id", "server_name", "tool_name", "description", "schema"}
)
_TOOL_SEARCH_FIELDS = frozenset({"tool_name", "description", "schema"})
_SCHEMA_BOILERPLATE_KEYS = frozenset(
    {
        "$defs",
        "$schema",
        "additionalProperties",
        "allOf",
        "anyOf",
        "definitions",
        "items",
        "oneOf",
        "properties",
        "required",
        "type",
    }
)


class MCPToolManager:
    """Implement model-facing MCP operations over authorized service inputs."""

    @staticmethod
    async def list_servers(
        services: Collection[MCPService],
        *,
        query: str | None = None,
        search_fields: Collection[str] | None = None,
        search_regex: bool = False,
        search_case_sensitive: bool = False,
        include_tools: bool = False,
        all_tools: bool = False,
        include_description: bool = True,
        include_schema: bool = False,
        server_info: str | None = None,
        current_page: int = 1,
        entries_per_page: int = _DEFAULT_ENTRIES_PER_PAGE,
    ) -> dict[str, Any]:
        """Return authorized MCP servers matching the requested criteria."""

        validation_error: dict[str, Any] | None = MCPToolManager._validate_search_request(
            query=query,
            search_fields=search_fields,
            supported_search_fields=_SERVER_SEARCH_FIELDS,
            current_page=current_page,
            entries_per_page=entries_per_page,
        )
        if validation_error is not None:
            return validation_error
        if server_info is not None and (
            not isinstance(server_info, str) or server_info not in _SERVER_INFO_MODES
        ):
            return MCPToolManager._error(
                "invalid_server_info",
                "server_info must be cached or latest",
            )

        include_tools = include_tools or all_tools

        try:
            fields, pattern = MCPToolManager._prepare_search(
                search_fields=search_fields,
                query=query,
                search_regex=search_regex,
                search_case_sensitive=search_case_sensitive,
                default_fields=_SERVER_SEARCH_FIELDS,
            )
        except regex.error as err:
            return MCPToolManager._error("invalid_regex", str(err))

        matches: list[dict[str, Any]] = []
        try:
            for service in services:
                server_matches, tools, matching_tools = (
                    await MCPToolManager._match_server(
                        service,
                        query=query,
                        fields=fields,
                        pattern=pattern,
                        search_case_sensitive=search_case_sensitive,
                        include_tools=include_tools,
                    )
                )
                if not server_matches and not matching_tools:
                    continue
                matches.append(
                    MCPToolManager._build_server_result(
                        service.config,
                        query=query,
                        server_matches=server_matches,
                        tools=tools,
                        matching_tools=matching_tools,
                        include_tools=include_tools,
                        all_tools=all_tools,
                        include_description=include_description,
                        include_schema=include_schema,
                        tool_limit=entries_per_page,
                    )
                )
        except TimeoutError:
            return MCPToolManager._error("regex_timeout", "The MCP catalogue search timed out")

        page = MCPToolManager._paginate(
            matches,
            result_key="servers",
            current_page=current_page,
            entries_per_page=entries_per_page,
        )
        if server_info is not None:
            services_by_id = {
                service.config.server.server_id: service for service in services
            }
            await MCPToolManager._add_server_information(
                page["servers"], services_by_id, server_info
            )
        return page

    @staticmethod
    async def list_tools(
        services: Collection[MCPService],
        *,
        server_ids: Collection[str] | None = None,
        query: str | None = None,
        search_fields: Collection[str] | None = None,
        search_regex: bool = False,
        search_case_sensitive: bool = False,
        include_description: bool = True,
        include_schema: bool = False,
        current_page: int = 1,
        entries_per_page: int = _DEFAULT_ENTRIES_PER_PAGE,
    ) -> dict[str, Any]:
        """Return discoverable tools from authorized MCP services."""
        validation_error = MCPToolManager._validate_search_request(
            query=query,
            search_fields=search_fields,
            supported_search_fields=_TOOL_SEARCH_FIELDS,
            current_page=current_page,
            entries_per_page=entries_per_page,
        )
        if validation_error is not None:
            return validation_error
        if server_ids is not None and (
            isinstance(server_ids, str)
            or not all(isinstance(server_id, str) for server_id in server_ids)
        ):
            return MCPToolManager._error("invalid_server_ids", "server_ids must be a list of strings")

        try:
            fields, pattern = MCPToolManager._prepare_search(
                search_fields=search_fields,
                query=query,
                search_regex=search_regex,
                search_case_sensitive=search_case_sensitive,
                default_fields=_TOOL_SEARCH_FIELDS,
            )
        except regex.error as err:
            return MCPToolManager._error("invalid_regex", str(err))

        selected_server_ids = (
            frozenset(server_ids) if server_ids is not None else None
        )
        matches: list[dict[str, Any]] = []
        try:
            for service in services:
                if (
                    selected_server_ids is not None
                    and service.config.server.server_id not in selected_server_ids
                ):
                    continue
                for tool in await MCPToolManager._discoverable_tools(service):
                    if query is not None and not MCPToolManager._matches(
                        MCPToolManager._tool_search_values(tool, fields),
                        query,
                        pattern=pattern,
                        case_sensitive=search_case_sensitive,
                        timeout=service.config.regex_timeout,
                    ):
                        continue
                    result = MCPToolManager._tool_summary(
                        tool,
                        include_description=include_description,
                        include_schema=include_schema,
                    )
                    result["server_id"] = service.config.server.server_id
                    matches.append(result)
        except TimeoutError:
            return MCPToolManager._error(
                "regex_timeout", "The MCP catalogue search timed out"
            )

        return MCPToolManager._paginate(
            matches,
            result_key="tools",
            current_page=current_page,
            entries_per_page=entries_per_page,
        )

    @staticmethod
    async def get_tool_details(
        services: Collection[MCPService],
        *,
        server_id: str,
        tool_name: str,
    ) -> dict[str, Any]:
        """Return all available details for one authorized MCP tool."""
        if not isinstance(server_id, str) or not server_id:
            return MCPToolManager._error("invalid_server_id", "server_id must be a non-empty string")
        if not isinstance(tool_name, str) or not tool_name:
            return MCPToolManager._error("invalid_tool_name", "tool_name must be a non-empty string")

        service = MCPToolManager._service_by_id(services, server_id)
        if service is None:
            return MCPToolManager._tool_not_found()

        tool = await MCPToolManager._find_tool(service, tool_name)
        if tool is None:
            return MCPToolManager._tool_not_found()
        return MCPToolManager._tool_details_result(server_id, tool)

    @staticmethod
    async def execute_tool(
        services: Collection[MCPService],
        *,
        server_id: str,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute one authorized MCP tool without retrying side effects."""
        if not isinstance(server_id, str) or not server_id:
            return MCPToolManager._error(
                "invalid_server_id", "server_id must be a non-empty string"
            )
        if not isinstance(tool_name, str) or not tool_name:
            return MCPToolManager._error("invalid_tool_name", "tool_name must be a non-empty string")
        if arguments is not None and (
            not isinstance(arguments, Mapping)
            or not all(isinstance(key, str) for key in arguments)
        ):
            return MCPToolManager._error("invalid_arguments", "arguments must be an object with string keys")

        service: MCPService | None = MCPToolManager._service_by_id(services, server_id)
        if service is None:
            return MCPToolManager._tool_not_found()
        tool = await MCPToolManager._find_tool(service, tool_name)
        if tool is None:
            return MCPToolManager._tool_not_found()

        result = await service.client.call_tool(tool.name, arguments)
        return MCPToolManager._tool_result(server_id, tool.name, result)

    @staticmethod
    def _prepare_search(
        *,
        search_fields: Collection[str] | None,
        query: str | None,
        search_regex: bool,
        search_case_sensitive: bool,
        default_fields: Collection[str],
    ) -> tuple[frozenset[str], regex.Pattern[str] | None]:
        """Resolve default fields and compile an optional search expression."""
        fields = (
            frozenset(search_fields) if search_fields is not None else frozenset(default_fields)
        )
        if query is None or not search_regex:
            return fields, None
        flags = 0 if search_case_sensitive else regex.IGNORECASE
        return fields, regex.compile(query, flags)

    @staticmethod
    async def _match_server(
        service: MCPService,
        *,
        query: str | None,
        fields: Collection[str],
        pattern: regex.Pattern[str] | None,
        search_case_sensitive: bool,
        include_tools: bool,
    ) -> tuple[bool, tuple[MCPTool, ...], tuple[MCPTool, ...]]:
        """Evaluate server metadata and eligible tools against one search."""

        server_matches = query is None or MCPToolManager._matches(
            MCPToolManager._server_search_values(service.config, fields),
            query,
            pattern=pattern,
            case_sensitive=search_case_sensitive,
            timeout=service.config.regex_timeout,
        )
        requires_tools = include_tools or (
            query is not None and bool(set(fields) & _TOOL_SEARCH_FIELDS)
        )
        tools: tuple[MCPTool, ...] = (
            await MCPToolManager._discoverable_tools(service) if requires_tools else ()
        )
        matching_tools = (
            tools
            if query is None else tuple(
                tool
                for tool in tools
                if MCPToolManager._matches(
                    MCPToolManager._tool_search_values(tool, fields),
                    query,
                    pattern=pattern,
                    case_sensitive=search_case_sensitive,
                    timeout=service.config.regex_timeout,
                )
            )
        )
        return server_matches, tools, matching_tools

    @staticmethod
    def _build_server_result(
        config: MCPServiceConfig,
        *,
        query: str | None,
        server_matches: bool,
        tools: tuple[MCPTool, ...],
        matching_tools: tuple[MCPTool, ...],
        include_tools: bool,
        all_tools: bool,
        include_description: bool,
        include_schema: bool,
        tool_limit: int,
    ) -> dict[str, Any]:
        """Build one model-facing server result after matching is complete."""
        server = config.server
        result: dict[str, Any] = {
            "id": server.server_id,
            "name": server.name,
            "description": server.description,
        }
        if not include_tools:
            return result

        returned_tools = tools if query is None or all_tools else matching_tools
        result["tools"] = [
            MCPToolManager._tool_summary(
                tool,
                include_description=include_description,
                include_schema=include_schema,
            )
            for tool in returned_tools[:tool_limit]
        ]
        result["tool_matches"] = len(returned_tools)
        result["tools_returned"] = len(result["tools"])
        result["tools_truncated"] = len(returned_tools) > tool_limit
        if query is not None and server_matches and not matching_tools:
            result["tool_match_message"] = (
                "Server metadata matched, but no discoverable tools matched the "
                "criteria. Try searching without tool fields or list all tools."
            )
        return result

    @staticmethod
    async def _add_server_information(
        results: list[dict[str, Any]],
        services_by_id: Mapping[str, MCPService],
        mode: str,
    ) -> None:
        """Add cached or live initialization details to one result page."""
        for result in results:
            service = services_by_id[result["id"]]
            cached = service.config.server_information
            if mode == "cached":
                if cached is not None:
                    result["server_info"] = cached.to_dict()
                    result["server_info_source"] = "cached"
                continue

            try:
                latest = await service.client.check_connection()
            except Exception:  # A metadata failure must not hide the server.
                result["server_info_error"] = {
                    "code": "server_info_unavailable",
                    "message": "Latest server information could not be retrieved",
                }
                continue

            latest_data = latest.to_dict()
            result["server_info"] = latest_data
            result["server_info_source"] = "latest"
            if cached is None:
                result["server_info_comparison"] = "no_cached_information"
                continue

            cached_data = cached.to_dict()
            changed_fields = [
                field
                for field in latest_data.keys() | cached_data.keys()
                if latest_data.get(field) != cached_data.get(field)
            ]
            result["server_info_changed"] = bool(changed_fields)
            if changed_fields:
                result["changed_fields"] = sorted(changed_fields)

    @staticmethod
    def _paginate(
        items: Sequence[dict[str, Any]],
        *,
        result_key: str,
        current_page: int,
        entries_per_page: int,
    ) -> dict[str, Any]:
        """Return one result page with consistent pagination metadata."""
        max_matches = len(items)
        max_pages = (
            (max_matches + entries_per_page - 1) // entries_per_page
            if max_matches
            else 0
        )
        start = (current_page - 1) * entries_per_page
        return {
            result_key: list(items[start : start + entries_per_page]),
            "max_matches": max_matches,
            "current_page": current_page,
            "max_pages": max_pages,
            "entries_per_page": entries_per_page,
        }

    @staticmethod
    async def _discoverable_tools(service: MCPService) -> tuple[MCPTool, ...]:
        """Return live, policy-filtered tools that are not exposed permanently."""
        if not service.config.on_demand_discovery:
            return ()
        return service.config.filter_on_demand_tools(await service.client.list_tools())

    @staticmethod
    async def _find_tool(service: MCPService, tool_name: str) -> MCPTool | None:
        """Find an exact permanent or discoverable tool allowed by policy."""
        permanent_tool = next(
            (
                tool
                for tool in service.config.permanent_tools
                if tool.name == tool_name
            ),
            None,
        )
        if permanent_tool is not None:
            return permanent_tool
        return next(
            (
                tool
                for tool in await MCPToolManager._discoverable_tools(service)
                if tool.name == tool_name
            ),
            None,
        )

    @staticmethod
    def _service_by_id(
        services: Collection[MCPService], server_id: str
    ) -> MCPService | None:
        """Return an authorized service by stable ID without leaking others."""
        return next(
            (
                service
                for service in services
                if service.config.server.server_id == server_id
            ),
            None,
        )

    @staticmethod
    def _tool_details_result(server_id: str, tool: MCPTool) -> dict[str, Any]:
        """Return all known model-facing details for one tool."""
        result: dict[str, Any] = {
            "server_id": server_id,
            "tool_name": tool.name,
            "input_schema": tool.input_schema,
        }
        if tool.title is not None:
            result["title"] = tool.title
        if tool.description is not None:
            result["description"] = tool.description
        if tool.output_schema is not None:
            result["output_schema"] = tool.output_schema
        if tool.annotations:
            result["annotations"] = tool.annotations
        if tool.task_support is not None:
            result["execution"] = {"task_support": tool.task_support}
        return result

    @staticmethod
    def _tool_result(
        server_id: str, tool_name: str, result: MCPToolResult
    ) -> dict[str, Any]:
        """Return the serializable result of one tool invocation."""
        return {
            "server_id": server_id,
            "tool_name": tool_name,
            "content": list(result.content),
            "structured_content": result.structured_content,
            "is_error": result.is_error,
        }

    @staticmethod
    def _tool_not_found() -> dict[str, Any]:
        """Return a generic result that does not reveal authorization state."""
        return MCPToolManager._error("tool_not_found", "Tool was not found")

    @staticmethod
    def _server_search_values(config: MCPServiceConfig, fields: Collection[str]) -> tuple[str, ...]:
        """Return selected searchable values from server metadata."""
        server = config.server
        values: list[str] = []
        if "server_id" in fields:
            values.append(server.server_id)
        if "server_name" in fields:
            values.append(server.name)
        if "description" in fields:
            values.append(server.description)
        return tuple(values)

    @staticmethod
    def _tool_search_values(tool: MCPTool, fields: Collection[str]) -> tuple[str, ...]:
        """Return selected searchable values from one tool definition."""
        values: list[str] = []
        if "tool_name" in fields:
            values.append(tool.name)
        if "description" in fields and tool.description is not None:
            values.append(tool.description)
        if "schema" in fields:
            values.extend(MCPToolManager._schema_search_values(tool.input_schema))
        return tuple(values)

    @staticmethod
    def _schema_search_values(value: Any) -> tuple[str, ...]:
        """Extract meaningful searchable values from a JSON schema."""
        values: list[str] = []
        if isinstance(value, dict):
            for key, child in value.items():
                if key not in _SCHEMA_BOILERPLATE_KEYS:
                    values.append(str(key))
                if key != "type":
                    values.extend(MCPToolManager._schema_search_values(child))
        elif isinstance(value, list):
            for child in value:
                values.extend(MCPToolManager._schema_search_values(child))
        elif isinstance(value, (str, int, float, bool)):
            values.append(str(value))
        return tuple(values)

    @staticmethod
    def _matches(
        values: Iterable[str],
        query: str,
        *,
        pattern: regex.Pattern[str] | None,
        case_sensitive: bool,
        timeout: float,
    ) -> bool:
        """Return whether any independently searched field matches the query."""
        if pattern is not None:
            return any(pattern.search(value, timeout=timeout) is not None for value in values)
        if not case_sensitive:
            query = query.casefold()
            return any(query in value.casefold() for value in values)
        return any(query in value for value in values)

    @staticmethod
    def _tool_summary(
        tool: MCPTool,
        *,
        include_description: bool,
        include_schema: bool,
    ) -> dict[str, Any]:
        """Return the compact tool representation used by list_servers."""
        result: dict[str, Any] = {"tool_name": tool.name}
        if include_description:
            result["description"] = tool.description
        if include_schema:
            result["input_schema"] = tool.input_schema
            if tool.output_schema is not None:
                result["output_schema"] = tool.output_schema
        return result

    @staticmethod
    def _validate_search_request(
        *,
        query: str | None,
        search_fields: Collection[str] | None,
        supported_search_fields: Collection[str],
        current_page: int,
        entries_per_page: int,
    ) -> dict[str, Any] | None:
        """Validate shared search and pagination arguments."""
        if query is not None and not isinstance(query, str):
            return MCPToolManager._error("invalid_query", "query must be a string")
        if search_fields is not None:
            if (
                isinstance(search_fields, str)
                or not search_fields
                or not all(isinstance(field, str) for field in search_fields)
            ):
                return MCPToolManager._error("invalid_search_fields", "search_fields must be a non-empty list")
            invalid_fields = set(search_fields) - set(supported_search_fields)
            if invalid_fields:
                return MCPToolManager._error(
                    "invalid_search_fields",
                    f"Unsupported search fields: {', '.join(sorted(invalid_fields))}",
                )
        if (
            isinstance(current_page, bool)
            or not isinstance(current_page, int)
            or current_page < 1
        ):
            return MCPToolManager._error("invalid_current_page", "current_page must be at least 1" )
        if (
            isinstance(entries_per_page, bool)
            or not isinstance(entries_per_page, int)
            or entries_per_page < 1
            or entries_per_page > _MAX_ENTRIES_PER_PAGE
        ):
            return MCPToolManager._error(
                "invalid_entries_per_page",
                f"entries_per_page must be between 1 and {_MAX_ENTRIES_PER_PAGE}",
            )
        return None

    @staticmethod
    def _error(code: str, message: str) -> dict[str, Any]:
        """Return a serializable model-facing error."""
        return {"error": {"code": code, "message": message}}
