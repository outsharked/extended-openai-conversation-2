"""Serializable configuration and tool-access policy for one MCP service."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace
from typing import Any

import regex

from .config import MCPServerConfig
from .mcp_const import (
    CONF_MCP_DISABLED_TOOL_MODE,
    CONF_MCP_DISABLED_TOOLS,
    CONF_MCP_INCLUDE_PERMANENT_SCHEMAS,
    CONF_MCP_ON_DEMAND_DISCOVERY,
    CONF_MCP_PERFORM_TOOL_DISCOVERY,
    CONF_MCP_PERMANENT_TOOL_CATALOGUE,
    CONF_MCP_PERMANENT_TOOL_MODE,
    CONF_MCP_PERMANENT_TOOLS,
    CONF_MCP_REGEX_TIMEOUT,
    CONF_MCP_SERVER_INFORMATION,
    CONF_MCP_TOOL_NAME_PATTERN,
    DEFAULT_MCP_DISABLED_TOOL_MODE,
    DEFAULT_MCP_INCLUDE_PERMANENT_SCHEMAS,
    DEFAULT_MCP_ON_DEMAND_DISCOVERY,
    DEFAULT_MCP_PERFORM_TOOL_DISCOVERY,
    DEFAULT_MCP_PERMANENT_TOOL_MODE,
    DEFAULT_MCP_REGEX_TIMEOUT,
    DEFAULT_MCP_TOOL_NAME_PATTERN,
    MCP_TOOL_FILTER_SELECTED,
    MCP_TOOL_FILTER_UNSELECTED,
)
from .mcp_datatypes import MCPDiscoveryResult, MCPServerInformation, MCPTool
from .mcp_errors import MCPConfigError

_FILTER_MODES = {MCP_TOOL_FILTER_SELECTED, MCP_TOOL_FILTER_UNSELECTED}


@dataclass(frozen=True, slots=True)
class MCPServiceConfig:
    """Connection, discovery snapshot, and model-visible tool policies."""

    server: MCPServerConfig
    perform_tool_discovery: bool = DEFAULT_MCP_PERFORM_TOOL_DISCOVERY
    permanent_tool_mode: str = DEFAULT_MCP_PERMANENT_TOOL_MODE
    permanent_tool_selection: tuple[str, ...] = ()
    permanent_tools: tuple[MCPTool, ...] = ()
    include_permanent_schemas: bool = DEFAULT_MCP_INCLUDE_PERMANENT_SCHEMAS
    on_demand_discovery: bool = DEFAULT_MCP_ON_DEMAND_DISCOVERY
    disabled_tool_mode: str = DEFAULT_MCP_DISABLED_TOOL_MODE
    disabled_tools: tuple[str, ...] = ()
    tool_name_pattern: str = DEFAULT_MCP_TOOL_NAME_PATTERN
    regex_timeout: float = DEFAULT_MCP_REGEX_TIMEOUT
    server_information: MCPServerInformation | None = None

    def __post_init__(self) -> None:
        """Validate policy values that are independent of a host framework."""
        if self.permanent_tool_mode not in _FILTER_MODES:
            raise MCPConfigError("permanent_tool_mode", "invalid_filter_mode")
        if self.disabled_tool_mode not in _FILTER_MODES:
            raise MCPConfigError("disabled_tool_mode", "invalid_filter_mode")
        if (
            isinstance(self.regex_timeout, bool)
            or not isinstance(self.regex_timeout, (int, float))
            or self.regex_timeout <= 0
        ):
            raise MCPConfigError("regex_timeout", "invalid_regex_timeout")
        try:
            regex.compile(self.tool_name_pattern)
        except regex.error as err:
            raise MCPConfigError(
                "tool_name_pattern", "invalid_tool_name_pattern"
            ) from err

        tool_names = [tool.name for tool in self.permanent_tools]
        if len(tool_names) != len(set(tool_names)):
            raise MCPConfigError("permanent_tools", "invalid_permanent_tools")

    @classmethod
    def from_selection(
        cls,
        server: MCPServerConfig,
        discovery: MCPDiscoveryResult | None,
        *,
        perform_tool_discovery: bool,
        permanent_tool_mode: str,
        selected_tool_names: Collection[str],
        include_permanent_schemas: bool,
        on_demand_discovery: bool,
        disabled_tool_mode: str,
        disabled_tool_names: Collection[str],
        tool_name_pattern: str,
        regex_timeout: float,
        server_information: MCPServerInformation | None = None,
    ) -> MCPServiceConfig:
        """Validate, apply hard filters, and normalize a policy selection."""
        permanent_names: set[str] = cls._validate_names(selected_tool_names, "permanent_tools")
        disabled_names: set[str] = cls._validate_names(disabled_tool_names, "disabled_tools")
        available_tools: tuple[MCPTool, ...] = discovery.tools if discovery is not None else ()
        available_names: set[str] = {tool.name for tool in available_tools}
        if not on_demand_discovery:
            disabled_tool_mode = DEFAULT_MCP_DISABLED_TOOL_MODE
            disabled_names.clear()
            tool_name_pattern = DEFAULT_MCP_TOOL_NAME_PATTERN
        if discovery is None:
            if permanent_names or disabled_names:
                raise MCPConfigError("permanent_tools", "mcp_discovery_required")
            include_permanent_schemas = False
        elif (permanent_names | disabled_names) - available_names:
            raise MCPConfigError("permanent_tools", "invalid_permanent_tools")

        config: MCPServiceConfig = cls(
            server=server,
            perform_tool_discovery=perform_tool_discovery,
            permanent_tool_mode=permanent_tool_mode,
            permanent_tool_selection=tuple(sorted(permanent_names)),
            include_permanent_schemas=include_permanent_schemas,
            on_demand_discovery=on_demand_discovery,
            disabled_tool_mode=disabled_tool_mode,
            disabled_tools=tuple(sorted(disabled_names)),
            tool_name_pattern=tool_name_pattern,
            regex_timeout=regex_timeout,
            server_information=server_information,
        )
        permanent_tools: tuple[MCPTool, ...] = tuple(
            tool
            for tool in available_tools
            if config._selection_matches(
                tool.name, config.permanent_tool_selection, config.permanent_tool_mode
            )
        )
        disabled_tool_names = (
           {
                tool.name
                for tool in available_tools
                if config._selection_matches(
                    tool.name, config.disabled_tools, config.disabled_tool_mode
                )
            }
            if config.disabled_tools else set()
        )
        if disabled_tool_names & {tool.name for tool in permanent_tools}:
            raise MCPConfigError("disabled_tools", "permanent_disabled_tool_conflict")
        return replace(config, permanent_tools=permanent_tools)

    @staticmethod
    def _validate_names(names: Collection[str], field: str) -> set[str]:
        if isinstance(names, str) or not all(isinstance(name, str) for name in names):
            raise MCPConfigError(field, f"invalid_{field}")
        return set(names)

    @staticmethod
    def _selection_matches(name: str, selected: Collection[str], mode: str) -> bool:
        is_selected = name in selected
        return is_selected if mode == MCP_TOOL_FILTER_SELECTED else not is_selected

    def tool_allowed(self, tool_name: str) -> bool:
        """Return whether a canonical tool name passes on-demand filters."""
        # if no explicitly disabled tools then allow falling through to the regex
        disabled: bool = bool(self.disabled_tools) and self._selection_matches(
            tool_name, self.disabled_tools, self.disabled_tool_mode
        )
        if disabled:
            return False
        if not self.tool_name_pattern:
            return True
        try:
            return (
                # this disabled filter is to say any tools that match should be disabled,
                # so if it returns None then that means the tool is allowed
                regex.fullmatch(self.tool_name_pattern, tool_name, timeout=self.regex_timeout)
                is None
            )
        except TimeoutError:
            return False

    def filter_allowed_tools(self, tools: Collection[MCPTool]) -> tuple[MCPTool, ...]:
        """Apply exact and regular-expression policy before agent discovery."""
        return tuple(tool for tool in tools if self.tool_allowed(tool.name))

    def filter_on_demand_tools(self, tools: Collection[MCPTool]) -> tuple[MCPTool, ...]:
        """Apply hard filters and omit tools already supplied permanently."""
        permanent_names = {tool.name for tool in self.permanent_tools}
        return tuple(
            tool
            for tool in self.filter_allowed_tools(tools)
            if tool.name not in permanent_names
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MCPServiceConfig:
        """Build a service configuration from its persisted mapping."""
        server_information = data.get(CONF_MCP_SERVER_INFORMATION)
        catalogue = data.get(CONF_MCP_PERMANENT_TOOL_CATALOGUE, ())
        permanent_tools: tuple[MCPTool, ...] = tuple(MCPTool.from_dict(tool) for tool in catalogue)
        return cls(
            server=MCPServerConfig.from_dict(data),
            perform_tool_discovery=data.get(
                CONF_MCP_PERFORM_TOOL_DISCOVERY, DEFAULT_MCP_PERFORM_TOOL_DISCOVERY
            ),
            permanent_tool_mode=data.get(
                CONF_MCP_PERMANENT_TOOL_MODE, DEFAULT_MCP_PERMANENT_TOOL_MODE
            ),
            permanent_tool_selection=tuple(data.get(CONF_MCP_PERMANENT_TOOLS, ())),
            permanent_tools=permanent_tools,
            include_permanent_schemas=data.get(
                CONF_MCP_INCLUDE_PERMANENT_SCHEMAS,
                DEFAULT_MCP_INCLUDE_PERMANENT_SCHEMAS,
            ),
            on_demand_discovery=data.get(
                CONF_MCP_ON_DEMAND_DISCOVERY, DEFAULT_MCP_ON_DEMAND_DISCOVERY
            ),
            disabled_tool_mode=data.get(
                CONF_MCP_DISABLED_TOOL_MODE, DEFAULT_MCP_DISABLED_TOOL_MODE
            ),
            disabled_tools=tuple(data.get(CONF_MCP_DISABLED_TOOLS, ())),
            tool_name_pattern=data.get(
                CONF_MCP_TOOL_NAME_PATTERN, DEFAULT_MCP_TOOL_NAME_PATTERN
            ),
            regex_timeout=data.get(CONF_MCP_REGEX_TIMEOUT, DEFAULT_MCP_REGEX_TIMEOUT),
            server_information=(
                MCPServerInformation.from_dict(server_information)
                if isinstance(server_information, Mapping)
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a flat mapping suitable for host configuration storage."""
        data = {
            **self.server.to_dict(),
            **self.to_policy_dict(),
            CONF_MCP_PERMANENT_TOOL_CATALOGUE: [
                tool.to_dict() for tool in self.permanent_tools
            ],
        }
        if self.server_information is not None:
            data[CONF_MCP_SERVER_INFORMATION] = self.server_information.to_dict()
        return data

    def to_policy_dict(self) -> dict[str, Any]:
        """Return normalized tool-discovery policy values."""
        return {
            CONF_MCP_PERFORM_TOOL_DISCOVERY: self.perform_tool_discovery,
            CONF_MCP_PERMANENT_TOOL_MODE: self.permanent_tool_mode,
            CONF_MCP_PERMANENT_TOOLS: list(self.permanent_tool_selection),
            CONF_MCP_INCLUDE_PERMANENT_SCHEMAS: self.include_permanent_schemas,
            CONF_MCP_ON_DEMAND_DISCOVERY: self.on_demand_discovery,
            CONF_MCP_DISABLED_TOOL_MODE: self.disabled_tool_mode,
            CONF_MCP_DISABLED_TOOLS: list(self.disabled_tools),
            CONF_MCP_TOOL_NAME_PATTERN: self.tool_name_pattern,
            CONF_MCP_REGEX_TIMEOUT: self.regex_timeout,
        }
