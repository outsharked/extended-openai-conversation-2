"""Render and estimate the permanent MCP context supplied to a model."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math

from .service_config import MCPServiceConfig

DEFAULT_MCP_PROMPT = (
    "Use the MCP dispatcher with list_servers to find available servers, "
    "list_tools to discover tools, get_tool_details to retrieve all available "
    "details for a tool, and execute_tool to run it. Use the server_id and tool name "
    "provided in this context or returned by discovery. Tools already described in this "
    "context do not need to be discovered again."
)

@dataclass(frozen=True, slots=True)
class MCPRenderedContext:
    """Model-facing context rendered from one MCP service configuration."""

    content: str
    selected_tool_count: int


@dataclass(frozen=True, slots=True)
class MCPContextEstimate:
    """Display-only size information for one rendered MCP context block."""

    selected_tool_count: int
    character_count: int
    byte_count: int
    estimated_tokens: int

    @classmethod
    def from_context(cls, context: MCPRenderedContext) -> MCPContextEstimate:
        """Estimate the size of already-rendered model context."""
        byte_count = len(context.content.encode("utf-8"))
        return cls(
            selected_tool_count=context.selected_tool_count,
            character_count=len(context.content),
            byte_count=byte_count,
            estimated_tokens=math.ceil(byte_count / 4),
        )

class MCPContextRenderer:
    @staticmethod
    def render_permanent_context(config: MCPServiceConfig) -> MCPRenderedContext:
        """Render stable model context without calculating display estimates."""

        tools = config.permanent_tools
        if not tools:
            return MCPRenderedContext(content="", selected_tool_count=0)

        rendered_tools: list[dict[str, object]] = []
        for tool in sorted(tools, key=lambda item: item.name):
            rendered_tool: dict[str, object] = {
                "name": tool.name,
                "description": tool.description,
            }
            if config.include_permanent_schemas:
                rendered_tool["input_schema"] = tool.input_schema
                if tool.output_schema is not None:
                    rendered_tool["output_schema"] = tool.output_schema
            rendered_tools.append(rendered_tool)

        context: dict[str, object] = {
            "server_id": config.server.server_id,
            "server_name": config.server.name,
            "server_description": config.server.description,
            "tools": rendered_tools,
        }
        return MCPRenderedContext(
            content=json.dumps(
                context,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            selected_tool_count=len(tools),
        )
