"""Datatypes used by the MCP Client API"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class MCPServerInformation:
    """Public server information returned by MCP initialization."""

    name: str
    version: str
    protocol_version: str
    instructions: str | None
    capabilities: dict[str, Any]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MCPServerInformation:
        """Build server information from its public serializable mapping."""
        return cls(
            name=data["name"],
            version=data["version"],
            protocol_version=data["protocol_version"],
            instructions=data.get("instructions"),
            capabilities=dict(data.get("capabilities", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the public serializable mapping."""
        return {
            "name": self.name,
            "version": self.version,
            "protocol_version": self.protocol_version,
            "instructions": self.instructions,
            "capabilities": self.capabilities,
        }


@dataclass(frozen=True, slots=True)
class MCPTool:
    """Serializable MCP tool definition returned by catalogue discovery."""

    name: str
    title: str | None
    description: str | None
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None
    annotations: dict[str, Any] | None = None
    task_support: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MCPTool:
        """Build a tool snapshot from its public serializable mapping."""
        output_schema = data.get("output_schema")
        annotations = data.get("annotations")
        task_support = data.get("task_support")
        legacy_definition = data.get("definition")
        if isinstance(legacy_definition, Mapping):
            if annotations is None:
                annotations = legacy_definition.get("annotations")
            if task_support is None:
                execution = legacy_definition.get("execution")
                if isinstance(execution, Mapping):
                    task_support = execution.get("taskSupport")
        return cls(
            name=data["name"],
            title=data.get("title"),
            description=data.get("description"),
            input_schema=dict(data["input_schema"]),
            output_schema=(
                dict(output_schema) if isinstance(output_schema, Mapping) else None
            ),
            annotations=(
                dict(annotations) if isinstance(annotations, Mapping) else None
            ),
            task_support=task_support if isinstance(task_support, str) else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete public serializable tool snapshot."""
        result: dict[str, Any] = {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
        }
        if self.annotations:
            result["annotations"] = self.annotations
        if self.task_support is not None:
            result["task_support"] = self.task_support
        return result


@dataclass(frozen=True, slots=True)
class MCPToolResult:
    """Serializable result returned by an MCP tool invocation."""

    content: tuple[dict[str, Any], ...]
    structured_content: dict[str, Any] | None
    is_error: bool


@dataclass(frozen=True, slots=True)
class MCPDiscoveryResult:
    """Server metadata and complete tool catalogue from one MCP session."""

    server: MCPServerInformation
    tools: tuple[MCPTool, ...]
