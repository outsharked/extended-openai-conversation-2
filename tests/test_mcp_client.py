"""Tests for the extractable MCP client public API."""

from typing import Any
from unittest.mock import patch

import pytest

from custom_components.extended_openai_conversation.mcp_client import (
    MCPClientConfigFlow,
    MCPConfigError,
    MCPContextEstimate,
    MCPContextRenderer,
    MCPDiscoveryResult,
    MCPServerConfig,
    MCPServerInformation,
    MCPServiceConfig,
    MCPTool,
)


def _server(server_id: str = "vault", **overrides: Any) -> MCPServerConfig:
    """Return a valid MCP server configuration."""
    values = {
        "server_id": server_id,
        "name": "Vault MCP" if server_id == "vault" else "Tool Server",
        "url": "https://mcp.example.com/api",
        **overrides,
    }
    return MCPServerConfig(**values)


def _tool(name: str, **overrides: Any) -> MCPTool:
    """Return a minimal MCP tool definition."""
    values = {
        "name": name,
        "title": None,
        "description": None,
        "input_schema": {"type": "object"},
        "output_schema": None,
        **overrides,
    }
    return MCPTool(**values)


def _discovery(*tools: MCPTool) -> MCPDiscoveryResult:
    """Return common server information with the supplied catalogue."""
    return MCPDiscoveryResult(
        server=MCPServerInformation(
            name="Vault Server",
            version="1.0.0",
            protocol_version="2025-06-18",
            instructions=None,
            capabilities={},
        ),
        tools=tools,
    )


def test_server_config_normalises_and_serialises_values():
    """The public configuration model owns normalisation and serialisation."""
    config = MCPServerConfig(
        server_id=" vault ",
        name=" Vault MCP ",
        url=" https://mcp.example.com/api ",
        description=" Storage tools ",
        headers={"Authorization": "Bearer token"},
    )

    assert config.to_dict() == {
        "server_id": "vault",
        "name": "Vault MCP",
        "url": "https://mcp.example.com/api",
        "description": "Storage tools",
        "headers": {"Authorization": "Bearer token"},
        "use_oauth": False,
    }
    assert MCPServerConfig.from_dict(config.to_dict()) == config


def test_server_config_round_trips_oauth_selection():
    """OAuth is an explicit, serializable server connection setting."""
    config = MCPServerConfig(
        server_id="vault",
        name="Vault MCP",
        url="https://mcp.example.com/api",
        use_oauth=True,
    )

    assert config.to_dict()["use_oauth"] is True
    assert MCPServerConfig.from_dict(config.to_dict()) == config


def test_service_config_round_trips_complete_permanent_catalogue():
    """Persisted service data retains policies and complete tool snapshots."""
    tool = MCPTool(
        name="read_secret",
        title="Read secret",
        description="Read a stored secret.",
        input_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
        },
        output_schema={"type": "object"},
        annotations={"readOnlyHint": True},
        task_support="optional",
    )
    config = MCPServiceConfig(
        server=MCPServerConfig(
            server_id="vault",
            name="Vault MCP",
            url="https://mcp.example.com/api",
            description="Storage tools",
            headers={"Authorization": "Bearer token"},
        ),
        permanent_tool_selection=("read_secret",),
        permanent_tools=(tool,),
        include_permanent_schemas=False,
        on_demand_discovery=True,
        regex_timeout=0.2,
        server_information=MCPServerInformation(
            name="Vault Server",
            version="1.0.0",
            protocol_version="2025-06-18",
            instructions="Server instructions",
            capabilities={"tools": {"listChanged": True}},
        ),
    )

    persisted = config.to_dict()

    assert persisted["permanent_tools"] == ["read_secret"]
    assert persisted["permanent_tool_catalogue"] == [tool.to_dict()]
    assert persisted["permanent_tool_catalogue"][0]["input_schema"] == (
        tool.input_schema
    )
    assert "provide_default_prompt" not in persisted
    assert "definition" not in persisted["permanent_tool_catalogue"][0]
    assert MCPServiceConfig.from_dict(persisted) == config

    legacy_tool = {
        key: value
        for key, value in tool.to_dict().items()
        if key not in {"annotations", "task_support"}
    }
    legacy_tool["definition"] = {
        "name": "read_secret",
        "annotations": {"readOnlyHint": True},
        "execution": {"taskSupport": "optional"},
    }
    assert MCPTool.from_dict(legacy_tool) == tool


def test_service_config_selection_keeps_only_selected_tool_snapshots():
    """Selection validation stores complete definitions for selected tools."""
    server = _server()
    tools = tuple(_tool(name) for name in ("read_secret", "write_secret"))
    discovery = _discovery(*tools)

    config = MCPServiceConfig.from_selection(
        server,
        discovery,
        perform_tool_discovery=True,
        permanent_tool_mode="selected",
        selected_tool_names=["write_secret"],
        include_permanent_schemas=True,
        on_demand_discovery=True,
        disabled_tool_mode="selected",
        disabled_tool_names=[],
        tool_name_pattern="",
        regex_timeout=0.1,
        server_information=discovery.server,
    )

    assert config.permanent_tools == (tools[1],)
    assert config.server_information is discovery.server


def test_service_config_applies_filters_only_to_on_demand_tools():
    """Permanent tools bypass filters that constrain on-demand discovery."""
    server = _server()
    tools = tuple(
        _tool(name) for name in ("read_secret", "read_public", "write_secret")
    )
    discovery = _discovery(*tools)

    config = MCPServiceConfig.from_selection(
        server,
        discovery,
        perform_tool_discovery=True,
        permanent_tool_mode="selected",
        selected_tool_names=["write_secret"],
        include_permanent_schemas=False,
        on_demand_discovery=True,
        disabled_tool_mode="selected",
        disabled_tool_names=[],
        tool_name_pattern=r"write_.*",
        regex_timeout=0.1,
    )

    assert [tool.name for tool in config.permanent_tools] == ["write_secret"]
    assert [tool.name for tool in config.filter_on_demand_tools(tools)] == [
        "read_secret",
        "read_public",
    ]
    assert not config.tool_allowed("write_secret")
    assert config.tool_allowed("read_secret_extra_suffix")


@pytest.mark.parametrize("disabled_mode", ["selected", "unselected"])
def test_empty_disabled_selection_does_not_affect_regex_filter(
    disabled_mode: str,
):
    """An empty exact selection is inactive regardless of its mode."""
    config = MCPServiceConfig(
        server=_server("tools"),
        disabled_tool_mode=disabled_mode,
        disabled_tools=(),
        tool_name_pattern=r".*delete.*",
    )

    assert config.tool_allowed("read_record")
    assert not config.tool_allowed("delete_record")


def test_explicit_disabled_tool_short_circuits_regex_filter():
    """A tool disabled by exact selection is rejected without regex evaluation."""
    config = MCPServiceConfig(
        server=_server("tools"),
        disabled_tool_mode="selected",
        disabled_tools=("read_record",),
        tool_name_pattern=r".*delete.*",
    )

    with patch(
        "custom_components.extended_openai_conversation.mcp_client."
        "service_config.regex.fullmatch"
    ) as fullmatch:
        assert not config.tool_allowed("read_record")

    fullmatch.assert_not_called()


def test_disabled_selection_and_regex_are_independent_deny_filters():
    """Either an exact disabled selection or a regex match rejects a tool."""
    config = MCPServiceConfig(
        server=_server("tools"),
        disabled_tool_mode="selected",
        disabled_tools=("read_record",),
        tool_name_pattern=r".*delete.*",
    )

    assert not config.tool_allowed("read_record")
    assert not config.tool_allowed("delete_record")
    assert config.tool_allowed("update_record")


def test_service_config_rejects_permanent_disabled_tool_conflict():
    """Resolved permanent and disabled selections must not overlap."""
    server = _server()
    tool = _tool("write_secret")
    discovery = _discovery(tool)

    with pytest.raises(MCPConfigError) as err:
        MCPServiceConfig.from_selection(
            server,
            discovery,
            perform_tool_discovery=True,
            permanent_tool_mode="selected",
            selected_tool_names=["write_secret"],
            include_permanent_schemas=False,
            on_demand_discovery=True,
            disabled_tool_mode="selected",
            disabled_tool_names=["write_secret"],
            tool_name_pattern="",
            regex_timeout=0.1,
        )

    assert err.value.field == "disabled_tools"
    assert err.value.code == "permanent_disabled_tool_conflict"


def test_empty_disabled_selection_does_not_conflict_with_permanent_tools():
    """An empty inverted selection resolves to no disabled tools."""
    server = _server()
    tool = _tool("write_secret")
    discovery = _discovery(tool)

    config = MCPServiceConfig.from_selection(
        server,
        discovery,
        perform_tool_discovery=True,
        permanent_tool_mode="selected",
        selected_tool_names=["write_secret"],
        include_permanent_schemas=False,
        on_demand_discovery=True,
        disabled_tool_mode="unselected",
        disabled_tool_names=[],
        tool_name_pattern="",
        regex_timeout=0.1,
    )

    assert config.permanent_tools == (tool,)


def test_service_config_selection_rejects_unknown_tool():
    """A submitted tool name must exist in the discovered catalogue."""
    server = _server()
    discovery = _discovery()

    with pytest.raises(MCPConfigError) as err:
        MCPServiceConfig.from_selection(
            server,
            discovery,
            perform_tool_discovery=True,
            permanent_tool_mode="selected",
            selected_tool_names=["not_advertised"],
            include_permanent_schemas=True,
            on_demand_discovery=True,
            disabled_tool_mode="selected",
            disabled_tool_names=[],
            tool_name_pattern="",
            regex_timeout=0.1,
        )

    assert err.value.field == "permanent_tools"
    assert err.value.code == "invalid_permanent_tools"


def test_service_config_without_catalogue_has_no_permanent_tools():
    """Skipped discovery supports on-demand regex policy without selectors."""
    config = MCPServiceConfig.from_selection(
        _server(),
        None,
        perform_tool_discovery=False,
        permanent_tool_mode="selected",
        selected_tool_names=[],
        include_permanent_schemas=True,
        on_demand_discovery=True,
        disabled_tool_mode="selected",
        disabled_tool_names=[],
        tool_name_pattern="read_.*",
        regex_timeout=0.1,
    )

    assert config.permanent_tools == ()
    assert config.include_permanent_schemas is False
    assert config.server_information is None


@pytest.mark.parametrize(
    ("kwargs", "field", "code"),
    [
        ({"server_id": "Not Valid"}, "server_id", "invalid_server_id"),
        ({"name": " "}, "name", "invalid_name"),
        ({"url": "ftp://mcp.example.com"}, "url", "invalid_url"),
        ({"headers": {"Bad Header": "value"}}, "headers", "invalid_headers"),
        (
            {"headers": {"Authorization": "Bearer token\nInjected: value"}},
            "headers",
            "invalid_headers",
        ),
    ],
)
def test_server_config_rejects_invalid_values(kwargs, field, code):
    """Invalid host input produces stable public error fields and codes."""
    values = {
        "server_id": "vault",
        "name": "Vault MCP",
        "url": "https://mcp.example.com/api",
        "headers": {},
        "use_oauth": False,
        **kwargs,
    }

    with pytest.raises(MCPConfigError) as err:
        MCPServerConfig(**values)

    assert err.value.field == field
    assert err.value.code == code


def test_permanent_context_estimate_uses_the_rendered_schema_content():
    """Schema inclusion changes the exact context and its estimated token cost."""
    config = _server(description="Storage tools")
    tools = (
        MCPTool(
            name="read_secret",
            title="Read secret",
            description="Read a stored secret.",
            input_schema={
                "type": "object",
                "properties": {"key": {"type": "string"}},
            },
            output_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
        ),
    )

    metadata_context = MCPContextRenderer.render_permanent_context(
        MCPServiceConfig(
            server=config,
            permanent_tools=tools,
            include_permanent_schemas=False,
            disabled_tools=("read_secret",),
            tool_name_pattern="write_.*",
        )
    )
    schema_context = MCPContextRenderer.render_permanent_context(
        MCPServiceConfig(
            server=config,
            permanent_tools=tools,
            include_permanent_schemas=True,
            disabled_tools=("read_secret",),
            tool_name_pattern="write_.*",
        )
    )
    metadata_only = MCPContextEstimate.from_context(metadata_context)
    with_schemas = MCPContextEstimate.from_context(schema_context)

    assert metadata_only.selected_tool_count == 1
    assert '"input_schema"' not in metadata_context.content
    assert '"output_schema"' not in metadata_context.content
    assert '"input_schema"' in schema_context.content
    assert '"output_schema"' in schema_context.content
    assert "permanent" not in schema_context.content
    assert with_schemas.character_count > metadata_only.character_count
    assert with_schemas.byte_count > metadata_only.byte_count
    assert with_schemas.estimated_tokens > metadata_only.estimated_tokens


def test_config_flow_returns_framework_independent_initial_values():
    """The MCP flow supplies defaults without importing a host UI framework."""
    result = MCPClientConfigFlow({"mcp_server", "mcp_server_2"}).process()

    assert not result.is_complete
    assert result.errors == {}
    assert result.values == {
        "name": "MCP Server",
        "server_id": "mcp_server_3",
        "description": "",
        "headers": {},
        "use_oauth": False,
    }


def test_config_flow_returns_normalised_configuration():
    """Valid host input is normalized into the public MCP configuration."""
    result = MCPClientConfigFlow().process(
        {
            "name": " Vault MCP ",
            "server_id": " vault ",
            "url": " https://mcp.example.com/api ",
            "description": " Storage tools ",
            "headers": {"Authorization": "Bearer token"},
        }
    )

    assert result.is_complete
    assert result.errors == {}
    assert result.config == MCPServerConfig(
        name="Vault MCP",
        server_id="vault",
        url="https://mcp.example.com/api",
        description="Storage tools",
        headers={"Authorization": "Bearer token"},
    )


def test_config_flow_returns_field_error_and_preserves_values():
    """Invalid data is returned to the host with a stable field error."""
    values = {
        "name": "Vault MCP",
        "server_id": "Not Valid",
        "url": "https://mcp.example.com/api",
        "description": "Storage tools",
        "headers": {},
    }

    result = MCPClientConfigFlow().process(values)

    assert not result.is_complete
    assert result.config is None
    assert result.values == values
    assert result.errors == {"server_id": "invalid_server_id"}


def test_config_flow_rejects_missing_required_field():
    """Processing does not rely on a host framework to require form fields."""
    result = MCPClientConfigFlow().process(
        {
            "name": "Vault MCP",
            "server_id": "vault",
            "description": "Storage tools",
            "headers": {},
        }
    )

    assert not result.is_complete
    assert result.errors == {"url": "invalid_url"}


def test_config_flow_rejects_host_reserved_server_id():
    """The host can reserve IDs without exposing its entry implementation."""
    result = MCPClientConfigFlow({"vault"}).process(
        {
            "name": "Vault MCP",
            "server_id": "vault",
            "url": "https://mcp.example.com/api",
            "description": "Storage tools",
            "headers": {},
        }
    )

    assert not result.is_complete
    assert result.errors == {"server_id": "duplicate_server_id"}
