"""Tests for config_flow.py.

Note: these test validate_input() and the schema-building logic directly,
rather than driving the full flow through Home Assistant's flow manager
(hass.config_entries.flow.async_init(...)) - that needs the real
pytest-homeassistant-custom-component `hass` fixture, which this project's
conftest.py currently shadows project-wide with a lightweight mock for fast
unit testing. Actually exercising the flow steps end-to-end is tracked as a
follow-up (see issue #7) rather than attempted here with a mocked hass that
doesn't behave like the real flow manager.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2 as httpx
from openai import APIConnectionError, APIStatusError, AuthenticationError
import pytest

from custom_components.extended_openai_conversation.config_flow import (
    MCP_ERROR_ACCESS_DENIED,
    MCP_ERROR_AUTHENTICATION_FAILED,
    MCP_ERROR_CANNOT_CONNECT,
    MCP_ERROR_SERVER_ERROR,
    MCP_ERROR_SERVER_NOT_FOUND,
    ExtendedOpenAIAITaskSubentryFlowHandler,
    ExtendedOpenAIConversationConfigFlow,
    ExtendedOpenAISubentryFlowHandler,
    _mcp_failure_reason,
    _mcp_server_options,
    validate_input,
)
from custom_components.extended_openai_conversation.const import (
    CONF_API_PROVIDER,
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_FUNCTION_TOOLS,
    CONF_MCP_AUTO_ADD_DEFAULT_PROMPT,
    CONF_MCP_AUTO_ADD_PERMANENT_TOOLS,
    CONF_MCP_DEFAULT_PROMPT_PLACEMENT,
    CONF_MCP_OAUTH_ACTION,
    CONF_MCP_OAUTH_TIMEOUT,
    CONF_MCP_PERMANENT_TOOLS_PLACEMENT,
    CONF_MCP_SERVERS,
    CONF_PROMPT,
    CONF_SERVICE_TYPE,
    CONF_SKILLS,
    CONF_SKIP_AUTHENTICATION,
    DEFAULT_CONF_BASE_URL,
    DOMAIN,
    MCP_CONTEXT_PLACEMENTS,
    MCP_OAUTH_ACTION_PROCEED,
    MCP_OAUTH_ACTION_RETURN,
    SERVICE_TYPE_AI_ENDPOINT,
    SERVICE_TYPE_MCP,
)
from custom_components.extended_openai_conversation.mcp_client import (
    MCPAuthenticationError,
    MCPDiscoveryResult,
    MCPOAuthClientRegistration,
    MCPOAuthConfiguration,
    MCPServerConfig,
    MCPServerInformation,
    MCPServiceConfig,
    MCPTool,
)
from custom_components.extended_openai_conversation.mcp_client.mcp_const import (
    CONF_MCP_DESCRIPTION,
    CONF_MCP_DISABLED_TOOL_MODE,
    CONF_MCP_DISABLED_TOOLS,
    CONF_MCP_HEADERS,
    CONF_MCP_INCLUDE_PERMANENT_SCHEMAS,
    CONF_MCP_OAUTH_ISSUER,
    CONF_MCP_OAUTH_REGISTRATION_URL,
    CONF_MCP_ON_DEMAND_DISCOVERY,
    CONF_MCP_PERFORM_TOOL_DISCOVERY,
    CONF_MCP_PERMANENT_TOOL_MODE,
    CONF_MCP_PERMANENT_TOOLS,
    CONF_MCP_PROVIDE_DEFAULT_PROMPT,
    CONF_MCP_REGEX_TIMEOUT,
    CONF_MCP_SERVER_ID,
    CONF_MCP_SERVER_URL,
    CONF_MCP_TOOL_NAME_PATTERN,
    CONF_MCP_USE_OAUTH,
)
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import CONF_API_KEY, CONF_TOKEN
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.selector import SelectOptionDict


def _schema_keys(schema: dict) -> set[str]:
    """Extract the plain string keys from a dict of voluptuous Marker -> Selector."""
    return {getattr(key, "schema", key) for key in schema}


def _config_flow(hass) -> ExtendedOpenAIConversationConfigFlow:
    """Create a config flow with the lightweight test Home Assistant mock."""
    flow = ExtendedOpenAIConversationConfigFlow()
    flow.hass = hass
    hass.config_entries.async_entries.return_value = []
    flow._async_service_menu_status_lines = AsyncMock(return_value="status")
    return flow


def _mcp_server_config(**overrides: Any) -> MCPServerConfig:
    """Return the common MCP server configuration used by flow tests."""
    values = {
        "name": "Vault MCP",
        "server_id": "vault",
        "url": "https://mcp.example.com/api",
        **overrides,
    }
    return MCPServerConfig(**values)


def _mcp_server_information(**overrides: Any) -> MCPServerInformation:
    """Return common initialized MCP server information."""
    values = {
        "name": "Vault Server",
        "version": "1.0.0",
        "protocol_version": "2025-06-18",
        "instructions": None,
        "capabilities": {},
        **overrides,
    }
    return MCPServerInformation(**values)


def _mcp_tool(name: str = "read_secret", **overrides: Any) -> MCPTool:
    """Return a common discovered MCP tool."""
    values = {
        "name": name,
        "title": "Read secret" if name == "read_secret" else None,
        "description": f"{name.replace('_', ' ').capitalize()}.",
        "input_schema": {"type": "object"},
        "output_schema": None,
        **overrides,
    }
    return MCPTool(**values)


def _mcp_discovery(*tools: MCPTool) -> MCPDiscoveryResult:
    """Return a discovery result containing the supplied tools."""
    return MCPDiscoveryResult(
        server=_mcp_server_information(),
        tools=tools,
    )


def _mcp_form_values(
    config: MCPServerConfig, **overrides: Any
) -> dict[str, Any]:
    """Return a complete MCP server-form submission."""
    return {
        "name": config.name,
        CONF_MCP_SERVER_ID: config.server_id,
        CONF_MCP_DESCRIPTION: config.description,
        CONF_MCP_SERVER_URL: config.url,
        CONF_MCP_HEADERS: dict(config.headers),
        CONF_MCP_USE_OAUTH: config.use_oauth,
        CONF_MCP_PERFORM_TOOL_DISCOVERY: True,
        CONF_MCP_PROVIDE_DEFAULT_PROMPT: True,
        CONF_MCP_ON_DEMAND_DISCOVERY: True,
        **overrides,
    }


def _configure_mcp_tool_policy(
    flow: ExtendedOpenAIConversationConfigFlow,
    *tools: MCPTool,
    permanent: list[str],
    disabled: list[str],
) -> None:
    """Give a flow a valid discovered tool policy for state tests."""
    flow._mcp_discovery = _mcp_discovery(*tools)
    flow._mcp_options = {
        CONF_MCP_PERFORM_TOOL_DISCOVERY: True,
        CONF_MCP_PROVIDE_DEFAULT_PROMPT: True,
        CONF_MCP_PERMANENT_TOOL_MODE: "selected",
        CONF_MCP_PERMANENT_TOOLS: permanent,
        CONF_MCP_INCLUDE_PERMANENT_SCHEMAS: False,
        CONF_MCP_ON_DEMAND_DISCOVERY: True,
        CONF_MCP_DISABLED_TOOL_MODE: "selected",
        CONF_MCP_DISABLED_TOOLS: disabled,
        CONF_MCP_TOOL_NAME_PATTERN: ".*delete.*",
        CONF_MCP_REGEX_TIMEOUT: 0.1,
    }
    assert not flow._set_mcp_service_config(flow._mcp_policy_values())


async def test_user_step_asks_for_service_type(hass):
    """The initial Add Service step asks which service type to configure."""
    flow = _config_flow(hass)
    flow.async_show_form = MagicMock(return_value={"type": "form"})

    await flow.async_step_user()

    schema = flow.async_show_form.call_args.kwargs["data_schema"]
    assert flow.async_show_form.call_args.kwargs["step_id"] == "user"
    assert _schema_keys(schema.schema) == {CONF_SERVICE_TYPE}
    selector = next(iter(schema.schema.values()))
    assert selector.config["translation_key"] == "service_type"
    assert selector.config["options"] == [
        SERVICE_TYPE_AI_ENDPOINT,
        SERVICE_TYPE_MCP,
    ]


async def test_user_step_routes_to_ai_endpoint(hass):
    """Selecting AI endpoint opens the existing AI endpoint form."""
    flow = _config_flow(hass)
    flow.async_step_ai_endpoint = AsyncMock(return_value={"step_id": "ai_endpoint"})

    result = await flow.async_step_user({CONF_SERVICE_TYPE: SERVICE_TYPE_AI_ENDPOINT})

    assert result == {"step_id": "ai_endpoint"}
    flow.async_step_ai_endpoint.assert_awaited_once_with()


async def test_user_step_routes_to_mcp(hass):
    """Selecting MCP opens the MCP configuration form."""
    flow = _config_flow(hass)
    flow.async_step_mcp = AsyncMock(return_value={"step_id": "mcp"})

    result = await flow.async_step_user({CONF_SERVICE_TYPE: SERVICE_TYPE_MCP})

    assert result == {"step_id": "mcp"}
    flow.async_step_mcp.assert_awaited_once_with()


async def test_ai_endpoint_can_be_saved_without_review(hass):
    """A valid AI endpoint returns to a menu where it can be saved directly."""
    flow = _config_flow(hass)
    flow._selected_service_type = SERVICE_TYPE_AI_ENDPOINT
    flow.async_step_service_menu = AsyncMock(return_value={"type": "menu"})
    user_input = {
        "name": "Local AI",
        CONF_API_KEY: "test-key",
        CONF_BASE_URL: "https://ai.example.com/v1",
    }

    with patch(
        "custom_components.extended_openai_conversation.config_flow.validate_input",
        new=AsyncMock(),
    ):
        result = await flow.async_step_ai_endpoint(user_input)

    assert result == {"type": "menu"}
    assert flow._ai_endpoint_data == {
        **user_input,
        CONF_SERVICE_TYPE: SERVICE_TYPE_AI_ENDPOINT,
    }

    flow.async_create_entry = MagicMock(return_value={"type": "create_entry"})
    result = await flow.async_step_service_save()

    assert result == {"type": "create_entry"}
    assert flow.async_create_entry.call_args.kwargs["data"] == flow._ai_endpoint_data
    assert len(flow.async_create_entry.call_args.kwargs["subentries"]) == 2


async def test_mcp_step_keeps_config_and_returns_to_service_menu(hass):
    """A valid MCP form returns to the central menu and keeps its data."""
    flow = _config_flow(hass)
    flow._selected_service_type = SERVICE_TYPE_MCP
    flow.async_step_service_menu = AsyncMock(return_value={"step_id": "service_menu"})
    flow.async_create_entry = MagicMock(return_value={"type": "create_entry"})
    user_input = {
        "name": "Vault MCP",
        CONF_MCP_SERVER_ID: "vault",
        CONF_MCP_SERVER_URL: "https://mcp.example.com/api",
        CONF_MCP_DESCRIPTION: "Secure storage tools",
        CONF_MCP_HEADERS: {"Authorization": "Bearer test-token"},
        CONF_MCP_USE_OAUTH: False,
        CONF_MCP_PERFORM_TOOL_DISCOVERY: False,
        CONF_MCP_PROVIDE_DEFAULT_PROMPT: True,
        CONF_MCP_ON_DEMAND_DISCOVERY: True,
    }

    with patch(
        "custom_components.extended_openai_conversation.config_flow."
        "MCPClientConfigFlow.check_connection",
        new=AsyncMock(
            return_value=MCPServerInformation(
                name="Vault Server",
                version="1.0.0",
                protocol_version="2025-06-18",
                instructions=None,
                capabilities={},
            )
        ),
    ) as check_connection:
        result = await flow.async_step_mcp(user_input)

    assert result == {"step_id": "service_menu"}
    assert flow._mcp_config is not None
    assert flow._mcp_config.server_id == "vault"
    assert flow._mcp_config.url == "https://mcp.example.com/api"
    check_connection.assert_awaited_once()
    flow.async_step_service_menu.assert_awaited_once_with()
    flow.async_create_entry.assert_not_called()


async def test_mcp_form_includes_oauth_choice(hass):
    """The server form lets the user opt into managed OAuth."""
    flow = _config_flow(hass)
    flow.async_show_form = MagicMock(return_value={"type": "form"})

    result = await flow.async_step_mcp()

    assert result == {"type": "form"}
    schema = flow.async_show_form.call_args.kwargs["data_schema"]
    assert {
        CONF_MCP_USE_OAUTH,
        CONF_MCP_PERFORM_TOOL_DISCOVERY,
        CONF_MCP_PROVIDE_DEFAULT_PROMPT,
        CONF_MCP_ON_DEMAND_DISCOVERY,
    } <= _schema_keys(schema.schema)


async def test_service_menu_exposes_filtering_without_connecting(hass):
    """The menu performs no remote calls and exposes the unlocked filter page."""
    flow = _config_flow(hass)
    flow._selected_service_type = SERVICE_TYPE_MCP
    flow._mcp_config = _mcp_server_config()
    flow.async_show_menu = MagicMock(return_value={"type": "menu"})

    with patch(
        "custom_components.extended_openai_conversation.config_flow."
        "MCPClientConfigFlow.discover",
        new=AsyncMock(),
    ) as mock_discover:
        result = await flow.async_step_service_menu()

    assert result == {"type": "menu"}
    mock_discover.assert_not_awaited()
    assert flow.async_show_menu.call_args.kwargs == {
        "step_id": "service_menu",
        "menu_options": [
            "switch_to_ai_endpoint",
            "mcp",
            "mcp_filter_tools",
        ],
        "description_placeholders": {
            "status_lines": "status",
        },
    }


async def test_ai_endpoint_menu_only_offers_switch_and_save(hass):
    """A configured AI endpoint needs neither another configure step nor review."""
    flow = _config_flow(hass)
    flow._selected_service_type = SERVICE_TYPE_AI_ENDPOINT
    flow._ai_endpoint_data = {
        "name": "Local AI",
        CONF_API_KEY: "test-key",
        CONF_SERVICE_TYPE: SERVICE_TYPE_AI_ENDPOINT,
    }
    flow._ai_connection_status = True
    flow.async_show_menu = MagicMock(return_value={"type": "menu"})

    await flow.async_step_service_menu()

    assert flow.async_show_menu.call_args.kwargs["menu_options"] == [
        "switch_to_mcp",
        "service_save",
    ]


async def test_service_menu_status_lines_only_include_relevant_rows(hass):
    """Optional OAuth and error rows are omitted instead of showing dashes."""
    flow = ExtendedOpenAIConversationConfigFlow()
    flow.hass = hass
    translations = {
        f"component.{DOMAIN}.config.step.service_menu_connection.description": (
            "Connection succeeded: {status}"
        ),
        f"component.{DOMAIN}.config.step.service_menu_oauth.description": (
            "OAuth completed: {status}"
        ),
        f"component.{DOMAIN}.config.step.service_menu_filters.description": (
            "Filters configured: {status}"
        ),
        f"component.{DOMAIN}.config.step.service_menu_error.description": (
            "**Last error: {error}**"
        ),
        f"component.{DOMAIN}.config.error.cannot_connect": "Failed to connect",
    }

    with patch(
        "custom_components.extended_openai_conversation.config_flow."
        "async_get_translations",
        new=AsyncMock(return_value=translations),
    ):
        ai_lines = await flow._async_service_menu_status_lines(
            is_mcp=False,
            connection_status=True,
            oauth_status=None,
            filters_status=None,
        )
        flow._ai_error_reason = "cannot_connect"
        failed_ai_lines = await flow._async_service_menu_status_lines(
            is_mcp=False,
            connection_status=False,
            oauth_status=None,
            filters_status=None,
        )
        flow._mcp_error_reason = MCP_ERROR_CANNOT_CONNECT
        mcp_lines = await flow._async_service_menu_status_lines(
            is_mcp=True,
            connection_status=False,
            oauth_status=None,
            filters_status=True,
        )

    assert ai_lines == "Connection succeeded: ✅"
    assert "Connection succeeded: ❌" in failed_ai_lines
    assert "**Last error: Failed to connect**" in failed_ai_lines
    assert "OAuth completed" not in mcp_lines
    assert "Filters configured: ✅" in mcp_lines
    assert "**Last error: Failed to connect**" in mcp_lines


async def test_oauth_server_requires_authentication_before_filtering(hass):
    """OAuth settings are available before, and filtering only after, auth."""
    flow = _config_flow(hass)
    flow._selected_service_type = SERVICE_TYPE_MCP
    flow._mcp_config = _mcp_server_config(use_oauth=True)
    flow.async_show_menu = MagicMock(return_value={"type": "menu"})

    await flow.async_step_service_menu()

    assert flow.async_show_menu.call_args.kwargs["menu_options"] == [
        "switch_to_ai_endpoint",
        "mcp",
        "mcp_authenticate",
    ]

    flow._mcp_oauth_authenticated = True
    await flow.async_step_service_menu()

    assert flow.async_show_menu.call_args.kwargs["menu_options"] == [
        "switch_to_ai_endpoint",
        "mcp",
        "mcp_authenticate",
        "mcp_filter_tools",
    ]


async def test_oauth_callback_completion_returns_to_service_menu(hass):
    """A successful OAuth callback runs pending discovery and keeps the draft."""
    flow = _config_flow(hass)
    flow._selected_service_type = SERVICE_TYPE_MCP
    flow._mcp_config = _mcp_server_config(use_oauth=True)
    flow._mcp_options = {
        CONF_MCP_PERFORM_TOOL_DISCOVERY: True,
        CONF_MCP_PROVIDE_DEFAULT_PROMPT: True,
        CONF_MCP_ON_DEMAND_DISCOVERY: True,
    }
    flow._mcp_oauth_data = {
        "authorization_url": "https://auth.example.com/authorize",
        "token_url": "https://auth.example.com/token",
        "url": "https://mcp.example.com/api",
    }
    flow.async_step_service_menu = AsyncMock(return_value={"type": "menu"})
    flow._async_test_mcp_server = AsyncMock()

    result = await flow.async_oauth_create_entry(
        {"token": {"access_token": "access-token"}}
    )

    assert result == {"type": "menu"}
    assert flow._mcp_oauth_authenticated
    flow._async_test_mcp_server.assert_awaited_once_with(run_requested_discovery=True)


async def test_oauth_settings_confirm_discovers_metadata(hass):
    """OAuth metadata is fetched only after the OAuth Settings form is submitted."""
    flow = _config_flow(hass)
    flow._mcp_config = _mcp_server_config(use_oauth=True)
    flow.async_show_form = MagicMock(return_value={"type": "form"})
    flow._async_start_mcp_oauth = AsyncMock(return_value={"type": "external"})
    flow._mcp_error_reason = "oauth_discovery_failed"
    oauth_config = MCPOAuthConfiguration(
        authorization_url="https://auth.example.com/authorize",
        token_url="https://auth.example.com/token",
        scopes=("mcp",),
        resource="https://mcp.example.com/api",
        issuer="https://auth.example.com/",
        registration_url="https://auth.example.com/register",
    )

    assert await flow.async_step_mcp_authenticate() == {"type": "form"}
    assert flow._mcp_error_reason is None
    schema = flow.async_show_form.call_args.kwargs["data_schema"]
    assert _schema_keys(schema.schema) == {
        CONF_MCP_OAUTH_ACTION,
        CONF_MCP_OAUTH_TIMEOUT,
    }
    with patch(
        "custom_components.extended_openai_conversation.config_flow."
        "MCPOAuthClientRegistrar.async_discover_mcp_oauth",
        new=AsyncMock(return_value=oauth_config),
    ) as discover_oauth:
        result = await flow.async_step_mcp_authenticate(
            {
                CONF_MCP_OAUTH_ACTION: MCP_OAUTH_ACTION_PROCEED,
                CONF_MCP_OAUTH_TIMEOUT: 120,
            }
        )

    assert result == {"type": "external"}
    discover_oauth.assert_awaited_once_with("https://mcp.example.com/api")
    assert flow._mcp_oauth_data[CONF_MCP_OAUTH_ISSUER] == "https://auth.example.com/"
    assert (
        flow._mcp_oauth_data[CONF_MCP_OAUTH_REGISTRATION_URL]
        == "https://auth.example.com/register"
    )
    flow._async_start_mcp_oauth.assert_awaited_once_with()


async def test_oauth_settings_can_return_to_service_menu(hass):
    """The OAuth settings action can leave without starting discovery."""
    flow = _config_flow(hass)
    flow._mcp_config = _mcp_server_config(use_oauth=True)
    flow.async_step_service_menu = AsyncMock(return_value={"type": "menu"})

    with patch(
        "custom_components.extended_openai_conversation.config_flow."
        "MCPOAuthClientRegistrar.async_discover_mcp_oauth",
        new=AsyncMock(),
    ) as discover_oauth:
        result = await flow.async_step_mcp_authenticate(
            {
                CONF_MCP_OAUTH_ACTION: MCP_OAUTH_ACTION_RETURN,
                CONF_MCP_OAUTH_TIMEOUT: 45,
            }
        )

    assert result == {"type": "menu"}
    assert flow._mcp_oauth_timeout == 45
    discover_oauth.assert_not_awaited()
    flow.async_step_service_menu.assert_awaited_once_with()
    assert not flow._set_mcp_service_config(flow._mcp_policy_values())
    assert flow._service_configuration()[CONF_MCP_OAUTH_TIMEOUT] == 45


def test_zero_oauth_timeout_disables_timeout_task(hass):
    """A zero callback timeout leaves Home Assistant's external step unlimited."""
    flow = _config_flow(hass)
    flow._mcp_oauth_timeout = 0

    flow._schedule_mcp_oauth_timeout()

    assert flow._mcp_oauth_timeout_task is None
    hass.async_create_task.assert_not_called()


async def test_external_oauth_step_schedules_callback_timeout(hass):
    """Opening the authorization website starts one callback timeout task."""
    flow = _config_flow(hass)
    flow._mcp_oauth_timeout = 45
    timeout_task = MagicMock()

    def create_task(coroutine, name):
        coroutine.close()
        assert name == "MCP OAuth callback timeout"
        return timeout_task

    hass.async_create_task.side_effect = create_task
    with patch(
        "homeassistant.helpers.config_entry_oauth2_flow."
        "AbstractOAuth2FlowHandler.async_step_auth",
        new=AsyncMock(return_value={"type": FlowResultType.EXTERNAL_STEP}),
    ):
        result = await flow.async_step_auth()

    assert result == {"type": FlowResultType.EXTERNAL_STEP}
    assert flow._mcp_oauth_timeout_task is timeout_task


async def test_oauth_callback_cancels_callback_timeout(hass):
    """A browser callback cancels the pending timeout before it can fire."""
    flow = _config_flow(hass)
    timeout_task = MagicMock()
    flow._mcp_oauth_timeout_task = timeout_task

    with patch(
        "homeassistant.helpers.config_entry_oauth2_flow."
        "AbstractOAuth2FlowHandler.async_step_auth",
        new=AsyncMock(return_value={"type": FlowResultType.EXTERNAL_STEP_DONE}),
    ):
        result = await flow.async_step_auth({"code": "authorization-code"})

    assert result == {"type": FlowResultType.EXTERNAL_STEP_DONE}
    timeout_task.cancel.assert_called_once_with()
    assert flow._mcp_oauth_timeout_task is None


async def test_oauth_timeout_rejects_current_external_step(hass):
    """An expired callback timer resumes the external step with a timeout error."""
    flow = _config_flow(hass)
    flow.flow_id = "test-flow"
    flow._mcp_oauth_timeout = 120
    flow._mcp_oauth_attempt = 3
    hass.config_entries.flow.async_configure = AsyncMock()

    with patch(
        "custom_components.extended_openai_conversation.config_flow.asyncio.sleep",
        new=AsyncMock(),
    ) as sleep:
        await flow._async_handle_mcp_oauth_timeout(3)

    sleep.assert_awaited_once_with(120)
    hass.config_entries.flow.async_configure.assert_awaited_once_with(
        "test-flow", {"error": "timeout"}
    )


async def test_oauth_timeout_returns_to_menu_with_specific_error(hass):
    """The synthetic timeout rejection retains a specific menu error."""
    flow = _config_flow(hass)
    flow.external_data = {"error": "timeout"}
    flow.async_step_service_menu = AsyncMock(return_value={"type": "menu"})

    result = await flow.async_step_authorize_rejected()

    assert result == {"type": "menu"}
    assert flow._mcp_oauth_status is False
    assert flow._mcp_error_reason == "oauth_authorization_timeout"


async def test_oauth_discovery_failure_is_retained_on_service_menu(hass):
    """OAuth discovery failures remain visible after returning to the menu."""
    flow = _config_flow(hass)
    flow._selected_service_type = SERVICE_TYPE_MCP
    flow._mcp_config = _mcp_server_config(use_oauth=True)
    flow.async_show_menu = MagicMock(return_value={"type": "menu"})

    error_text = "Failed to discover OAuth settings from the configured server URL."
    flow._async_service_menu_status_lines.return_value = error_text
    with patch(
        "custom_components.extended_openai_conversation.config_flow."
        "MCPOAuthClientRegistrar.async_discover_mcp_oauth",
        new=AsyncMock(side_effect=MCPAuthenticationError("missing metadata")),
    ):
        result = await flow.async_step_mcp_authenticate(
            {
                CONF_MCP_OAUTH_ACTION: MCP_OAUTH_ACTION_PROCEED,
                CONF_MCP_OAUTH_TIMEOUT: 120,
            }
        )

    assert result == {"type": "menu"}
    assert flow._mcp_oauth_status is False
    assert flow._mcp_error_reason == "oauth_discovery_failed"
    assert flow.async_show_menu.call_args.kwargs["step_id"] == "service_menu"
    assert flow.async_show_menu.call_args.kwargs["description_placeholders"] == {
        "status_lines": error_text
    }


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (httpx.ConnectError("unreachable"), MCP_ERROR_CANNOT_CONNECT),
        (401, MCP_ERROR_AUTHENTICATION_FAILED),
        (403, MCP_ERROR_ACCESS_DENIED),
        (404, MCP_ERROR_SERVER_NOT_FOUND),
        (503, MCP_ERROR_SERVER_ERROR),
    ],
)
def test_mcp_failure_reason_classifies_transport_errors(error, expected):
    """Nested HTTP failures select a stable localized error key."""
    if isinstance(error, int):
        request = httpx.Request("GET", "https://mcp.example.com/api")
        response = httpx.Response(error, request=request)
        error = httpx.HTTPStatusError(
            "request failed", request=request, response=response
        )
    nested = ExceptionGroup("MCP request failed", [error])

    assert _mcp_failure_reason(nested, "unknown_mcp_error") == expected


async def test_oauth_reuses_credentials_for_discovered_issuer(hass):
    """A matching Application Credential starts authorization immediately."""
    flow = _config_flow(hass)
    flow._mcp_oauth_data = {
        "authorization_url": "https://auth.example.com/authorize",
        "token_url": "https://auth.example.com/token",
        "url": "https://mcp.example.com/api",
        CONF_MCP_OAUTH_ISSUER: "https://auth.example.com/",
    }
    implementation = MagicMock()
    flow.async_step_auth = AsyncMock(return_value={"type": "external"})

    with patch(
        "custom_components.extended_openai_conversation.config_flow."
        "async_get_implementations",
        new=AsyncMock(return_value={"https://auth.example.com/": implementation}),
    ):
        result = await flow._async_start_mcp_oauth()

    assert result == {"type": "external"}
    assert flow.flow_impl is implementation
    flow.async_step_auth.assert_awaited_once_with()


async def test_oauth_dynamically_registers_and_imports_credentials(hass):
    """An advertised registration endpoint creates reusable HA credentials."""
    flow = _config_flow(hass)
    flow._mcp_config = _mcp_server_config(use_oauth=True)
    flow._mcp_oauth_data = {
        "authorization_url": "https://auth.example.com/authorize",
        "token_url": "https://auth.example.com/token",
        "scope": ["mcp"],
        "url": "https://mcp.example.com/api",
        CONF_MCP_OAUTH_ISSUER: "https://auth.example.com/",
        CONF_MCP_OAUTH_REGISTRATION_URL: "https://auth.example.com/register",
    }
    registration = MCPOAuthClientRegistration("generated-client", "")
    implementation = MagicMock()
    flow.async_step_auth = AsyncMock(return_value={"type": "external"})

    with (
        patch(
            "custom_components.extended_openai_conversation.config_flow."
            "async_get_implementations",
            new=AsyncMock(
                side_effect=[{}, {"https://auth.example.com/": implementation}]
            ),
        ),
        patch(
            "custom_components.extended_openai_conversation.config_flow."
            "MCPOAuthClientRegistrar.async_register_mcp_oauth_client",
            new=AsyncMock(return_value=registration),
        ) as register_client,
        patch(
            "custom_components.extended_openai_conversation.config_flow."
            "async_import_client_credential",
            new=AsyncMock(),
        ) as import_credential,
        patch(
            "custom_components.extended_openai_conversation.config_flow."
            "async_get_redirect_uri",
            return_value="https://ha.example.com/auth/external/callback",
        ),
    ):
        result = await flow._async_start_mcp_oauth()

    assert result == {"type": "external"}
    register_client.assert_awaited_once()
    imported = import_credential.await_args.args[2]
    assert imported.client_id == "generated-client"
    assert imported.name == "Vault MCP"
    assert import_credential.await_args.kwargs["auth_domain"] == (
        "https://auth.example.com/"
    )
    assert flow.flow_impl is implementation


async def test_oauth_manual_credentials_are_imported_and_used(hass):
    """Pre-registered client credentials are persisted before authorization."""
    flow = _config_flow(hass)
    flow._mcp_oauth_data = {
        "authorization_url": "https://auth.example.com/authorize",
        "token_url": "https://auth.example.com/token",
        "url": "https://mcp.example.com/api",
        CONF_MCP_OAUTH_ISSUER: "https://auth.example.com/",
    }
    implementation = MagicMock()
    flow.async_step_auth = AsyncMock(return_value={"type": "external"})

    with (
        patch(
            "custom_components.extended_openai_conversation.config_flow."
            "async_import_client_credential",
            new=AsyncMock(),
        ) as import_credential,
        patch(
            "custom_components.extended_openai_conversation.config_flow."
            "async_get_implementations",
            new=AsyncMock(return_value={"https://auth.example.com/": implementation}),
        ),
    ):
        result = await flow.async_step_mcp_credentials(
            {
                "client_id": "registered-client",
                "client_secret": "secret",
                "name": "Vault credential",
            }
        )

    assert result == {"type": "external"}
    imported = import_credential.await_args.args[2]
    assert imported.client_id == "registered-client"
    assert imported.client_secret == "secret"
    assert imported.name == "Vault credential"
    assert flow.flow_impl is implementation


async def test_oauth_credential_save_failure_returns_localized_form(hass):
    """Application Credential storage failures keep the OAuth draft available."""
    flow = _config_flow(hass)
    flow._mcp_oauth_data = {
        "authorization_url": "https://auth.example.com/authorize",
        "token_url": "https://auth.example.com/token",
        "url": "https://mcp.example.com/api",
        CONF_MCP_OAUTH_ISSUER: "https://auth.example.com/",
    }
    flow.async_show_form = MagicMock(return_value={"type": "form"})

    with (
        patch(
            "custom_components.extended_openai_conversation.config_flow."
            "async_import_client_credential",
            new=AsyncMock(side_effect=HomeAssistantError("storage unavailable")),
        ),
        patch(
            "custom_components.extended_openai_conversation.config_flow."
            "async_get_redirect_uri",
            return_value="https://ha.example.com/auth/external/callback",
        ),
    ):
        result = await flow.async_step_mcp_credentials(
            {"client_id": "registered-client", "client_secret": ""}
        )

    assert result == {"type": "form"}
    assert flow.async_show_form.call_args.kwargs["step_id"] == "mcp_credentials"
    assert flow.async_show_form.call_args.kwargs["errors"] == {
        "base": "oauth_credentials_save_failed"
    }


async def test_service_menu_offers_optional_review_and_immediate_save(hass):
    """Valid MCP policies allow saving whether or not review has been opened."""
    flow = _config_flow(hass)
    flow._selected_service_type = SERVICE_TYPE_MCP
    flow._mcp_config = _mcp_server_config()
    assert not flow._set_mcp_service_config(flow._mcp_policy_values())
    flow.async_show_menu = MagicMock(return_value={"type": "menu"})

    await flow.async_step_service_menu()

    options = flow.async_show_menu.call_args.kwargs["menu_options"]
    assert "service_review" in options
    assert "service_save" in options


async def test_reconfigure_menu_keeps_existing_service_type(hass):
    """Reconfiguration does not offer an unsupported entry-type conversion."""
    flow = _config_flow(hass)
    flow._selected_service_type = SERVICE_TYPE_MCP
    flow._reconfigure_entry = MagicMock()
    flow._mcp_config = _mcp_server_config()
    flow.async_show_menu = MagicMock(return_value={"type": "menu"})

    await flow.async_step_service_menu()

    options = flow.async_show_menu.call_args.kwargs["menu_options"]
    assert "switch_to_ai_endpoint" not in options
    assert options == ["mcp", "mcp_filter_tools"]


async def test_reconfigure_restores_discovery_with_refreshable_oauth(hass):
    """Reconfigure refreshes OAuth and restores the requested tool catalogue."""
    flow = _config_flow(hass)
    config = MCPServiceConfig(
        server=_mcp_server_config(use_oauth=True)
    )
    old_token = {
        "access_token": "expired-token",
        "refresh_token": "refresh-token",
        "expires_at": 0,
    }
    refreshed_token = {
        "access_token": "refreshed-token",
        "refresh_token": "refresh-token",
        "expires_at": 9999999999,
    }
    entry = MagicMock()
    entry.entry_id = "entry-vault"
    entry.title = "Vault MCP"
    entry.data = {
        **config.to_dict(),
        CONF_SERVICE_TYPE: SERVICE_TYPE_MCP,
        "authorization_url": "https://auth.example.com/authorize",
        "token_url": "https://auth.example.com/token",
        "oauth_scope": ["mcp"],
        "oauth_issuer": "https://auth.example.com/",
        "auth_implementation": "https://auth.example.com/",
        CONF_TOKEN: old_token,
    }
    flow._get_reconfigure_entry = MagicMock(return_value=entry)
    flow.async_step_service_menu = AsyncMock(return_value={"type": "menu"})
    managed_auth = MagicMock(spec=httpx.Auth)
    discovery_result = _mcp_discovery()

    async def refresh_during_discovery(*_args, **_kwargs) -> MCPDiscoveryResult:
        entry.data = {**entry.data, CONF_TOKEN: refreshed_token}
        return discovery_result

    with (
        patch(
            "custom_components.extended_openai_conversation.config_flow."
            "async_get_mcp_oauth_auth",
            new=AsyncMock(return_value=managed_auth),
        ) as get_auth,
        patch(
            "custom_components.extended_openai_conversation.config_flow."
            "MCPClientConfigFlow.check_connection",
            new=AsyncMock(),
        ) as check_connection,
        patch(
            "custom_components.extended_openai_conversation.config_flow."
            "MCPClientConfigFlow.discover",
            new=AsyncMock(side_effect=refresh_during_discovery),
        ) as discover,
    ):
        result = await flow.async_step_reconfigure()

    assert result == {"type": "menu"}
    get_auth.assert_awaited_once_with(hass, entry)
    discover.assert_awaited_once_with(config.server, auth=managed_auth)
    check_connection.assert_not_awaited()
    assert flow._mcp_connection_status is True
    assert flow._mcp_discovery_status is True
    assert flow._mcp_discovery is discovery_result
    assert flow._mcp_oauth_data is not None
    assert flow._mcp_oauth_data[CONF_TOKEN] == refreshed_token


async def test_requested_initial_discovery_caches_catalogue(hass):
    """The server submission's requested discovery retrieves one catalogue."""
    flow = _config_flow(hass)
    config = _mcp_server_config()
    discovery = _mcp_discovery(
        _mcp_tool(description="Read a stored secret."),
    )
    flow._mcp_config = config
    flow._mcp_options = {
        CONF_MCP_PERFORM_TOOL_DISCOVERY: True,
        CONF_MCP_PROVIDE_DEFAULT_PROMPT: True,
        CONF_MCP_ON_DEMAND_DISCOVERY: True,
    }

    with patch(
        "custom_components.extended_openai_conversation.config_flow."
        "MCPClientConfigFlow.discover",
        new=AsyncMock(return_value=discovery),
    ) as mock_discover:
        await flow._async_test_mcp_server(run_requested_discovery=True)

    mock_discover.assert_awaited_once_with(config, auth=None)
    assert flow._mcp_discovery is discovery
    assert flow._mcp_server_information is discovery.server
    assert flow._mcp_connection_status is True
    assert flow._mcp_discovery_status is True


async def test_mcp_cosmetic_edit_preserves_discovery_and_tool_policy(hass):
    """Changing display metadata does not invalidate the tool catalogue."""
    flow = _config_flow(hass)
    flow._selected_service_type = SERVICE_TYPE_MCP
    flow._mcp_config = _mcp_server_config(description="Old description")
    _configure_mcp_tool_policy(
        flow, _mcp_tool(), permanent=["read_secret"], disabled=[]
    )
    flow._async_test_mcp_server = AsyncMock()
    flow.async_step_service_menu = AsyncMock(return_value={"type": "menu"})

    await flow.async_step_mcp(
        _mcp_form_values(
            flow._mcp_config,
            name="Renamed Vault",
            description="New description",
        )
    )

    assert flow._mcp_discovery is not None
    assert flow._mcp_options[CONF_MCP_PERMANENT_TOOLS] == ["read_secret"]
    assert flow._mcp_options[CONF_MCP_TOOL_NAME_PATTERN] == ".*delete.*"
    assert flow._mcp_service_config is not None
    assert flow._mcp_service_config.server.name == "Renamed Vault"


@pytest.mark.parametrize(
    ("changed_value", "expected_value"),
    [
        ({CONF_MCP_HEADERS: {"Authorization": "Bearer replacement"}}, "headers"),
        ({CONF_MCP_SERVER_URL: "https://new.example.com/mcp"}, "url"),
    ],
)
async def test_mcp_connection_edit_preserves_policy_while_discovery_is_stale(
    hass, changed_value, expected_value
):
    """Transport edits stale discovery without discarding filter choices."""
    flow = _config_flow(hass)
    flow._selected_service_type = SERVICE_TYPE_MCP
    flow._mcp_config = _mcp_server_config()
    _configure_mcp_tool_policy(
        flow, _mcp_tool(), permanent=["read_secret"], disabled=[]
    )
    flow.async_step_service_menu = AsyncMock(return_value={"type": "menu"})

    with patch(
        "custom_components.extended_openai_conversation.config_flow."
        "MCPClientConfigFlow.discover",
        new=AsyncMock(side_effect=RuntimeError("discovery failed")),
    ):
        await flow.async_step_mcp(
            _mcp_form_values(flow._mcp_config, **changed_value)
        )

    assert flow._mcp_discovery is None
    assert flow._mcp_discovery_status is False
    assert flow._mcp_options[CONF_MCP_PERMANENT_TOOLS] == ["read_secret"]
    assert flow._mcp_options[CONF_MCP_TOOL_NAME_PATTERN] == ".*delete.*"
    assert flow._mcp_service_config is not None
    if expected_value == "headers":
        assert flow._mcp_service_config.server.headers == changed_value[
            CONF_MCP_HEADERS
        ]
    else:
        assert flow._mcp_service_config.server.url == changed_value[
            CONF_MCP_SERVER_URL
        ]


async def test_successful_rediscovery_prunes_only_missing_tool_selections(hass):
    """An authoritative catalogue removes stale exact-name selections."""
    flow = _config_flow(hass)
    flow._selected_service_type = SERVICE_TYPE_MCP
    flow._mcp_config = _mcp_server_config()
    _configure_mcp_tool_policy(
        flow,
        _mcp_tool("keep"),
        _mcp_tool("removed_permanent"),
        _mcp_tool("removed_disabled"),
        permanent=["keep", "removed_permanent"],
        disabled=["removed_disabled"],
    )
    flow.async_step_service_menu = AsyncMock(return_value={"type": "menu"})

    with patch(
        "custom_components.extended_openai_conversation.config_flow."
        "MCPClientConfigFlow.discover",
        new=AsyncMock(return_value=_mcp_discovery(_mcp_tool("keep"))),
    ):
        await flow.async_step_mcp(
            _mcp_form_values(
                flow._mcp_config,
                **{CONF_MCP_HEADERS: {"Authorization": "Bearer replacement"}},
            )
        )

    assert flow._mcp_options[CONF_MCP_PERMANENT_TOOLS] == ["keep"]
    assert flow._mcp_options[CONF_MCP_DISABLED_TOOLS] == []
    assert flow._mcp_options[CONF_MCP_TOOL_NAME_PATTERN] == ".*delete.*"


async def test_disabling_discovery_retains_draft_until_save(hass):
    """Hidden tool selections are cleared only when the draft is saved."""
    flow = _config_flow(hass)
    flow._selected_service_type = SERVICE_TYPE_MCP
    flow._mcp_config = _mcp_server_config()
    _configure_mcp_tool_policy(
        flow,
        _mcp_tool("permanent"),
        _mcp_tool("disabled"),
        permanent=["permanent"],
        disabled=["disabled"],
    )
    flow.async_step_service_menu = AsyncMock(return_value={"type": "menu"})

    with patch(
        "custom_components.extended_openai_conversation.config_flow."
        "MCPClientConfigFlow.check_connection",
        new=AsyncMock(return_value=_mcp_server_information()),
    ):
        await flow.async_step_mcp(
            _mcp_form_values(
                flow._mcp_config,
                **{CONF_MCP_PERFORM_TOOL_DISCOVERY: False},
            )
        )

    assert flow._mcp_options[CONF_MCP_PERMANENT_TOOLS] == ["permanent"]
    assert flow._mcp_options[CONF_MCP_DISABLED_TOOLS] == ["disabled"]

    flow.async_create_entry = MagicMock(return_value={"type": "create_entry"})
    await flow.async_step_service_save()

    entry_data = flow.async_create_entry.call_args.kwargs["data"]
    assert entry_data[CONF_MCP_PERMANENT_TOOLS] == []
    assert entry_data[CONF_MCP_DISABLED_TOOLS] == []
    assert entry_data["permanent_tool_catalogue"] == []
    assert entry_data[CONF_MCP_TOOL_NAME_PATTERN] == ".*delete.*"


async def test_filter_tools_step_displays_catalogue_policy_options(hass):
    """Successful discovery exposes permanent and disabled tool controls."""
    flow = _config_flow(hass)
    flow._mcp_config = _mcp_server_config()
    flow._mcp_options = {
        CONF_MCP_PERFORM_TOOL_DISCOVERY: True,
        CONF_MCP_ON_DEMAND_DISCOVERY: True,
    }
    flow._mcp_discovery = _mcp_discovery(
        _mcp_tool(description="Read a stored secret."),
        _mcp_tool("write_secret", description="Write a stored secret."),
    )
    flow.async_show_form = MagicMock(return_value={"type": "form"})

    result = await flow.async_step_mcp_filter_tools()

    assert result == {"type": "form"}
    call = flow.async_show_form.call_args.kwargs
    assert call["step_id"] == "mcp_filter_tools"
    assert call["description_placeholders"] == {"tool_count": "2"}
    schema = call["data_schema"].schema
    assert _schema_keys(schema) == {
        CONF_MCP_PERMANENT_TOOL_MODE,
        CONF_MCP_PERMANENT_TOOLS,
        CONF_MCP_INCLUDE_PERMANENT_SCHEMAS,
        CONF_MCP_DISABLED_TOOL_MODE,
        CONF_MCP_DISABLED_TOOLS,
        CONF_MCP_TOOL_NAME_PATTERN,
        CONF_MCP_REGEX_TIMEOUT,
    }
    permanent_tools = next(
        selector
        for marker, selector in schema.items()
        if getattr(marker, "schema", marker) == CONF_MCP_PERMANENT_TOOLS
    )
    assert permanent_tools.config["multiple"] is True
    assert permanent_tools.config["options"] == [
        {"value": "read_secret", "label": "Read secret"},
        {"value": "write_secret", "label": "write_secret"},
    ]
    permanent_mode = next(
        selector
        for marker, selector in schema.items()
        if getattr(marker, "schema", marker) == CONF_MCP_PERMANENT_TOOL_MODE
    )
    assert permanent_mode.config["translation_key"] == "mcp_tool_filter_mode"
    assert permanent_mode.config["options"] == ["selected", "unselected"]


async def test_filter_tools_without_catalogue_only_displays_regex_options(hass):
    """Manual tool selectors require a discovered catalogue."""
    flow = _config_flow(hass)
    flow._mcp_config = _mcp_server_config()
    flow._mcp_options = {
        CONF_MCP_PERFORM_TOOL_DISCOVERY: False,
        CONF_MCP_ON_DEMAND_DISCOVERY: True,
    }
    flow.async_show_form = MagicMock(return_value={"type": "form"})

    result = await flow.async_step_mcp_filter_tools()

    assert result == {"type": "form"}
    schema = flow.async_show_form.call_args.kwargs["data_schema"].schema
    assert _schema_keys(schema) == {
        CONF_MCP_TOOL_NAME_PATTERN,
        CONF_MCP_REGEX_TIMEOUT,
    }


async def test_filter_submission_clears_omitted_optional_values(hass):
    """Clearing optional filter fields replaces their previously saved values."""
    flow = _config_flow(hass)
    flow._mcp_config = _mcp_server_config()
    flow._mcp_discovery = _mcp_discovery(
        _mcp_tool(description="Read a stored secret."),
    )
    flow._mcp_options = {
        CONF_MCP_PERFORM_TOOL_DISCOVERY: True,
        CONF_MCP_PROVIDE_DEFAULT_PROMPT: True,
        CONF_MCP_PERMANENT_TOOL_MODE: "selected",
        CONF_MCP_PERMANENT_TOOLS: ["read_secret"],
        CONF_MCP_INCLUDE_PERMANENT_SCHEMAS: False,
        CONF_MCP_ON_DEMAND_DISCOVERY: True,
        CONF_MCP_DISABLED_TOOL_MODE: "selected",
        CONF_MCP_DISABLED_TOOLS: ["read_secret"],
        CONF_MCP_TOOL_NAME_PATTERN: ".*delete.*",
        CONF_MCP_REGEX_TIMEOUT: 0.1,
    }
    flow.async_step_service_menu = AsyncMock(return_value={"type": "menu"})

    result = await flow.async_step_mcp_filter_tools(
        {
            CONF_MCP_PERMANENT_TOOL_MODE: "selected",
            CONF_MCP_INCLUDE_PERMANENT_SCHEMAS: False,
            CONF_MCP_DISABLED_TOOL_MODE: "selected",
            CONF_MCP_REGEX_TIMEOUT: 0.1,
        }
    )

    assert result == {"type": "menu"}
    assert flow._mcp_options[CONF_MCP_PERMANENT_TOOLS] == []
    assert flow._mcp_options[CONF_MCP_DISABLED_TOOLS] == []
    assert flow._mcp_options[CONF_MCP_TOOL_NAME_PATTERN] == ""


async def test_filter_submission_can_be_reviewed_and_saved(hass):
    """Saved tool settings can be reviewed before the MCP service is saved."""
    flow = _config_flow(hass)
    flow._selected_service_type = SERVICE_TYPE_MCP
    flow._mcp_config = _mcp_server_config(description="Storage tools")
    flow._mcp_discovery = _mcp_discovery(
        _mcp_tool(
            description="Read a stored secret.",
            input_schema={
                "type": "object",
                "properties": {"key": {"type": "string"}},
            },
        ),
    )
    flow.async_step_service_menu = AsyncMock(return_value={"type": "menu"})
    selection = {
        CONF_MCP_PERFORM_TOOL_DISCOVERY: True,
        CONF_MCP_PROVIDE_DEFAULT_PROMPT: True,
        CONF_MCP_PERMANENT_TOOL_MODE: "selected",
        CONF_MCP_PERMANENT_TOOLS: ["read_secret"],
        CONF_MCP_INCLUDE_PERMANENT_SCHEMAS: True,
        CONF_MCP_ON_DEMAND_DISCOVERY: True,
        CONF_MCP_DISABLED_TOOL_MODE: "selected",
        CONF_MCP_DISABLED_TOOLS: [],
        CONF_MCP_TOOL_NAME_PATTERN: "read_.*",
        CONF_MCP_REGEX_TIMEOUT: 0.1,
    }

    result = await flow.async_step_mcp_filter_tools(selection)

    assert result == {"type": "menu"}
    assert flow._mcp_options == selection
    flow.async_step_service_menu.assert_awaited_once_with()

    flow.async_show_form = MagicMock(return_value={"type": "form"})
    with patch(
        "custom_components.extended_openai_conversation.config_flow."
        "MCPClientConfigFlow.check_connection",
        new=AsyncMock(return_value=flow._mcp_discovery.server),
    ), patch(
        "custom_components.extended_openai_conversation.config_flow."
        "async_get_mcp_default_prompt",
        new=AsyncMock(return_value="Localized MCP prompt"),
    ):
        result = await flow.async_step_mcp_review()

    assert result == {"type": "form"}
    call = flow.async_show_form.call_args.kwargs
    assert call["step_id"] == "mcp_review"
    placeholders = call["description_placeholders"]
    assert placeholders["permanent_tools_selected"] == "✅"
    assert placeholders["selected_tool_count"] == "1"
    assert placeholders["schemas_included"] == "✅"
    assert placeholders["on_demand_discovery"] == "✅"
    assert placeholders["regex_timeout"] == "0.1"
    assert int(placeholders["character_count"].replace(",", "")) > 0
    assert int(placeholders["byte_count"].replace(",", "")) > 0
    assert int(placeholders["estimated_tokens"].replace(",", "")) > 0
    flow.async_step_service_menu.reset_mock()

    result = await flow.async_step_mcp_review({})

    assert result == {"type": "menu"}
    flow.async_step_service_menu.assert_awaited_once_with()

    flow.async_create_entry = MagicMock(return_value={"type": "create_entry"})
    result = await flow.async_step_service_save()

    assert result == {"type": "create_entry"}
    flow.async_create_entry.assert_called_once_with(
        title="Vault MCP",
        data={
            **flow._mcp_service_config.to_dict(),
            CONF_SERVICE_TYPE: SERVICE_TYPE_MCP,
        },
    )


async def test_mcp_without_initial_discovery_can_be_reviewed_and_saved(hass):
    """A service without initial discovery can be reviewed and saved."""
    flow = _config_flow(hass)
    flow._selected_service_type = SERVICE_TYPE_MCP
    flow._mcp_config = _mcp_server_config()
    flow._mcp_discovery = None
    flow._mcp_options = {
        CONF_MCP_PERFORM_TOOL_DISCOVERY: False,
        CONF_MCP_PROVIDE_DEFAULT_PROMPT: True,
        CONF_MCP_ON_DEMAND_DISCOVERY: True,
    }
    flow.async_step_service_menu = AsyncMock(return_value={"type": "menu"})

    await flow.async_step_mcp_filter_tools(
        {
            CONF_MCP_TOOL_NAME_PATTERN: "read_.*",
            CONF_MCP_REGEX_TIMEOUT: 0.1,
        }
    )

    await flow.async_step_mcp_review({})
    flow.async_create_entry = MagicMock(return_value={"type": "create_entry"})
    result = await flow.async_step_service_save()

    assert result == {"type": "create_entry"}
    entry_data = flow.async_create_entry.call_args.kwargs["data"]
    assert entry_data[CONF_SERVICE_TYPE] == SERVICE_TYPE_MCP
    assert entry_data[CONF_MCP_PERMANENT_TOOLS] == []
    assert entry_data["permanent_tool_catalogue"] == []


async def test_mcp_filter_tools_rejects_tool_outside_discovered_catalogue(hass):
    """Crafted tool selections cannot escape the discovered catalogue."""
    flow = _config_flow(hass)
    flow._mcp_config = _mcp_server_config()
    flow._mcp_discovery = _mcp_discovery()
    flow._mcp_options = {
        CONF_MCP_PERFORM_TOOL_DISCOVERY: True,
        CONF_MCP_PROVIDE_DEFAULT_PROMPT: True,
        CONF_MCP_ON_DEMAND_DISCOVERY: True,
    }
    flow.async_show_form = MagicMock(return_value={"type": "form"})

    result = await flow.async_step_mcp_filter_tools(
        {
            CONF_MCP_PERMANENT_TOOL_MODE: "selected",
            CONF_MCP_PERMANENT_TOOLS: ["not_advertised"],
            CONF_MCP_INCLUDE_PERMANENT_SCHEMAS: True,
            CONF_MCP_DISABLED_TOOL_MODE: "selected",
            CONF_MCP_DISABLED_TOOLS: [],
            CONF_MCP_TOOL_NAME_PATTERN: "",
            CONF_MCP_REGEX_TIMEOUT: 0.1,
        }
    )

    assert result == {"type": "form"}
    assert flow._mcp_service_config is None
    assert flow.async_show_form.call_args.kwargs["step_id"] == "mcp_filter_tools"
    assert flow.async_show_form.call_args.kwargs["errors"] == {
        CONF_MCP_PERMANENT_TOOLS: "invalid_permanent_tools"
    }


async def test_mcp_filter_tools_rejects_permanent_disabled_conflict(hass):
    """The filter form flags tools selected for both policy sets."""
    flow = _config_flow(hass)
    flow._mcp_config = _mcp_server_config()
    flow._mcp_discovery = _mcp_discovery(
        _mcp_tool(description="Read a stored secret."),
    )
    flow._mcp_options = {
        CONF_MCP_PERFORM_TOOL_DISCOVERY: True,
        CONF_MCP_PROVIDE_DEFAULT_PROMPT: True,
        CONF_MCP_ON_DEMAND_DISCOVERY: True,
    }
    flow.async_show_form = MagicMock(return_value={"type": "form"})

    result = await flow.async_step_mcp_filter_tools(
        {
            CONF_MCP_PERMANENT_TOOL_MODE: "selected",
            CONF_MCP_PERMANENT_TOOLS: ["read_secret"],
            CONF_MCP_INCLUDE_PERMANENT_SCHEMAS: True,
            CONF_MCP_DISABLED_TOOL_MODE: "selected",
            CONF_MCP_DISABLED_TOOLS: ["read_secret"],
            CONF_MCP_TOOL_NAME_PATTERN: "",
            CONF_MCP_REGEX_TIMEOUT: 0.1,
        }
    )

    assert result == {"type": "form"}
    assert flow._mcp_service_config is None
    assert flow.async_show_form.call_args.kwargs["errors"] == {
        CONF_MCP_DISABLED_TOOLS: "permanent_disabled_tool_conflict"
    }


async def test_initial_discovery_failure_retains_server_draft(hass):
    """Failed discovery is informational and retains the server configuration."""
    flow = _config_flow(hass)
    flow._selected_service_type = SERVICE_TYPE_MCP
    config = _mcp_server_config()
    flow._mcp_config = config
    flow._mcp_options = {CONF_MCP_PERFORM_TOOL_DISCOVERY: True}

    with patch(
        "custom_components.extended_openai_conversation.config_flow."
        "MCPClientConfigFlow.discover",
        new=AsyncMock(side_effect=Exception("Discovery failed")),
    ) as mock_discover:
        await flow._async_test_mcp_server(run_requested_discovery=True)

    mock_discover.assert_awaited_once_with(config, auth=None)
    assert flow._mcp_config is config
    assert flow._mcp_discovery is None
    assert flow._mcp_connection_status is False
    assert flow._mcp_discovery_status is False


async def test_disabled_initial_discovery_only_checks_connection(hass):
    """Disabling initial discovery initializes once without listing tools."""
    flow = _config_flow(hass)
    flow._mcp_config = _mcp_server_config()
    flow._mcp_options = {CONF_MCP_PERFORM_TOOL_DISCOVERY: False}
    server_information = _mcp_server_information()

    with (
        patch(
            "custom_components.extended_openai_conversation.config_flow."
            "MCPClientConfigFlow.discover",
            new=AsyncMock(),
        ) as mock_discover,
        patch(
            "custom_components.extended_openai_conversation.config_flow."
            "MCPClientConfigFlow.check_connection",
            new=AsyncMock(return_value=server_information),
        ) as check_connection,
    ):
        await flow._async_test_mcp_server(run_requested_discovery=True)

    mock_discover.assert_not_awaited()
    check_connection.assert_awaited_once_with(flow._mcp_config, auth=None)
    assert flow._mcp_discovery is None
    assert flow._mcp_server_information is server_information
    assert not flow._set_mcp_service_config(flow._mcp_policy_values())
    assert flow._mcp_service_config is not None
    assert flow._mcp_service_config.server_information is server_information


async def test_mcp_step_rejects_duplicate_server_id(hass):
    """MCP server IDs must be unique across centrally configured services."""
    flow = _config_flow(hass)
    existing_entry = MagicMock()
    existing_entry.data = {
        CONF_SERVICE_TYPE: SERVICE_TYPE_MCP,
        CONF_MCP_SERVER_ID: "vault",
    }
    hass.config_entries.async_entries.return_value = [existing_entry]
    flow.async_show_form = MagicMock(return_value={"type": "form"})

    await flow.async_step_mcp(
        {
            "name": "Another Vault",
            CONF_MCP_SERVER_ID: "vault",
            CONF_MCP_SERVER_URL: "https://mcp.example.com/api",
            CONF_MCP_HEADERS: {},
        }
    )

    assert flow.async_show_form.call_args.kwargs["errors"] == {
        CONF_MCP_SERVER_ID: "duplicate_server_id"
    }


def test_mcp_services_do_not_support_agent_subentries():
    """MCP services are shared connections, not agent containers."""
    entry = MagicMock()
    entry.data = {CONF_SERVICE_TYPE: SERVICE_TYPE_MCP}

    assert (
        ExtendedOpenAIConversationConfigFlow.async_get_supported_subentry_types(entry)
        == {}
    )


def test_legacy_services_still_support_agent_subentries():
    """Entries without a type maintain the legacy AI endpoint behavior."""
    entry = MagicMock()
    entry.data = {}

    subentry_types = (
        ExtendedOpenAIConversationConfigFlow.async_get_supported_subentry_types(entry)
    )

    assert set(subentry_types) == {"conversation", "ai_task_data"}


async def test_validate_input_calls_get_authenticated_client(hass):
    """A valid config is passed through to get_authenticated_client as-is."""
    with patch(
        "custom_components.extended_openai_conversation.config_flow.get_authenticated_client",
        new=AsyncMock(),
    ) as mock_get_client:
        data = {
            CONF_API_KEY: "sk-test",
            CONF_BASE_URL: "https://my-proxy.example.com",
        }
        await validate_input(hass, data)

    mock_get_client.assert_called_once()
    assert mock_get_client.call_args.kwargs["api_key"] == "sk-test"
    assert (
        mock_get_client.call_args.kwargs["base_url"] == "https://my-proxy.example.com"
    )


async def test_validate_input_probes_connection_when_authentication_is_skipped(hass):
    """The config flow still probes an endpoint when client auth checking is skipped."""
    client = MagicMock()
    with (
        patch(
            "custom_components.extended_openai_conversation.config_flow."
            "get_authenticated_client",
            new=AsyncMock(return_value=client),
        ),
        patch(
            "custom_components.extended_openai_conversation.config_flow."
            "async_check_openai_connection",
            new=AsyncMock(),
        ) as check_connection,
    ):
        await validate_input(
            hass,
            {
                CONF_API_KEY: "unused",
                CONF_BASE_URL: "https://local.example.com/v1",
                CONF_SKIP_AUTHENTICATION: True,
            },
        )

    check_connection.assert_awaited_once_with(hass, client)


async def test_validate_input_clears_default_base_url(hass):
    """The default (OpenAI's own) base_url is popped rather than sent through,
    so future changes to OpenAI's own default URL don't require reconfiguring
    existing entries."""
    with patch(
        "custom_components.extended_openai_conversation.config_flow.get_authenticated_client",
        new=AsyncMock(),
    ) as mock_get_client:
        data = {CONF_API_KEY: "sk-test", CONF_BASE_URL: DEFAULT_CONF_BASE_URL}
        await validate_input(hass, data)

    assert CONF_BASE_URL not in data
    assert mock_get_client.call_args.kwargs["base_url"] is None


async def test_validate_input_azure_without_base_url_raises(hass):
    """Azure OpenAI requires a custom base URL - a missing one is a clear error,
    not a confusing failure from the OpenAI client itself."""
    data = {CONF_API_KEY: "sk-test", CONF_API_PROVIDER: "azure"}

    with pytest.raises(HomeAssistantError, match="Azure OpenAI requires"):
        await validate_input(hass, data)


async def test_ai_endpoint_http_error_does_not_report_success(hass):
    """An HTTP error keeps the endpoint form open with a connection error."""
    flow = _config_flow(hass)
    flow._selected_service_type = SERVICE_TYPE_AI_ENDPOINT
    flow.async_show_form = MagicMock(return_value={"type": "form"})
    response = httpx.Response(
        404,
        request=httpx.Request("GET", "https://missing.example.com/v1/models"),
    )
    error = APIStatusError("not found", response=response, body=None)

    with patch(
        "custom_components.extended_openai_conversation.config_flow.validate_input",
        new=AsyncMock(side_effect=error),
    ):
        await flow.async_step_ai_endpoint(
            {
                "name": "Missing",
                CONF_API_KEY: "test-key",
                CONF_BASE_URL: "https://missing.example.com/v1",
            }
        )

    assert flow._ai_endpoint_data is None
    assert flow.async_show_form.call_args.kwargs["errors"] == {
        "base": "cannot_connect"
    }


async def test_ai_endpoint_authentication_error_is_reported(hass):
    """Skip-auth exposes a 401 but allows the endpoint draft to be saved."""
    flow = _config_flow(hass)
    flow._selected_service_type = SERVICE_TYPE_AI_ENDPOINT
    flow.async_show_form = MagicMock(return_value={"type": "form"})
    flow.async_show_menu = MagicMock(return_value={"type": "menu"})
    response = httpx.Response(
        401,
        request=httpx.Request("GET", "https://local.example.com/v1/models"),
    )
    error = AuthenticationError("unauthorized", response=response, body=None)

    with patch(
        "custom_components.extended_openai_conversation.config_flow.validate_input",
        new=AsyncMock(side_effect=error),
    ):
        result = await flow.async_step_ai_endpoint(
            {
                "name": "Local AI",
                CONF_API_KEY: "invalid",
                CONF_BASE_URL: "https://local.example.com/v1",
                CONF_SKIP_AUTHENTICATION: True,
            }
        )

    assert result == {"type": "menu"}
    assert flow._ai_connection_status is False
    assert flow._ai_error_reason == "invalid_auth"
    assert flow.async_show_menu.call_args.kwargs["menu_options"] == [
        "switch_to_mcp",
        "service_save",
    ]


async def test_authentication_error_blocks_when_skip_is_disabled(hass):
    """The normal validation path still blocks an endpoint rejected with 401."""
    flow = _config_flow(hass)
    flow._selected_service_type = SERVICE_TYPE_AI_ENDPOINT
    flow.async_show_form = MagicMock(return_value={"type": "form"})
    response = httpx.Response(
        401,
        request=httpx.Request("GET", "https://local.example.com/v1/models"),
    )

    with patch(
        "custom_components.extended_openai_conversation.config_flow.validate_input",
        new=AsyncMock(
            side_effect=AuthenticationError(
                "unauthorized", response=response, body=None
            )
        ),
    ):
        await flow.async_step_ai_endpoint(
            {
                "name": "Local AI",
                CONF_API_KEY: "invalid",
                CONF_BASE_URL: "https://local.example.com/v1",
                CONF_SKIP_AUTHENTICATION: False,
            }
        )

    assert flow._ai_endpoint_data is None
    assert flow.async_show_form.call_args.kwargs["errors"] == {
        "base": "invalid_auth"
    }


async def test_skip_authentication_allows_saving_unreachable_endpoint(hass):
    """Skip-auth makes connection failures visible but non-blocking."""
    flow = _config_flow(hass)
    flow._selected_service_type = SERVICE_TYPE_AI_ENDPOINT
    flow.async_show_menu = MagicMock(return_value={"type": "menu"})

    with patch(
        "custom_components.extended_openai_conversation.config_flow.validate_input",
        new=AsyncMock(
            side_effect=APIConnectionError(
                request=httpx.Request(
                    "GET", "https://missing.example.com/v1/models"
                )
            )
        ),
    ):
        result = await flow.async_step_ai_endpoint(
            {
                "name": "Missing",
                CONF_API_KEY: "unused",
                CONF_BASE_URL: "https://missing.example.com/v1",
                CONF_SKIP_AUTHENTICATION: True,
            }
        )

    assert result == {"type": "menu"}
    assert flow._ai_connection_status is False
    assert flow._ai_error_reason == "cannot_connect"
    assert "service_save" in flow.async_show_menu.call_args.kwargs["menu_options"]


def test_openai_config_option_schema_includes_core_options():
    """The shared subentry schema includes the expected core option keys,
    given at least one skill is available (see the dedicated test below for
    what happens when none are)."""
    handler = ExtendedOpenAISubentryFlowHandler()
    handler.context = {"source": "user"}

    schema = handler.openai_config_option_schema(
        {
            CONF_MCP_AUTO_ADD_DEFAULT_PROMPT: False,
            CONF_MCP_AUTO_ADD_PERMANENT_TOOLS: False,
        },
        skills=[{"name": "skill_a"}],
        mcp_servers=[SelectOptionDict(value="vault", label="Vault MCP")],
    )

    keys = _schema_keys(schema)
    assert CONF_PROMPT in keys
    assert CONF_CHAT_MODEL in keys
    assert CONF_FUNCTION_TOOLS in keys
    assert CONF_SKILLS in keys
    assert CONF_MCP_SERVERS in keys
    assert CONF_MCP_AUTO_ADD_DEFAULT_PROMPT in keys
    assert CONF_MCP_AUTO_ADD_PERMANENT_TOOLS in keys

    auto_add_key = next(
        key
        for key in schema
        if getattr(key, "schema", key) == CONF_MCP_AUTO_ADD_PERMANENT_TOOLS
    )
    assert auto_add_key.default() is False

    auto_add_prompt_key = next(
        key
        for key in schema
        if getattr(key, "schema", key) == CONF_MCP_AUTO_ADD_DEFAULT_PROMPT
    )
    assert auto_add_prompt_key.default() is False


def test_openai_config_option_schema_omits_skills_field_when_none_available():
    """With no skills loaded, the skills selector is dropped from the schema
    entirely (rather than shown as an always-empty dropdown)."""
    handler = ExtendedOpenAISubentryFlowHandler()
    handler.context = {"source": "user"}

    schema = handler.openai_config_option_schema({}, skills=[])

    assert CONF_SKILLS not in _schema_keys(schema)
    # everything else is still there
    assert CONF_PROMPT in _schema_keys(schema)


def test_openai_config_option_schema_defaults_skills_for_new_subentry():
    """A brand-new subentry with no skills option defaults to all loaded skills."""
    handler = ExtendedOpenAISubentryFlowHandler()
    handler.context = {"source": "user"}  # _is_new == True
    skills = [{"name": "skill_a"}, {"name": "skill_b"}]

    schema = handler.openai_config_option_schema({}, skills=skills)

    skills_key = next(k for k in schema if getattr(k, "schema", k) == CONF_SKILLS)
    assert skills_key.default() == ["skill_a", "skill_b"]


def test_openai_config_option_schema_preserves_existing_skills_on_reconfigure():
    """An existing subentry's already-chosen skills aren't silently reset to
    "all skills" on reconfigure."""
    handler = ExtendedOpenAISubentryFlowHandler()
    handler.context = {"source": "reconfigure"}  # _is_new == False
    skills = [{"name": "skill_a"}, {"name": "skill_b"}]

    schema = handler.openai_config_option_schema(
        {CONF_SKILLS: ["skill_a"]}, skills=skills
    )

    skills_key = next(k for k in schema if getattr(k, "schema", k) == CONF_SKILLS)
    assert skills_key.default() == ["skill_a"]


def test_mcp_server_options_use_stable_ids_and_entry_titles(hass):
    """Agent selectors show configured MCP entries but store stable server IDs."""
    mcp_entry = MagicMock()
    mcp_entry.title = "Vault MCP"
    mcp_entry.data = {
        CONF_SERVICE_TYPE: SERVICE_TYPE_MCP,
        CONF_MCP_SERVER_ID: "vault",
    }
    ai_entry = MagicMock()
    ai_entry.title = "OpenAI"
    ai_entry.data = {CONF_SERVICE_TYPE: SERVICE_TYPE_AI_ENDPOINT}
    hass.config_entries.async_entries.return_value = [mcp_entry, ai_entry]

    assert _mcp_server_options(hass) == [
        {"value": "vault", "label": "Vault MCP"}
    ]


def test_openai_config_option_schema_preserves_existing_mcp_servers():
    """Reconfiguring a conversation agent preserves its selected servers."""
    handler = ExtendedOpenAISubentryFlowHandler()
    handler.context = {"source": "reconfigure"}

    schema = handler.openai_config_option_schema(
        {CONF_MCP_SERVERS: ["vault"]},
        mcp_servers=[SelectOptionDict(value="vault", label="Vault MCP")],
    )

    servers_key = next(
        key for key in schema if getattr(key, "schema", key) == CONF_MCP_SERVERS
    )
    assert servers_key.default() == ["vault"]


async def test_ai_task_schema_includes_mcp_servers(hass):
    """AI Task agents can select from the same configured MCP services."""
    handler = ExtendedOpenAIAITaskSubentryFlowHandler()
    handler.hass = hass
    handler.context = {"source": "reconfigure"}
    handler.options = {
        CONF_MCP_SERVERS: ["vault"],
        CONF_MCP_PERMANENT_TOOLS_PLACEMENT: "user",
        CONF_MCP_DEFAULT_PROMPT_PLACEMENT: "disabled",
    }
    entry = MagicMock()
    entry.state = ConfigEntryState.LOADED
    handler._get_entry = MagicMock(return_value=entry)
    handler.async_show_form = MagicMock(return_value={"type": "form"})
    mcp_entry = MagicMock()
    mcp_entry.title = "Vault MCP"
    mcp_entry.data = {
        CONF_SERVICE_TYPE: SERVICE_TYPE_MCP,
        CONF_MCP_SERVER_ID: "vault",
    }
    hass.config_entries.async_entries.return_value = [mcp_entry]

    await handler.async_step_init()

    schema = handler.async_show_form.call_args.kwargs["data_schema"].schema
    servers_key = next(
        key for key in schema if getattr(key, "schema", key) == CONF_MCP_SERVERS
    )
    assert servers_key.default() == ["vault"]
    expected_placements = {
        CONF_MCP_PERMANENT_TOOLS_PLACEMENT: "user",
        CONF_MCP_DEFAULT_PROMPT_PLACEMENT: "disabled",
    }
    for option, default in expected_placements.items():
        placement_key = next(
            key for key in schema if getattr(key, "schema", key) == option
        )
        assert placement_key.default() == default
        placement_selector = schema[placement_key]
        assert placement_selector.config["options"] == list(MCP_CONTEXT_PLACEMENTS)
        assert (
            placement_selector.config["translation_key"]
            == "mcp_context_placement"
        )
