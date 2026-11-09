"""Config flow for OpenAI Conversation integration."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import replace
import logging
import types
from typing import Any, override

import httpx2 as httpx
from openai._exceptions import APIConnectionError, APIStatusError, AuthenticationError
import probatio as vol
import yaml

from homeassistant.components.application_credentials import (
    ClientCredential,
    async_import_client_credential,
)
from homeassistant.config_entries import (
    SOURCE_REAUTH,
    ConfigEntry,
    ConfigEntryState,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import (
    CONF_API_KEY,
    CONF_CLIENT_ID,
    CONF_CLIENT_SECRET,
    CONF_NAME,
    CONF_TOKEN,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowResultType, UnknownFlow
from homeassistant.exceptions import (
    ConfigEntryAuthFailed,
    HomeAssistantError,
    UnknownImplementationError,
)
from homeassistant.helpers.config_entry_oauth2_flow import (
    AbstractOAuth2FlowHandler,
    async_get_config_entry_implementation,
    async_get_implementations,
    async_get_redirect_uri,
)
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    ObjectSelector,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)
from homeassistant.helpers.translation import async_get_translations

from .application_credentials import oauth_application_context
from .const import (
    API_PROVIDERS,
    CONF_ADVANCED_OPTIONS,
    CONF_API_PROVIDER,
    CONF_API_VERSION,
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_CONTEXT_COMPACTION_PROMPT,
    CONF_CONTEXT_COMPACTION_RECENT_TURNS,
    CONF_CONTEXT_COMPACTION_TARGET,
    CONF_CONTEXT_THRESHOLD,
    CONF_CONTEXT_TRUNCATE_STRATEGY,
    CONF_EXTRA_BODY,
    CONF_FUNCTION_TOOLS,
    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    CONF_MAX_TOKENS,
    CONF_MCP_AUTO_ADD_DEFAULT_PROMPT,
    CONF_MCP_AUTO_ADD_PERMANENT_TOOLS,
    CONF_MCP_DEFAULT_PROMPT_PLACEMENT,
    CONF_MCP_OAUTH_ACTION,
    CONF_MCP_OAUTH_TIMEOUT,
    CONF_MCP_PERMANENT_TOOLS_PLACEMENT,
    CONF_MCP_SERVERS,
    CONF_ORGANIZATION,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_SERVICE_TIER,
    CONF_SERVICE_TYPE,
    CONF_SHORTEN_TOOL_CALL_ID,
    CONF_SKILLS,
    CONF_SKIP_AUTHENTICATION,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONTEXT_TRUNCATE_STRATEGIES,
    DEFAULT_ADVANCED_OPTIONS,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_AI_TASK_OPTIONS,
    DEFAULT_API_PROVIDER,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CONF_BASE_URL,
    DEFAULT_CONF_FUNCTION_TOOLS,
    DEFAULT_CONTEXT_COMPACTION_PROMPT,
    DEFAULT_CONTEXT_COMPACTION_RECENT_TURNS,
    DEFAULT_CONTEXT_COMPACTION_TARGET,
    DEFAULT_CONTEXT_THRESHOLD,
    DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_EXTRA_BODY,
    DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MCP_AUTO_ADD_DEFAULT_PROMPT,
    DEFAULT_MCP_AUTO_ADD_PERMANENT_TOOLS,
    DEFAULT_MCP_DEFAULT_PROMPT_PLACEMENT,
    DEFAULT_MCP_OAUTH_TIMEOUT,
    DEFAULT_MCP_PERMANENT_TOOLS_PLACEMENT,
    DEFAULT_NAME,
    DEFAULT_PROMPT,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_SERVICE_TIER,
    DEFAULT_SHORTEN_TOOL_CALL_ID,
    DEFAULT_SKIP_AUTHENTICATION,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DOMAIN,
    MCP_CONTEXT_PLACEMENTS,
    MCP_OAUTH_ACTION_PROCEED,
    MCP_OAUTH_ACTION_RETURN,
    REASONING_EFFORT_OPTIONS,
    SERVICE_TIER_OPTIONS,
    SERVICE_TYPE_AI_ENDPOINT,
    SERVICE_TYPE_MCP,
    SERVICE_TYPES,
)
from .helpers import (
    async_check_openai_connection,
    get_authenticated_client,
    get_model_config,
    get_service_type,
)
from .mcp_client import (
    CONF_MCP_AUTHORIZATION_URL,
    CONF_MCP_DESCRIPTION,
    CONF_MCP_DISABLED_TOOL_MODE,
    CONF_MCP_DISABLED_TOOLS,
    CONF_MCP_HEADERS,
    CONF_MCP_INCLUDE_PERMANENT_SCHEMAS,
    CONF_MCP_OAUTH_ISSUER,
    CONF_MCP_OAUTH_REGISTRATION_URL,
    CONF_MCP_OAUTH_SCOPE,
    CONF_MCP_ON_DEMAND_DISCOVERY,
    CONF_MCP_PERFORM_TOOL_DISCOVERY,
    CONF_MCP_PERMANENT_TOOL_MODE,
    CONF_MCP_PERMANENT_TOOLS,
    CONF_MCP_REGEX_TIMEOUT,
    CONF_MCP_SERVER_ID,
    CONF_MCP_SERVER_URL,
    CONF_MCP_TOKEN_URL,
    CONF_MCP_TOOL_NAME_PATTERN,
    CONF_MCP_USE_OAUTH,
    DEFAULT_MCP_DISABLED_TOOL_MODE,
    DEFAULT_MCP_INCLUDE_PERMANENT_SCHEMAS,
    DEFAULT_MCP_ON_DEMAND_DISCOVERY,
    DEFAULT_MCP_PERFORM_TOOL_DISCOVERY,
    DEFAULT_MCP_PERMANENT_TOOL_MODE,
    DEFAULT_MCP_REGEX_TIMEOUT,
    DEFAULT_MCP_TOOL_NAME_PATTERN,
    MCP_TOOL_FILTER_SELECTED,
    MCP_TOOL_FILTER_UNSELECTED,
    MCPAuthenticationError,
    MCPClientConfigFlow,
    MCPClientConfigFlowResult,
    MCPConfigError,
    MCPContextEstimate,
    MCPContextRenderer,
    MCPDiscoveryResult,
    MCPOAuthClientRegistrar,
    MCPOAuthClientRegistration,
    MCPOAuthConfiguration,
    MCPProtocolError,
    MCPServerConfig,
    MCPServerInformation,
    MCPServiceConfig,
)
from .mcp_oauth import (
    async_get_mcp_oauth_auth,
    oauth_application_context_from_data,
    static_oauth_auth,
)
from .skills import SkillManager

_LOGGER = logging.getLogger(__name__)

MCP_ERROR_ACCESS_DENIED = "access_denied"
MCP_ERROR_AUTHENTICATION_FAILED = "authentication_failed"
MCP_ERROR_CANNOT_CONNECT = "cannot_connect"
MCP_ERROR_INVALID_RESPONSE = "invalid_mcp_response"
MCP_ERROR_OAUTH_DISCOVERY_FAILED = "oauth_discovery_failed"
MCP_ERROR_OAUTH_TIMEOUT = "oauth_authorization_timeout"
MCP_ERROR_SERVER_ERROR = "mcp_server_error"
MCP_ERROR_SERVER_NOT_FOUND = "mcp_server_not_found"
MCP_ERROR_UNKNOWN = "unknown_mcp_error"


def _mcp_connection_changed(
    previous: MCPServerConfig | None, current: MCPServerConfig
) -> bool:
    """Return whether the remote endpoint or its transport settings changed."""
    if previous is None:
        return True
    return (
        previous.url,
        dict(previous.headers),
        previous.use_oauth,
    ) != (
        current.url,
        dict(current.headers),
        current.use_oauth,
    )


def _mcp_failure_reason(error: BaseException, default: str) -> str:
    """Map nested transport failures to a localized config-flow reason."""
    pending: list[BaseException] = [error]
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
        return MCP_ERROR_AUTHENTICATION_FAILED
    if 403 in statuses:
        return MCP_ERROR_ACCESS_DENIED
    if 404 in statuses:
        return MCP_ERROR_SERVER_NOT_FOUND
    if any(500 <= status <= 599 for status in statuses):
        return MCP_ERROR_SERVER_ERROR

    if any(
        isinstance(current, (httpx.NetworkError, httpx.TimeoutException))
        for current in exceptions
    ):
        return MCP_ERROR_CANNOT_CONNECT
    if any(
        isinstance(current, ConfigEntryAuthFailed) for current in exceptions
    ):
        return MCP_ERROR_AUTHENTICATION_FAILED
    if any(isinstance(current, MCPProtocolError) for current in exceptions):
        return MCP_ERROR_INVALID_RESPONSE
    if any(isinstance(current, MCPAuthenticationError) for current in exceptions):
        return default
    return default

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_SERVICE_TYPE): SelectSelector(
            SelectSelectorConfig(
                options=[item["key"] for item in SERVICE_TYPES],
                mode=SelectSelectorMode.DROPDOWN,
                translation_key="service_type",
            )
        )
    }
)

STEP_AI_ENDPOINT_DATA_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_NAME, default="ChatGPT"): str,
        vol.Required(CONF_API_KEY): str,
        vol.Optional(CONF_BASE_URL, default=DEFAULT_CONF_BASE_URL): str,
        vol.Optional(CONF_API_VERSION): str,
        vol.Optional(CONF_ORGANIZATION): str,
        vol.Optional(
            CONF_SKIP_AUTHENTICATION, default=DEFAULT_SKIP_AUTHENTICATION
        ): bool,
        vol.Optional(CONF_API_PROVIDER, default=DEFAULT_API_PROVIDER): SelectSelector(
            SelectSelectorConfig(
                options=[
                    SelectOptionDict(
                        value=api_provider["key"], label=api_provider["label"]
                    )
                    for api_provider in API_PROVIDERS
                ],
                mode=SelectSelectorMode.DROPDOWN,
            )
        ),
    }
)

DEFAULT_CONF_FUNCTION_TOOLS_STR = yaml.dump(
    DEFAULT_CONF_FUNCTION_TOOLS, sort_keys=False
)

DEFAULT_OPTIONS = types.MappingProxyType(
    {
        CONF_PROMPT: DEFAULT_PROMPT,
        CONF_CHAT_MODEL: DEFAULT_CHAT_MODEL,
        CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
        CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION: DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
        CONF_MCP_AUTO_ADD_DEFAULT_PROMPT: DEFAULT_MCP_AUTO_ADD_DEFAULT_PROMPT,
        CONF_MCP_AUTO_ADD_PERMANENT_TOOLS: DEFAULT_MCP_AUTO_ADD_PERMANENT_TOOLS,
        CONF_TOP_P: DEFAULT_TOP_P,
        CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
        CONF_FUNCTION_TOOLS: DEFAULT_CONF_FUNCTION_TOOLS_STR,
        CONF_CONTEXT_COMPACTION_PROMPT: DEFAULT_CONTEXT_COMPACTION_PROMPT,
        CONF_CONTEXT_COMPACTION_RECENT_TURNS: DEFAULT_CONTEXT_COMPACTION_RECENT_TURNS,
        CONF_CONTEXT_COMPACTION_TARGET: DEFAULT_CONTEXT_COMPACTION_TARGET,
        CONF_CONTEXT_THRESHOLD: DEFAULT_CONTEXT_THRESHOLD,
        CONF_CONTEXT_TRUNCATE_STRATEGY: DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
        CONF_SHORTEN_TOOL_CALL_ID: DEFAULT_SHORTEN_TOOL_CALL_ID,
        CONF_ADVANCED_OPTIONS: DEFAULT_ADVANCED_OPTIONS,
    }
)


def _mcp_server_options(hass: HomeAssistant) -> list[SelectOptionDict]:
    """Return configured MCP servers for agent selectors."""
    return [
        SelectOptionDict(
            value=entry.data[CONF_MCP_SERVER_ID],
            label=entry.title,
        )
        for entry in hass.config_entries.async_entries(DOMAIN)
        if get_service_type(entry) == SERVICE_TYPE_MCP
        and isinstance(entry.data.get(CONF_MCP_SERVER_ID), str)
    ]


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Validate the user input allows us to connect.

    Data has the keys from STEP_USER_DATA_SCHEMA with values provided by the user.
    """
    api_key = data[CONF_API_KEY]
    base_url = data.get(CONF_BASE_URL)
    api_version = data.get(CONF_API_VERSION)
    organization = data.get(CONF_ORGANIZATION)
    skip_authentication = data.get(CONF_SKIP_AUTHENTICATION, False)
    api_provider = data.get(CONF_API_PROVIDER)

    if base_url == DEFAULT_CONF_BASE_URL:
        # Do not set base_url if using OpenAI for case of OpenAI's base_url change
        base_url = None
        data.pop(CONF_BASE_URL)

    if api_provider == "azure" and not base_url:
        raise HomeAssistantError("Azure OpenAI requires a custom base URL.")

    client = await get_authenticated_client(
        hass=hass,
        api_key=api_key,
        base_url=base_url,
        api_version=api_version,
        organization=organization,
        api_provider=api_provider,
        skip_authentication=skip_authentication,
    )
    if skip_authentication:
        await async_check_openai_connection(hass, client)


class ExtendedOpenAIConversationConfigFlow(AbstractOAuth2FlowHandler, domain=DOMAIN):
    """Handle a config flow for OpenAI Conversation."""

    VERSION = 2
    DOMAIN = DOMAIN
    logger = _LOGGER

    def __init__(self) -> None:
        """Initialize the config flow."""
        super().__init__()
        self._mcp_config: MCPServerConfig | None = None
        self._mcp_discovery: MCPDiscoveryResult | None = None
        self._mcp_server_information: MCPServerInformation | None = None
        self._mcp_options: dict[str, Any] | None = None
        self._mcp_service_config: MCPServiceConfig | None = None
        self._mcp_oauth_data: dict[str, Any] | None = None
        self._mcp_oauth_from_entry: bool = False
        self._mcp_oauth_authenticated: bool = False
        self._mcp_oauth_status: bool | None = None
        self._mcp_oauth_timeout: float = DEFAULT_MCP_OAUTH_TIMEOUT
        self._mcp_oauth_timeout_task: asyncio.Task[None] | None = None
        self._mcp_oauth_attempt: int = 0
        self._mcp_connection_status: bool | None = None
        self._mcp_discovery_status: bool | None = None
        self._mcp_error_reason: str | None = None
        self._selected_service_type: str | None = None
        self._ai_endpoint_data: dict[str, Any] | None = None
        self._ai_connection_status: bool | None = None
        self._ai_error_reason: str | None = None
        self._reconfigure_entry: ConfigEntry | None = None

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Choose the type of service to configure."""
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA
            )

        if user_input[CONF_SERVICE_TYPE] == SERVICE_TYPE_MCP:
            self._selected_service_type = SERVICE_TYPE_MCP
            return await self.async_step_mcp()

        self._selected_service_type = SERVICE_TYPE_AI_ENDPOINT
        return await self.async_step_ai_endpoint()

    async def async_step_service_menu(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Show the central service configuration workflow."""
        if self._selected_service_type is None:
            return await self.async_step_user()

        options: list[str] = []
        is_mcp: bool = self._selected_service_type == SERVICE_TYPE_MCP
        if is_mcp:
            options.append("mcp")
            if self._mcp_config is not None:
                if self._mcp_config.use_oauth:
                    options.append("mcp_authenticate")
                if not self._mcp_config.use_oauth or self._mcp_oauth_authenticated:
                    options.append("mcp_filter_tools")
        else:
            options.append("ai_endpoint")

        service_configuration = self._service_configuration()

        if is_mcp and service_configuration is not None:
            options.append("service_review")
        if service_configuration is not None:
            options.append("service_save")

        connection_status = (
            self._mcp_connection_status if is_mcp else self._ai_connection_status
        )
        oauth_status = (
            self._mcp_oauth_status
            if is_mcp and self._mcp_config is not None and self._mcp_config.use_oauth
            else None
        )
        filters_status = self._mcp_service_config is not None if is_mcp else None
        status_lines = await self._async_service_menu_status_lines(
            is_mcp=is_mcp,
            connection_status=connection_status,
            oauth_status=oauth_status,
            filters_status=filters_status,
        )

        return self.async_show_menu(
            step_id="service_menu",
            menu_options=options,
            description_placeholders={"status_lines": status_lines},
        )

    async def _async_service_menu_status_lines(
        self,
        *,
        is_mcp: bool,
        connection_status: bool | None,
        oauth_status: bool | None,
        filters_status: bool | None,
    ) -> str:
        """Build only the localized status rows relevant to this service."""
        translations = await async_get_translations(
            self.hass,
            self.hass.config.language,
            "config",
            integrations={DOMAIN},
        )

        def status_line(key: str, status: bool | None) -> str:
            translation_key = (
                f"component.{DOMAIN}.config.step.service_menu_{key}.description"
            )
            template = translations.get(translation_key, "{status}")
            return template.format(status=self._status_icon(status))

        lines: list[str] = []
        lines.append(status_line("connection", connection_status))
        if oauth_status is not None:
            lines.append(status_line("oauth", oauth_status))
        if filters_status is not None:
            lines.append(status_line("filters", filters_status))
        error_reason = self._mcp_error_reason if is_mcp else self._ai_error_reason
        if error_reason is not None:
            error_key = f"component.{DOMAIN}.config.error.{error_reason}"
            error = translations.get(error_key, error_reason)
            translation_key = (
                f"component.{DOMAIN}.config.step.service_menu_error.description"
            )
            error_template = translations.get(translation_key, "**{error}**")
            lines.append(error_template.format(error=error))

        return "\n\n".join(lines)

    async def async_step_ai_endpoint(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Configure an AI endpoint service."""

        errors = {}

        if user_input is not None:
            error_reason: str | None = None
            try:
                await validate_input(self.hass, user_input)
            except APIConnectionError:
                error_reason = "cannot_connect"
            except AuthenticationError:
                error_reason = "invalid_auth"
            except APIStatusError:
                error_reason = "cannot_connect"
            except Exception:  # pylint: disable=broad-except
                _LOGGER.exception("Unexpected exception")
                error_reason = "unknown"

            if error_reason is None or user_input.get(
                CONF_SKIP_AUTHENTICATION, False
            ):
                data = {**user_input, CONF_SERVICE_TYPE: SERVICE_TYPE_AI_ENDPOINT}
                self._ai_connection_status = error_reason is None
                self._ai_error_reason = error_reason
                self._ai_endpoint_data = data
                return await self.async_step_service_menu()
            errors["base"] = error_reason

        schema = self.add_suggested_values_to_schema(
            STEP_AI_ENDPOINT_DATA_SCHEMA, self._ai_endpoint_data or {}
        )
        return self.async_show_form(
            step_id="ai_endpoint",
            data_schema=schema,
            errors=errors,
        )

    async def async_step_mcp(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Configure an MCP server and perform its requested initial check."""
        if user_input is None:
            self._mcp_error_reason = None
        result: MCPClientConfigFlowResult = MCPClientConfigFlow(
            self._mcp_server_ids()
        ).process(user_input)

        if result.config is not None:
            assert user_input is not None
            previous_config = self._mcp_config
            connection_changed = _mcp_connection_changed(previous_config, result.config)
            if result.config != self._mcp_config:
                self._mcp_error_reason = None
                if connection_changed:
                    self._mcp_discovery = None
                    self._mcp_server_information = None
                    self._mcp_connection_status = None
                    self._mcp_discovery_status = None
                if self._mcp_service_config is not None:
                    self._mcp_service_config = replace(
                        self._mcp_service_config,
                        server=result.config,
                        server_information=(
                            None
                            if connection_changed
                            else self._mcp_service_config.server_information
                        ),
                    )
            if self._oauth_connection_changed(previous_config, result.config):
                self._mcp_oauth_data = None
                self._mcp_oauth_from_entry = False
                self._mcp_oauth_authenticated = False
                self._mcp_oauth_status = None
                self._mcp_error_reason = None
            self._mcp_config = result.config
            self._mcp_options = {
                **(self._mcp_options or {}),
                CONF_MCP_PERFORM_TOOL_DISCOVERY: user_input.get(
                    CONF_MCP_PERFORM_TOOL_DISCOVERY,
                    DEFAULT_MCP_PERFORM_TOOL_DISCOVERY,
                ),
                CONF_MCP_ON_DEMAND_DISCOVERY: user_input.get(
                    CONF_MCP_ON_DEMAND_DISCOVERY,
                    DEFAULT_MCP_ON_DEMAND_DISCOVERY,
                ),
            }
            if result.config.use_oauth and not self._mcp_oauth_authenticated:
                self._mcp_connection_status = None
                self._mcp_discovery_status = None
            else:
                await self._async_test_mcp_server(run_requested_discovery=True)
            if self._mcp_discovery is not None:
                self._reconcile_mcp_tool_selections()
                self._set_mcp_service_config(self._mcp_policy_values())
            elif self._mcp_service_config is None:
                self._set_mcp_service_config(self._mcp_policy_values())
            return await self.async_step_service_menu()

        values = result.values
        if user_input is None and self._mcp_config is not None:
            values = {**self._mcp_config.to_dict(), **(self._mcp_options or {})}

        schema = vol.Schema(
            {
                vol.Required(
                    CONF_NAME,
                    default=values[CONF_NAME],
                ): str,
                vol.Required(
                    CONF_MCP_SERVER_ID,
                    default=values[CONF_MCP_SERVER_ID],
                ): str,
                vol.Optional(
                    CONF_MCP_DESCRIPTION,
                    description={
                        "suggested_value":values.get(CONF_MCP_DESCRIPTION, ""),
                    }
                ): str,
                vol.Required(CONF_MCP_SERVER_URL): str,
                vol.Optional(
                    CONF_MCP_HEADERS,
                    default=values.get(CONF_MCP_HEADERS, {}),
                ): ObjectSelector(),
                vol.Required(
                    CONF_MCP_USE_OAUTH,
                    default=values.get(CONF_MCP_USE_OAUTH, False),
                ): BooleanSelector(),
                vol.Required(
                    CONF_MCP_PERFORM_TOOL_DISCOVERY,
                    default=values.get(
                        CONF_MCP_PERFORM_TOOL_DISCOVERY,
                        DEFAULT_MCP_PERFORM_TOOL_DISCOVERY,
                    ),
                ): BooleanSelector(),
                vol.Required(
                    CONF_MCP_ON_DEMAND_DISCOVERY,
                    default=values.get(
                        CONF_MCP_ON_DEMAND_DISCOVERY,
                        DEFAULT_MCP_ON_DEMAND_DISCOVERY,
                    ),
                ): BooleanSelector(),
            }
        )
        schema = self.add_suggested_values_to_schema(schema, values)

        return self.async_show_form(
            step_id="mcp",
            data_schema=schema,
            errors=result.errors,
        )

    async def async_step_mcp_authenticate(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Discover OAuth endpoints and start Home Assistant's OAuth flow."""
        if self._mcp_config is None or not self._mcp_config.use_oauth:
            return self.async_abort(reason="mcp_invalid_flow_state")
        if user_input is None:
            self._mcp_error_reason = None
            return self._show_mcp_authenticate_form()

        action = user_input.get(CONF_MCP_OAUTH_ACTION)
        if action not in {MCP_OAUTH_ACTION_PROCEED, MCP_OAUTH_ACTION_RETURN}:
            return self._show_mcp_authenticate_form(
                user_input,
                errors={CONF_MCP_OAUTH_ACTION: "invalid_oauth_action"},
            )
        timeout = user_input.get(CONF_MCP_OAUTH_TIMEOUT)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout < 0
        ):
            return self._show_mcp_authenticate_form(
                user_input,
                errors={CONF_MCP_OAUTH_TIMEOUT: "invalid_oauth_timeout"},
            )
        self._mcp_oauth_timeout = float(timeout)
        if action == MCP_OAUTH_ACTION_RETURN:
            self._cancel_mcp_oauth_timeout()
            return await self.async_step_service_menu()

        try:
            oauth_config = await MCPOAuthClientRegistrar.async_discover_mcp_oauth(self._mcp_config.url)
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.exception("Unable to discover MCP OAuth metadata")
            self._mcp_oauth_status = False
            self._mcp_error_reason = _mcp_failure_reason(
                err, MCP_ERROR_OAUTH_DISCOVERY_FAILED
            )
            return await self.async_step_service_menu()

        self._mcp_oauth_data = {
            CONF_MCP_AUTHORIZATION_URL: oauth_config.authorization_url,
            CONF_MCP_TOKEN_URL: oauth_config.token_url,
            CONF_MCP_OAUTH_SCOPE: list(oauth_config.scopes),
            CONF_MCP_OAUTH_ISSUER: oauth_config.issuer,
            CONF_MCP_OAUTH_REGISTRATION_URL: oauth_config.registration_url,
            CONF_MCP_SERVER_URL: oauth_config.resource,
        }
        self._mcp_oauth_from_entry = False
        self._mcp_error_reason = None
        return await self._async_start_mcp_oauth()

    def _show_mcp_authenticate_form(
        self,
        values: dict[str, Any] | None = None,
        *,
        errors: dict[str, str] | None = None,
    ) -> ConfigFlowResult:
        """Show OAuth navigation and callback-timeout settings."""
        schema = vol.Schema(
            {
                vol.Required(
                    CONF_MCP_OAUTH_ACTION, default=MCP_OAUTH_ACTION_PROCEED
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=[MCP_OAUTH_ACTION_PROCEED, MCP_OAUTH_ACTION_RETURN],
                        mode=SelectSelectorMode.LIST,
                        translation_key="mcp_oauth_action",
                    )
                ),
                vol.Required(
                    CONF_MCP_OAUTH_TIMEOUT, default=self._mcp_oauth_timeout
                ): NumberSelector(NumberSelectorConfig(min=0, step=1)),
            }
        )
        if values is not None:
            schema = self.add_suggested_values_to_schema(schema, values)
        return self.async_show_form(
            step_id="mcp_authenticate",
            data_schema=schema,
            errors=errors or {},
        )

    async def _async_start_mcp_oauth(self) -> ConfigFlowResult:
        """Reuse, register, or collect issuer-bound OAuth credentials."""
        if self._mcp_oauth_data is None:
            return self.async_abort(reason="mcp_invalid_flow_state")

        auth_domain = self._mcp_oauth_auth_domain()
        with self._oauth_application_context():
            implementations = await async_get_implementations(self.hass, self.DOMAIN)
        if implementation := implementations.get(auth_domain):
            self.flow_impl = implementation
            return await self.async_step_auth()

        registration_url = self._mcp_oauth_data.get(CONF_MCP_OAUTH_REGISTRATION_URL)
        if registration_url:
            try:
                registration = await MCPOAuthClientRegistrar.async_register_mcp_oauth_client(
                    self._mcp_oauth_configuration(),
                    async_get_redirect_uri(self.hass),
                    client_name=(
                        self._mcp_config.name
                        if self._mcp_config is not None
                        else "Extended OpenAI Conversation 2"
                    ),
                )
            except Exception:  # pylint: disable=broad-except
                _LOGGER.exception("Unable to dynamically register an MCP OAuth client")
                return self._show_mcp_credentials_form(
                    errors={"base": "oauth_registration_failed"}
                )
            return await self._async_import_mcp_credentials(registration)

        return self._show_mcp_credentials_form()

    async def async_step_mcp_credentials(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Collect and persist pre-registered MCP OAuth credentials."""
        if self._mcp_oauth_data is None:
            return self.async_abort(reason="mcp_invalid_flow_state")
        if user_input is None:
            self._mcp_error_reason = None
            return self._show_mcp_credentials_form()

        registration = MCPOAuthClientRegistration(
            client_id=user_input[CONF_CLIENT_ID].strip(),
            client_secret=user_input.get(CONF_CLIENT_SECRET, "").strip(),
        )
        if not registration.client_id:
            return self._show_mcp_credentials_form(
                user_input, errors={CONF_CLIENT_ID: "invalid_client_id"}
            )
        return await self._async_import_mcp_credentials(
            registration,
            name=user_input.get(CONF_NAME),
        )

    def _show_mcp_credentials_form(
        self,
        values: dict[str, Any] | None = None,
        *,
        errors: dict[str, str] | None = None,
    ) -> ConfigFlowResult:
        """Show the form for a pre-registered OAuth client."""
        default_name = (
            self._mcp_config.name
            if self._mcp_config is not None
            else "Extended OpenAI Conversation 2"
        )
        schema = vol.Schema(
            {
                vol.Required(CONF_CLIENT_ID): str,
                vol.Optional(CONF_CLIENT_SECRET, default=""): TextSelector(
                    TextSelectorConfig(type=TextSelectorType.PASSWORD)
                ),
                vol.Optional(CONF_NAME, default=default_name): str,
            }
        )
        if values is not None:
            schema = self.add_suggested_values_to_schema(schema, values)
        return self.async_show_form(
            step_id="mcp_credentials",
            data_schema=schema,
            description_placeholders={
                "redirect_uri": async_get_redirect_uri(self.hass)
            },
            errors=errors or {},
        )

    async def _async_import_mcp_credentials(
        self,
        registration: MCPOAuthClientRegistration,
        *,
        name: str | None = None,
    ) -> ConfigFlowResult:
        """Store credentials in Home Assistant and begin authorization."""
        auth_domain = self._mcp_oauth_auth_domain()
        try:
            await async_import_client_credential(
                self.hass,
                self.DOMAIN,
                ClientCredential(
                    client_id=registration.client_id,
                    client_secret=registration.client_secret,
                    name=name
                    or (
                        self._mcp_config.name
                        if self._mcp_config is not None
                        else "Extended OpenAI Conversation 2"
                    ),
                ),
                auth_domain=auth_domain,
            )
        except Exception:  # pylint: disable=broad-except
            _LOGGER.exception("Unable to save MCP OAuth application credentials")
            return self._show_mcp_credentials_form(
                errors={"base": "oauth_credentials_save_failed"}
            )
        with self._oauth_application_context():
            implementations = await async_get_implementations(self.hass, self.DOMAIN)
        if (implementation := implementations.get(auth_domain)) is None:
            _LOGGER.error("Imported MCP OAuth credentials could not be retrieved")
            return self._show_mcp_credentials_form(
                errors={"base": "oauth_credentials_unavailable"}
            )
        self.flow_impl = implementation
        return await self.async_step_auth()

    @override
    async def async_step_pick_implementation(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Resolve only credentials bound to the current OAuth issuer."""
        if self._mcp_oauth_data is not None:
            return await self._async_start_mcp_oauth()
        return await super().async_step_pick_implementation(user_input)

    @override
    async def async_step_auth(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Start OAuth and bound the wait for its browser callback."""
        if user_input is not None:
            self._cancel_mcp_oauth_timeout()
        result = await super().async_step_auth(user_input)
        if user_input is None and result["type"] == FlowResultType.EXTERNAL_STEP:
            self._schedule_mcp_oauth_timeout()
        return result

    def _schedule_mcp_oauth_timeout(self) -> None:
        """Start a new callback timeout for the current OAuth attempt."""
        self._cancel_mcp_oauth_timeout()
        if self._mcp_oauth_timeout == 0:
            return
        self._mcp_oauth_attempt += 1
        attempt = self._mcp_oauth_attempt
        self._mcp_oauth_timeout_task = self.hass.async_create_task(
            self._async_handle_mcp_oauth_timeout(attempt),
            "MCP OAuth callback timeout",
        )

    def _cancel_mcp_oauth_timeout(self) -> None:
        """Cancel the outstanding callback timeout, if any."""
        self._mcp_oauth_attempt += 1
        task = self._mcp_oauth_timeout_task
        self._mcp_oauth_timeout_task = None
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    async def _async_handle_mcp_oauth_timeout(self, attempt: int) -> None:
        """Reject an external OAuth step whose callback never arrived."""
        try:
            await asyncio.sleep(self._mcp_oauth_timeout)
            if attempt != self._mcp_oauth_attempt:
                return
            self._mcp_oauth_timeout_task = None
            await self.hass.config_entries.flow.async_configure(
                self.flow_id,
                {"error": "timeout"},
            )
        except (asyncio.CancelledError, UnknownFlow):
            return

    @property
    @override
    def extra_authorize_data(self) -> dict[str, str]:
        """Request the scopes advertised by the MCP authorization server."""
        data: dict[str, str] = {"access_type": "offline", "prompt": "consent"}
        if self._mcp_oauth_data is not None and (
            scopes := self._mcp_oauth_data.get(CONF_MCP_OAUTH_SCOPE)
        ):
            data["scope"] = " ".join(scopes)
        return {**data, **super().extra_authorize_data}

    @override
    async def async_oauth_create_entry(self, data: dict[str, Any]) -> ConfigFlowResult:
        """Retain HA-managed OAuth data, then return to the service workflow."""
        self._cancel_mcp_oauth_timeout()
        if self._mcp_oauth_data is None:
            return self.async_abort(reason="mcp_invalid_flow_state")
        oauth_data = {**self._mcp_oauth_data, **data}

        if self.source == SOURCE_REAUTH:
            entry = self._get_reauth_entry()
            return self.async_update_reload_and_abort(
                entry,
                data={**entry.data, **oauth_data},
            )

        self._mcp_oauth_data = oauth_data
        self._mcp_oauth_from_entry = False
        self._mcp_oauth_authenticated = True
        self._mcp_oauth_status = True
        self._mcp_error_reason = None
        await self._async_test_mcp_server(run_requested_discovery=True)
        self._set_mcp_service_config(self._mcp_policy_values())
        return await self.async_step_service_menu()

    @override
    async def async_step_creation(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Return failed OAuth token exchanges to the retained MCP draft."""
        result = await super().async_step_creation(user_input)
        if result["type"] == FlowResultType.ABORT:
            self._mcp_oauth_authenticated = False
            self._mcp_oauth_status = False
            self._mcp_error_reason = MCP_ERROR_AUTHENTICATION_FAILED
            return await self.async_step_service_menu()
        return result

    @override
    async def async_step_authorize_rejected(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Return a rejected OAuth authorization to the retained MCP draft."""
        self._cancel_mcp_oauth_timeout()
        self._mcp_oauth_authenticated = False
        self._mcp_oauth_status = False
        self._mcp_error_reason = (
            MCP_ERROR_OAUTH_TIMEOUT
            if self.external_data.get("error") == "timeout"
            else MCP_ERROR_ACCESS_DENIED
        )
        return await self.async_step_service_menu()

    async def async_step_mcp_filter_tools(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Configure permanent exposure and hard tool-access filters."""
        if self._mcp_config is None:
            return self.async_abort(reason="mcp_invalid_flow_state")
        if user_input is None:
            self._mcp_error_reason = None
        if self._mcp_config.use_oauth and not self._mcp_oauth_authenticated:
            return await self.async_step_service_menu()

        errors: dict[str, str] = {}
        if user_input is not None:
            current_options = self._mcp_policy_values()
            submitted_options = dict(user_input)
            if (
                current_options[CONF_MCP_PERFORM_TOOL_DISCOVERY]
                and self._mcp_discovery is not None
            ):
                submitted_options.setdefault(CONF_MCP_PERMANENT_TOOLS, [])
            if current_options[CONF_MCP_ON_DEMAND_DISCOVERY]:
                submitted_options.setdefault(
                    CONF_MCP_TOOL_NAME_PATTERN, DEFAULT_MCP_TOOL_NAME_PATTERN
                )
                if self._mcp_discovery is not None:
                    submitted_options.setdefault(CONF_MCP_DISABLED_TOOLS, [])
            options = {**current_options, **submitted_options}
            errors = self._set_mcp_service_config(options)
            if not errors:
                return await self.async_step_service_menu()

        values = self._mcp_policy_values()
        schema_fields: dict[Any, Any] = {}
        discovery = self._mcp_discovery
        if values[CONF_MCP_PERFORM_TOOL_DISCOVERY] and discovery is not None:
            tool_options = [
                SelectOptionDict(value=tool.name, label=tool.title or tool.name)
                for tool in discovery.tools
            ]
            mode_options = [MCP_TOOL_FILTER_SELECTED, MCP_TOOL_FILTER_UNSELECTED]
            schema_fields.update(
                {
                    vol.Required(
                        CONF_MCP_PERMANENT_TOOL_MODE,
                        default=values[CONF_MCP_PERMANENT_TOOL_MODE],
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=mode_options,
                            mode=SelectSelectorMode.DROPDOWN,
                            translation_key="mcp_tool_filter_mode",
                        )
                    ),
                    vol.Optional(
                        CONF_MCP_PERMANENT_TOOLS,
                        default=values[CONF_MCP_PERMANENT_TOOLS],
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=tool_options,
                            multiple=True,
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Required(
                        CONF_MCP_INCLUDE_PERMANENT_SCHEMAS,
                        default=values[CONF_MCP_INCLUDE_PERMANENT_SCHEMAS],
                    ): BooleanSelector(),
                }
            )
        if values[CONF_MCP_ON_DEMAND_DISCOVERY]:
            if discovery is not None:
                schema_fields.update(
                    {
                        vol.Required(
                            CONF_MCP_DISABLED_TOOL_MODE,
                            default=values[CONF_MCP_DISABLED_TOOL_MODE],
                        ): SelectSelector(
                            SelectSelectorConfig(
                                options=mode_options,
                                mode=SelectSelectorMode.DROPDOWN,
                                translation_key="mcp_tool_filter_mode",
                            )
                        ),
                        vol.Optional(
                            CONF_MCP_DISABLED_TOOLS,
                            default=values[CONF_MCP_DISABLED_TOOLS],
                        ): SelectSelector(
                            SelectSelectorConfig(
                                options=tool_options,
                                multiple=True,
                                mode=SelectSelectorMode.DROPDOWN,
                            )
                        ),
                    }
                )
            schema_fields.update(
                {
                    vol.Optional(
                        CONF_MCP_TOOL_NAME_PATTERN,
                        description={
                            "suggested_value":values[CONF_MCP_TOOL_NAME_PATTERN],
                        },
                    ): str,
                    vol.Required(
                        CONF_MCP_REGEX_TIMEOUT,
                        default=values[CONF_MCP_REGEX_TIMEOUT],
                    ): NumberSelector(NumberSelectorConfig(min=0.01, step=0.01)),
                }
            )

        return self.async_show_form(
            step_id="mcp_filter_tools",
            data_schema=vol.Schema(schema_fields),
            description_placeholders={
                "tool_count": str(
                    len(self._mcp_discovery.tools) if self._mcp_discovery else 0
                )
            },
            errors=errors,
        )

    async def _async_test_mcp_server(self, *, run_requested_discovery: bool) -> None:
        """Run exactly one requested connection operation and retain its status."""
        if self._mcp_config is None:
            return
        self._mcp_error_reason = None
        processor = MCPClientConfigFlow(self._mcp_server_ids())
        discover = bool(
            (self._mcp_options or {}).get(
                CONF_MCP_PERFORM_TOOL_DISCOVERY,
                DEFAULT_MCP_PERFORM_TOOL_DISCOVERY,
            )
        )
        if not discover:
            self._mcp_discovery_status = None
        try:
            auth = (
                await self._async_mcp_oauth_auth()
                if self._mcp_config.use_oauth
                else None
            )
            if run_requested_discovery and discover:
                self._mcp_discovery = await processor.discover(
                    self._mcp_config, auth=auth
                )
                self._retain_mcp_server_information(self._mcp_discovery.server)
                self._mcp_connection_status = True
                self._mcp_discovery_status = True
                self._mcp_error_reason = None
            else:
                self._retain_mcp_server_information(
                    await processor.check_connection(self._mcp_config, auth=auth)
                )
                self._mcp_connection_status = True
                self._mcp_error_reason = None
                if not discover:
                    self._mcp_discovery_status = None
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.exception("Unable to connect to the MCP server")
            self._mcp_connection_status = False
            self._mcp_error_reason = _mcp_failure_reason(err, MCP_ERROR_UNKNOWN)
            if self._mcp_error_reason == MCP_ERROR_AUTHENTICATION_FAILED:
                self._mcp_oauth_authenticated = False
                self._mcp_oauth_status = False
            if run_requested_discovery and discover:
                self._mcp_discovery = None
                self._mcp_discovery_status = False
        finally:
            self._sync_mcp_oauth_token_from_entry()

    def _retain_mcp_server_information(
        self, server_information: MCPServerInformation
    ) -> None:
        """Retain initialization data from the latest successful connection."""
        self._mcp_server_information = server_information
        if self._mcp_service_config is not None:
            self._mcp_service_config = replace(
                self._mcp_service_config,
                server_information=server_information,
            )

    def _mcp_policy_values(self) -> dict[str, Any]:
        """Return complete policy values using the current draft and defaults."""
        return {
            CONF_MCP_PERFORM_TOOL_DISCOVERY: DEFAULT_MCP_PERFORM_TOOL_DISCOVERY,
            CONF_MCP_PERMANENT_TOOL_MODE: DEFAULT_MCP_PERMANENT_TOOL_MODE,
            CONF_MCP_PERMANENT_TOOLS: [],
            CONF_MCP_INCLUDE_PERMANENT_SCHEMAS: DEFAULT_MCP_INCLUDE_PERMANENT_SCHEMAS,
            CONF_MCP_ON_DEMAND_DISCOVERY: DEFAULT_MCP_ON_DEMAND_DISCOVERY,
            CONF_MCP_DISABLED_TOOL_MODE: DEFAULT_MCP_DISABLED_TOOL_MODE,
            CONF_MCP_DISABLED_TOOLS: [],
            CONF_MCP_TOOL_NAME_PATTERN: DEFAULT_MCP_TOOL_NAME_PATTERN,
            CONF_MCP_REGEX_TIMEOUT: DEFAULT_MCP_REGEX_TIMEOUT,
            **(self._mcp_options or {}),
        }

    def _reconcile_mcp_tool_selections(self) -> None:
        """Discard selected names absent from a successful discovery result."""
        if self._mcp_discovery is None or self._mcp_options is None:
            return
        available_names = {tool.name for tool in self._mcp_discovery.tools}
        for key in (CONF_MCP_PERMANENT_TOOLS, CONF_MCP_DISABLED_TOOLS):
            if key in self._mcp_options:
                self._mcp_options[key] = [
                    name
                    for name in self._mcp_options[key]
                    if name in available_names
                ]

    def _finalize_mcp_service_config(self) -> dict[str, str]:
        """Normalize the draft policy immediately before saving."""
        if self._mcp_config is None:
            return {"base": "mcp_invalid_flow_state"}
        if self._mcp_discovery is not None:
            self._reconcile_mcp_tool_selections()
        options = self._mcp_policy_values()
        if not options[CONF_MCP_PERFORM_TOOL_DISCOVERY]:
            options.update(
                {
                    CONF_MCP_PERMANENT_TOOL_MODE: DEFAULT_MCP_PERMANENT_TOOL_MODE,
                    CONF_MCP_PERMANENT_TOOLS: [],
                    CONF_MCP_INCLUDE_PERMANENT_SCHEMAS: False,
                    CONF_MCP_DISABLED_TOOLS: [],
                }
            )
        if not options[CONF_MCP_ON_DEMAND_DISCOVERY]:
            options.update(
                {
                    CONF_MCP_DISABLED_TOOL_MODE: DEFAULT_MCP_DISABLED_TOOL_MODE,
                    CONF_MCP_DISABLED_TOOLS: [],
                    CONF_MCP_TOOL_NAME_PATTERN: DEFAULT_MCP_TOOL_NAME_PATTERN,
                }
            )
        if self._mcp_discovery is not None:
            return self._set_mcp_service_config(options)
        if self._mcp_service_config is None or not options[
            CONF_MCP_PERFORM_TOOL_DISCOVERY
        ]:
            return self._set_mcp_service_config(options)

        try:
            self._mcp_service_config = replace(
                self._mcp_service_config,
                server=self._mcp_config,
                perform_tool_discovery=options[CONF_MCP_PERFORM_TOOL_DISCOVERY],
                permanent_tool_mode=options[CONF_MCP_PERMANENT_TOOL_MODE],
                permanent_tool_selection=tuple(
                    options[CONF_MCP_PERMANENT_TOOLS]
                ),
                include_permanent_schemas=options[
                    CONF_MCP_INCLUDE_PERMANENT_SCHEMAS
                ],
                on_demand_discovery=options[CONF_MCP_ON_DEMAND_DISCOVERY],
                disabled_tool_mode=options[CONF_MCP_DISABLED_TOOL_MODE],
                disabled_tools=tuple(options[CONF_MCP_DISABLED_TOOLS]),
                tool_name_pattern=options[CONF_MCP_TOOL_NAME_PATTERN],
                regex_timeout=options[CONF_MCP_REGEX_TIMEOUT],
                server_information=self._mcp_server_information,
            )
        except MCPConfigError as err:
            return {err.field: err.code}
        return {}

    def _set_mcp_service_config(self, options: dict[str, Any]) -> dict[str, str]:
        """Validate MCP policies and retain their normalized service config."""
        if self._mcp_config is None:
            return {"base": "mcp_invalid_flow_state"}

        try:
            service_config = MCPServiceConfig.from_selection(
                self._mcp_config,
                self._mcp_discovery,
                perform_tool_discovery=options[CONF_MCP_PERFORM_TOOL_DISCOVERY],
                permanent_tool_mode=options[CONF_MCP_PERMANENT_TOOL_MODE],
                selected_tool_names=options[CONF_MCP_PERMANENT_TOOLS],
                include_permanent_schemas=options[CONF_MCP_INCLUDE_PERMANENT_SCHEMAS],
                on_demand_discovery=options[CONF_MCP_ON_DEMAND_DISCOVERY],
                disabled_tool_mode=options[CONF_MCP_DISABLED_TOOL_MODE],
                disabled_tool_names=options[CONF_MCP_DISABLED_TOOLS],
                tool_name_pattern=options[CONF_MCP_TOOL_NAME_PATTERN],
                regex_timeout=options[CONF_MCP_REGEX_TIMEOUT],
                server_information=self._mcp_server_information,
            )
        except MCPConfigError as err:
            self._mcp_options = options
            self._mcp_service_config = None
            return {err.field: err.code}

        self._mcp_service_config = service_config
        self._mcp_options = service_config.to_policy_dict()
        return {}

    async def async_step_mcp_review(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Review MCP policy and estimated permanent context size."""
        if self._mcp_config is None:
            return self.async_abort(reason="mcp_invalid_flow_state")

        service_config: MCPServiceConfig | None = self._mcp_service_config
        if service_config is None:
            return self.async_abort(reason="mcp_invalid_flow_state")

        if user_input is not None:
            return await self.async_step_service_menu()

        if service_config.server.use_oauth and not self._mcp_oauth_authenticated:
            self._mcp_connection_status = None
        else:
            await self._async_test_mcp_server(run_requested_discovery=False)

        estimate: MCPContextEstimate = MCPContextEstimate.from_context(
            MCPContextRenderer.render_permanent_context(service_config)
        )

        return self.async_show_form(
            step_id="mcp_review",
            data_schema=vol.Schema({}),
            description_placeholders={
                "permanent_tools_selected": self._status_icon(
                    service_config.permanent_tool_mode == MCP_TOOL_FILTER_SELECTED
                ),
                "selected_tool_count": str(estimate.selected_tool_count),
                "schemas_included": self._status_icon(
                    bool(
                        service_config.include_permanent_schemas
                        and estimate.selected_tool_count
                    )
                ),
                "on_demand_discovery": self._status_icon(
                    service_config.on_demand_discovery
                ),
                "disabled_tools_selected": self._status_icon(
                    service_config.disabled_tool_mode == MCP_TOOL_FILTER_SELECTED
                ),
                "disabled_tool_count": str(len(service_config.disabled_tools)),
                "tool_name_pattern": service_config.tool_name_pattern or "",
                "connection_status": self._status_icon(self._mcp_connection_status),
                "discovery_status": self._status_icon(self._mcp_discovery_status),
                "discovered_tool_count": str(
                    len(self._mcp_discovery.tools) if self._mcp_discovery else 0
                ),
                "regex_timeout": str(service_config.regex_timeout),
                "character_count": str(estimate.character_count),
                "byte_count": str(estimate.byte_count),
                "estimated_tokens": str(estimate.estimated_tokens),
            },
        )

    async def async_step_service_review(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Review the current MCP service configuration."""
        if user_input is None:
            self._mcp_error_reason = None
        return await self.async_step_mcp_review(user_input)

    async def async_step_service_save(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Persist the locally validated service configuration."""
        if (
            self._selected_service_type == SERVICE_TYPE_MCP
            and self._finalize_mcp_service_config()
        ):
            return await self.async_step_service_menu()
        data = self._service_configuration()
        if data is None:
            return await self.async_step_service_menu()

        title = str(data.get(CONF_NAME, DEFAULT_NAME))
        if self._reconfigure_entry is not None:
            return self.async_update_reload_and_abort(
                self._reconfigure_entry, title=title, data=data
            )
        if self._selected_service_type == SERVICE_TYPE_MCP:
            return self.async_create_entry(title=title, data=data)
        return self.async_create_entry(
            title=title,
            data=data,
            subentries=[
                {
                    "subentry_type": "conversation",
                    "data": dict(DEFAULT_OPTIONS),
                    "title": DEFAULT_CONVERSATION_NAME,
                    "unique_id": None,
                },
                {
                    "subentry_type": "ai_task_data",
                    "data": dict(DEFAULT_AI_TASK_OPTIONS),
                    "title": DEFAULT_AI_TASK_NAME,
                    "unique_id": None,
                },
            ],
        )

    async def async_step_reauth(
        self, entry_data: Mapping[str, Any]
    ) -> ConfigFlowResult:
        """Start Home Assistant's standard OAuth reauthentication flow."""
        entry = self._get_reauth_entry()
        service_config = MCPServiceConfig.from_dict(entry.data)
        self._selected_service_type = SERVICE_TYPE_MCP
        self._mcp_config = service_config.server
        self._mcp_server_information = service_config.server_information
        self._mcp_oauth_data = dict(entry.data)
        self._mcp_oauth_timeout = float(
            entry.data.get(CONF_MCP_OAUTH_TIMEOUT, DEFAULT_MCP_OAUTH_TIMEOUT)
        )
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: Mapping[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Confirm and launch OAuth reauthentication."""
        if user_input is None:
            return self.async_show_form(step_id="reauth_confirm")
        entry = self._get_reauth_entry()
        try:
            with self._oauth_application_context():
                self.flow_impl = await async_get_config_entry_implementation(
                    self.hass, entry
                )
        except UnknownImplementationError:
            return await self.async_step_mcp_authenticate()
        return await self.async_step_auth()

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Reconfigure an existing service through the shared workflow menu."""
        entry = self._get_reconfigure_entry()
        self._reconfigure_entry = entry
        self._selected_service_type = get_service_type(entry)
        if self._selected_service_type == SERVICE_TYPE_MCP:
            service_config = MCPServiceConfig.from_dict(entry.data)
            self._mcp_config = service_config.server
            self._mcp_discovery = None
            self._mcp_server_information = service_config.server_information
            self._mcp_options = service_config.to_policy_dict()
            self._mcp_service_config = service_config
            if service_config.server.use_oauth:
                self._mcp_oauth_timeout = float(
                    entry.data.get(
                        CONF_MCP_OAUTH_TIMEOUT, DEFAULT_MCP_OAUTH_TIMEOUT
                    )
                )
                self._mcp_oauth_data = {
                    key: entry.data[key]
                    for key in (
                        CONF_MCP_AUTHORIZATION_URL,
                        CONF_MCP_TOKEN_URL,
                        CONF_MCP_OAUTH_SCOPE,
                        CONF_MCP_OAUTH_ISSUER,
                        CONF_MCP_OAUTH_REGISTRATION_URL,
                        CONF_MCP_SERVER_URL,
                        "auth_implementation",
                        CONF_TOKEN,
                    )
                    if key in entry.data
                }
                self._mcp_oauth_authenticated = CONF_TOKEN in self._mcp_oauth_data
                self._mcp_oauth_status = True if self._mcp_oauth_authenticated else None
                self._mcp_oauth_from_entry = self._mcp_oauth_authenticated
        else:
            self._ai_endpoint_data = dict(entry.data)
            try:
                await validate_input(self.hass, dict(self._ai_endpoint_data))
            except AuthenticationError:
                self._ai_connection_status = False
                self._ai_error_reason = "invalid_auth"
            except (APIConnectionError, APIStatusError):
                self._ai_connection_status = False
                self._ai_error_reason = "cannot_connect"
            except Exception:  # pylint: disable=broad-except
                self._ai_connection_status = False
                self._ai_error_reason = "unknown"
            else:
                self._ai_connection_status = True
                self._ai_error_reason = None
        if (
            self._selected_service_type == SERVICE_TYPE_MCP
            and self._mcp_config is not None
            and (not self._mcp_config.use_oauth or self._mcp_oauth_authenticated)
        ):
            await self._async_test_mcp_server(run_requested_discovery=True)
        return await self.async_step_service_menu()

    @staticmethod
    def _status_icon(status: bool | None) -> str:
        """Return a locale-neutral status marker for translated descriptions."""
        if status is None:
            return "—"
        return "✅" if status else "❌"

    def _service_configuration(self) -> dict[str, Any] | None:
        """Return the current valid service mapping without remote calls."""
        if self._selected_service_type == SERVICE_TYPE_AI_ENDPOINT:
            return self._ai_endpoint_data
        service_config = self._mcp_service_config
        if service_config is None:
            return None
        data = {**service_config.to_dict(), CONF_SERVICE_TYPE: SERVICE_TYPE_MCP}
        if service_config.server.use_oauth:
            data[CONF_MCP_OAUTH_TIMEOUT] = self._mcp_oauth_timeout
            if self._mcp_oauth_data is not None:
                data.update(self._mcp_oauth_data)
        return data

    async def _async_mcp_oauth_auth(self) -> httpx.Auth:
        """Return refreshable saved auth or static auth for a new OAuth result."""
        if self._mcp_oauth_data is None or CONF_TOKEN not in self._mcp_oauth_data:
            raise HomeAssistantError("MCP OAuth authentication has not completed")
        if self._mcp_oauth_from_entry and self._reconfigure_entry is not None:
            return await async_get_mcp_oauth_auth(
                self.hass, self._reconfigure_entry
            )
        return static_oauth_auth(self._mcp_oauth_data)

    def _sync_mcp_oauth_token_from_entry(self) -> None:
        """Retain a token refreshed by Home Assistant during reconfiguration."""
        if (
            not self._mcp_oauth_from_entry
            or self._reconfigure_entry is None
            or self._mcp_oauth_data is None
            or CONF_TOKEN not in self._reconfigure_entry.data
        ):
            return
        self._mcp_oauth_data[CONF_TOKEN] = self._reconfigure_entry.data[CONF_TOKEN]

    def _oauth_application_context(self) -> AbstractContextManager[None]:
        """Activate the current server for Home Assistant Application Credentials."""
        if self._mcp_oauth_data is None:
            raise HomeAssistantError("MCP OAuth metadata has not been discovered")
        return oauth_application_context(
            oauth_application_context_from_data(self._mcp_oauth_data)
        )

    def _mcp_oauth_auth_domain(self) -> str:
        """Return the authorization-server issuer that owns these credentials."""
        if self._mcp_oauth_data is None or not (
            issuer := self._mcp_oauth_data.get(CONF_MCP_OAUTH_ISSUER)
        ):
            raise HomeAssistantError("MCP OAuth issuer has not been discovered")
        return str(issuer)

    def _mcp_oauth_configuration(self) -> MCPOAuthConfiguration:
        """Rebuild the framework-independent OAuth discovery result."""
        if self._mcp_oauth_data is None:
            raise HomeAssistantError("MCP OAuth metadata has not been discovered")
        registration_url = self._mcp_oauth_data.get(CONF_MCP_OAUTH_REGISTRATION_URL)
        return MCPOAuthConfiguration(
            authorization_url=str(self._mcp_oauth_data[CONF_MCP_AUTHORIZATION_URL]),
            token_url=str(self._mcp_oauth_data[CONF_MCP_TOKEN_URL]),
            resource=str(self._mcp_oauth_data[CONF_MCP_SERVER_URL]),
            issuer=self._mcp_oauth_auth_domain(),
            registration_url=(str(registration_url) if registration_url else None),
            scopes=tuple(self._mcp_oauth_data.get(CONF_MCP_OAUTH_SCOPE, ())),
        )

    @staticmethod
    def _oauth_connection_changed(
        previous: MCPServerConfig | None, current: MCPServerConfig
    ) -> bool:
        """Return whether existing OAuth credentials are no longer reusable."""
        if previous is None:
            return current.use_oauth
        return (
            previous.url,
            previous.use_oauth,
        ) != (current.url, current.use_oauth)

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this integration."""
        if get_service_type(config_entry) == SERVICE_TYPE_MCP:
            return {}
        return {
            "conversation": ExtendedOpenAISubentryFlowHandler,
            "ai_task_data": ExtendedOpenAIAITaskSubentryFlowHandler,
        }

    def _mcp_server_ids(self) -> set[str]:
        """Return the MCP server IDs reserved by Home Assistant entries."""
        return {
            entry.data[CONF_MCP_SERVER_ID]
            for entry in self.hass.config_entries.async_entries(DOMAIN)
            if (
                self._reconfigure_entry is None
                or entry.entry_id != self._reconfigure_entry.entry_id
            )
            and get_service_type(entry) == SERVICE_TYPE_MCP
            and CONF_MCP_SERVER_ID in entry.data
        }


class ExtendedOpenAISubentryFlowHandler(ConfigSubentryFlow):
    """Flow for managing OpenAI subentries."""

    options: dict[str, Any]
    _temp_data: dict[str, Any] | None = None
    _available_skills: list[dict[str, Any]] | None = None

    @property
    def _is_new(self) -> bool:
        """Return if this is a new subentry."""
        return bool(self.source == "user")

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Add a subentry."""
        self.options = dict(DEFAULT_OPTIONS)
        return await self.async_step_init()

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle reconfiguration of a subentry."""
        self.options = dict(self._get_reconfigure_subentry().data)
        return await self.async_step_init()

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Manage the options."""
        # abort if entry is not loaded
        if self._get_entry().state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        # Load available skills
        if self._available_skills is None:
            self._available_skills = await self._async_get_skills()

        if user_input is not None:
            # Check if advanced options is enabled
            if user_input.get(CONF_ADVANCED_OPTIONS, False):
                # Store data and move to advanced step
                self._temp_data = user_input
                return await self.async_step_advanced()

            # No advanced options, save directly
            if self._is_new:
                title = user_input.get(CONF_NAME, DEFAULT_NAME)
                if CONF_NAME in user_input:
                    del user_input[CONF_NAME]
                return self.async_create_entry(
                    title=title,
                    data=user_input,
                )
            return self.async_update_and_abort(
                self._get_entry(),
                self._get_reconfigure_subentry(),
                data=user_input,
            )

        schema = self.openai_config_option_schema(
            self.options,
            self._available_skills,
            _mcp_server_options(self.hass),
        )

        if self._is_new:
            schema = {
                vol.Optional(CONF_NAME, default=DEFAULT_NAME): str,
                **schema,
            }

        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(schema), self.options
            ),
        )

    async def async_step_advanced(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle advanced options step."""
        if user_input is not None:
            # Merge advanced options with temp data
            final_data = {**(self._temp_data or {}), **user_input}

            if self._is_new:
                title = final_data.get(CONF_NAME, DEFAULT_NAME)
                final_data.pop(CONF_NAME, None)
                return self.async_create_entry(
                    title=title,
                    data=final_data,
                )
            return self.async_update_and_abort(
                self._get_entry(),
                self._get_reconfigure_subentry(),
                data=final_data,
            )

        # Build schema for advanced options based on selected model
        chat_model = (self._temp_data or {}).get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        model_config = get_model_config(
            chat_model, self._get_entry().data.get(CONF_BASE_URL)
        )

        schema: dict[Any, Any] = {}

        # Add top_p if supported
        if model_config["supports_top_p"]:
            schema[
                vol.Optional(
                    CONF_TOP_P,
                    default=DEFAULT_TOP_P,
                )
            ] = NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05))

        # Add temperature if supported
        if model_config["supports_temperature"]:
            schema[
                vol.Optional(
                    CONF_TEMPERATURE,
                    default=DEFAULT_TEMPERATURE,
                )
            ] = NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05))

        # Add reasoning_effort if supported (o1, o3, o4, gpt-5 models)
        if model_config.get("supports_reasoning_effort"):
            schema[
                vol.Optional(
                    CONF_REASONING_EFFORT,
                    default=DEFAULT_REASONING_EFFORT,
                )
            ] = SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(value=opt, label=opt.capitalize())
                        for opt in REASONING_EFFORT_OPTIONS
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                )
            )

        # Add service_tier if supported (o3, o4, gpt-5 models)
        if model_config.get("supports_service_tier"):
            schema[
                vol.Optional(
                    CONF_SERVICE_TIER,
                    default=DEFAULT_SERVICE_TIER,
                )
            ] = SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(value=opt, label=opt.capitalize())
                        for opt in SERVICE_TIER_OPTIONS
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                )
            )

        # Add extra_body - passthrough for OpenAI-compatible backends
        # (ollama, llama.cpp, vLLM, LM Studio). Accepts a Jinja-templatable
        # JSON string; empty disables.
        schema[
            vol.Optional(
                CONF_EXTRA_BODY,
                default=DEFAULT_EXTRA_BODY,
            )
        ] = TemplateSelector()

        # Add shorten_tool_call_id option (for Mistral AI compatibility)
        schema[
            vol.Optional(
                CONF_SHORTEN_TOOL_CALL_ID,
                default=DEFAULT_SHORTEN_TOOL_CALL_ID,
            )
        ] = BooleanSelector()

        if (self._temp_data or {}).get(
            CONF_CONTEXT_TRUNCATE_STRATEGY, DEFAULT_CONTEXT_TRUNCATE_STRATEGY
        ) == "compact":
            schema[
                vol.Optional(
                    CONF_CONTEXT_COMPACTION_TARGET,
                    default=DEFAULT_CONTEXT_COMPACTION_TARGET,
                )
            ] = NumberSelector(
                NumberSelectorConfig(
                    min=1, max=100, step=1, unit_of_measurement="%"
                )
            )
            schema[
                vol.Optional(
                    CONF_CONTEXT_COMPACTION_RECENT_TURNS,
                    default=DEFAULT_CONTEXT_COMPACTION_RECENT_TURNS,
                )
            ] = NumberSelector(NumberSelectorConfig(min=0, step=1))
            schema[
                vol.Optional(
                    CONF_CONTEXT_COMPACTION_PROMPT,
                    default=DEFAULT_CONTEXT_COMPACTION_PROMPT,
                )
            ] = TextSelector(TextSelectorConfig(multiline=True))

        return self.async_show_form(
            step_id="advanced",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(schema), self.options
            ),
        )

    async def _async_get_skills(self) -> list[dict[str, Any]]:
        """Load available skills using SkillManager."""
        skill_manager = await SkillManager.async_get_instance(self.hass)
        return [
            {
                "name": skill.name,
                "description": skill.description,
            }
            for skill in skill_manager.get_all_skills()
        ]

    def openai_config_option_schema(
        self,
        options: dict[str, Any],
        skills: list[dict[str, Any]] | None = None,
        mcp_servers: list[SelectOptionDict] | None = None,
    ) -> dict:
        """Return a schema for OpenAI completion options."""
        # If creating a new subentry and no skills in options, default to all loaded skills
        default_skills: list[str] = []
        if self._is_new and CONF_SKILLS not in options and skills:
            default_skills = [skill["name"] for skill in skills]

        current_skills = options.get(CONF_SKILLS, default_skills)

        schema: dict = {
            vol.Optional(
                CONF_PROMPT,
                default=DEFAULT_PROMPT,
            ): TemplateSelector(),
            vol.Optional(
                CONF_CHAT_MODEL,
                default=DEFAULT_CHAT_MODEL,
            ): str,
            vol.Optional(
                CONF_MAX_TOKENS,
                default=DEFAULT_MAX_TOKENS,
            ): int,
            vol.Optional(
                CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
                default=DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
            ): int,
            vol.Optional(CONF_SKILLS, default=current_skills): SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(
                            value=skill["name"],
                            label=skill["name"],
                        )
                        for skill in (skills or [])
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                    multiple=True,
                )
            ),
            vol.Optional(
                CONF_MCP_SERVERS,
                default=options.get(CONF_MCP_SERVERS, []),
            ): SelectSelector(
                SelectSelectorConfig(
                    options=mcp_servers or [],
                    mode=SelectSelectorMode.DROPDOWN,
                    multiple=True,
                )
            ),
            vol.Optional(
                CONF_MCP_AUTO_ADD_PERMANENT_TOOLS,
                default=options.get(
                    CONF_MCP_AUTO_ADD_PERMANENT_TOOLS,
                    DEFAULT_MCP_AUTO_ADD_PERMANENT_TOOLS,
                ),
            ): BooleanSelector(),
            vol.Optional(
                CONF_MCP_AUTO_ADD_DEFAULT_PROMPT,
                default=options.get(
                    CONF_MCP_AUTO_ADD_DEFAULT_PROMPT,
                    DEFAULT_MCP_AUTO_ADD_DEFAULT_PROMPT,
                ),
            ): BooleanSelector(),
            vol.Optional(
                CONF_FUNCTION_TOOLS,
                default=DEFAULT_CONF_FUNCTION_TOOLS_STR,
            ): TemplateSelector(),
            vol.Optional(
                CONF_CONTEXT_THRESHOLD,
                default=DEFAULT_CONTEXT_THRESHOLD,
            ): int,
            vol.Optional(
                CONF_CONTEXT_TRUNCATE_STRATEGY,
                default=DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
            ): SelectSelector(
                SelectSelectorConfig(
                    options=list(CONTEXT_TRUNCATE_STRATEGIES),
                    mode=SelectSelectorMode.DROPDOWN,
                    translation_key="context_truncate_strategy",
                )
            ),
            vol.Optional(
                CONF_ADVANCED_OPTIONS,
                default=DEFAULT_ADVANCED_OPTIONS,
            ): BooleanSelector(),
        }

        # Remove skills field if no skills available
        if not skills:
            schema = {
                key: value
                for key, value in schema.items()
                if not (isinstance(key, vol.Optional) and key.schema == CONF_SKILLS)
            }

        return schema


class ExtendedOpenAIAITaskSubentryFlowHandler(ConfigSubentryFlow):
    """Flow for managing AI Task subentries."""

    options: dict[str, Any]
    _temp_data: dict[str, Any] | None = None

    @property
    def _is_new(self) -> bool:
        """Return if this is a new subentry."""
        return bool(self.source == "user")

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Add a subentry."""
        self.options = dict(DEFAULT_AI_TASK_OPTIONS)
        return await self.async_step_init()

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle reconfiguration of a subentry."""
        self.options = dict(self._get_reconfigure_subentry().data)
        return await self.async_step_init()

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Manage the options."""
        # Abort if entry is not loaded
        if self._get_entry().state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        if user_input is not None:
            # Check if advanced options is enabled
            if user_input.get(CONF_ADVANCED_OPTIONS, False):
                # Store data and move to advanced step
                self._temp_data = user_input
                return await self.async_step_advanced()

            # No advanced options, save directly
            if self._is_new:
                title = user_input.get(CONF_NAME, DEFAULT_AI_TASK_NAME)
                if CONF_NAME in user_input:
                    del user_input[CONF_NAME]
                return self.async_create_entry(
                    title=title,
                    data=user_input,
                )
            return self.async_update_and_abort(
                self._get_entry(),
                self._get_reconfigure_subentry(),
                data=user_input,
            )

        schema: dict = {}

        if self._is_new:
            schema[vol.Optional(CONF_NAME, default=DEFAULT_AI_TASK_NAME)] = str

        schema.update(
            {
                vol.Optional(
                    CONF_CHAT_MODEL,
                    default=DEFAULT_CHAT_MODEL,
                ): str,
                vol.Optional(
                    CONF_MAX_TOKENS,
                    default=DEFAULT_MAX_TOKENS,
                ): int,
                vol.Optional(
                    CONF_MCP_SERVERS,
                    default=self.options.get(CONF_MCP_SERVERS, []),
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=_mcp_server_options(self.hass),
                        mode=SelectSelectorMode.DROPDOWN,
                        multiple=True,
                    )
                ),
                vol.Optional(
                    CONF_MCP_PERMANENT_TOOLS_PLACEMENT,
                    default=self.options.get(
                        CONF_MCP_PERMANENT_TOOLS_PLACEMENT,
                        DEFAULT_MCP_PERMANENT_TOOLS_PLACEMENT,
                    ),
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=list(MCP_CONTEXT_PLACEMENTS),
                        mode=SelectSelectorMode.DROPDOWN,
                        translation_key="mcp_context_placement",
                    )
                ),
                vol.Optional(
                    CONF_MCP_DEFAULT_PROMPT_PLACEMENT,
                    default=self.options.get(
                        CONF_MCP_DEFAULT_PROMPT_PLACEMENT,
                        DEFAULT_MCP_DEFAULT_PROMPT_PLACEMENT,
                    ),
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=list(MCP_CONTEXT_PLACEMENTS),
                        mode=SelectSelectorMode.DROPDOWN,
                        translation_key="mcp_context_placement",
                    )
                ),
                vol.Optional(
                    CONF_CONTEXT_THRESHOLD,
                    default=self.options.get(
                        CONF_CONTEXT_THRESHOLD, DEFAULT_CONTEXT_THRESHOLD
                    ),
                ): int,
                vol.Optional(
                    CONF_CONTEXT_TRUNCATE_STRATEGY,
                    default=self.options.get(
                        CONF_CONTEXT_TRUNCATE_STRATEGY,
                        DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
                    ),
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=list(CONTEXT_TRUNCATE_STRATEGIES),
                        mode=SelectSelectorMode.DROPDOWN,
                        translation_key="context_truncate_strategy",
                    )
                ),
                vol.Optional(
                    CONF_ADVANCED_OPTIONS,
                    default=DEFAULT_ADVANCED_OPTIONS,
                ): BooleanSelector(),
            }
        )

        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(schema), self.options
            ),
        )

    async def async_step_advanced(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle advanced options step."""
        if user_input is not None:
            # Merge advanced options with temp data
            final_data = {**(self._temp_data or {}), **user_input}

            if self._is_new:
                title = final_data.get(CONF_NAME, DEFAULT_AI_TASK_NAME)
                final_data.pop(CONF_NAME, None)
                return self.async_create_entry(
                    title=title,
                    data=final_data,
                )
            return self.async_update_and_abort(
                self._get_entry(),
                self._get_reconfigure_subentry(),
                data=final_data,
            )

        # Build schema for advanced options based on selected model
        chat_model = (self._temp_data or {}).get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        model_config = get_model_config(
            chat_model, self._get_entry().data.get(CONF_BASE_URL)
        )

        schema: dict[Any, Any] = {}

        # Add top_p if supported
        if model_config["supports_top_p"]:
            schema[
                vol.Optional(
                    CONF_TOP_P,
                    default=DEFAULT_TOP_P,
                )
            ] = NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05))

        # Add temperature if supported
        if model_config["supports_temperature"]:
            schema[
                vol.Optional(
                    CONF_TEMPERATURE,
                    default=DEFAULT_TEMPERATURE,
                )
            ] = NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05))

        # Add reasoning_effort if supported (o1, o3, o4, gpt-5 models)
        if model_config.get("supports_reasoning_effort"):
            schema[
                vol.Optional(
                    CONF_REASONING_EFFORT,
                    default=DEFAULT_REASONING_EFFORT,
                )
            ] = SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(value=opt, label=opt.capitalize())
                        for opt in REASONING_EFFORT_OPTIONS
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                )
            )

        # Add service_tier if supported (o3, o4, gpt-5 models)
        if model_config.get("supports_service_tier"):
            schema[
                vol.Optional(
                    CONF_SERVICE_TIER,
                    default=DEFAULT_SERVICE_TIER,
                )
            ] = SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(value=opt, label=opt.capitalize())
                        for opt in SERVICE_TIER_OPTIONS
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                )
            )

        # Add extra_body - passthrough for OpenAI-compatible backends
        # (ollama, llama.cpp, vLLM, LM Studio). Accepts a Jinja-templatable
        # JSON string; empty disables.
        schema[
            vol.Optional(
                CONF_EXTRA_BODY,
                default=DEFAULT_EXTRA_BODY,
            )
        ] = TemplateSelector()

        # Add shorten_tool_call_id option (for Mistral AI compatibility)
        schema[
            vol.Optional(
                CONF_SHORTEN_TOOL_CALL_ID,
                default=DEFAULT_SHORTEN_TOOL_CALL_ID,
            )
        ] = BooleanSelector()

        if (self._temp_data or {}).get(
            CONF_CONTEXT_TRUNCATE_STRATEGY, DEFAULT_CONTEXT_TRUNCATE_STRATEGY
        ) == "compact":
            schema[
                vol.Optional(
                    CONF_CONTEXT_COMPACTION_TARGET,
                    default=DEFAULT_CONTEXT_COMPACTION_TARGET,
                )
            ] = NumberSelector(
                NumberSelectorConfig(
                    min=1, max=100, step=1, unit_of_measurement="%"
                )
            )
            schema[
                vol.Optional(
                    CONF_CONTEXT_COMPACTION_RECENT_TURNS,
                    default=DEFAULT_CONTEXT_COMPACTION_RECENT_TURNS,
                )
            ] = NumberSelector(NumberSelectorConfig(min=0, step=1))
            schema[
                vol.Optional(
                    CONF_CONTEXT_COMPACTION_PROMPT,
                    default=DEFAULT_CONTEXT_COMPACTION_PROMPT,
                )
            ] = TextSelector(TextSelectorConfig(multiline=True))

        return self.async_show_form(
            step_id="advanced",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(schema), self.options
            ),
        )
