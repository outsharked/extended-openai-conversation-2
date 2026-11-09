"""The OpenAI Conversation integration."""

from __future__ import annotations

import logging
from typing import cast

from openai import AsyncClient
from openai._exceptions import AuthenticationError, OpenAIError

from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import CONF_API_KEY, CONF_TOKEN, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.typing import ConfigType

from .const import (
    CONF_API_PROVIDER,
    CONF_API_VERSION,
    CONF_BASE_URL,
    CONF_ORGANIZATION,
    CONF_SKIP_AUTHENTICATION,
    DEFAULT_API_PROVIDER,
    DEFAULT_SKIP_AUTHENTICATION,
    DOMAIN,
    SERVICE_TYPE_MCP,
)
from .helpers import get_authenticated_client, get_service_type
from .mcp_client import MCPClient, MCPServerConfig
from .mcp_oauth import async_get_mcp_oauth_auth
from .services import async_setup_services
from .template import async_setup_templates, async_unload_templates

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.AI_TASK, Platform.CONVERSATION]
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

type ExtendedOpenAIConfigEntry = ConfigEntry[AsyncClient | MCPClient]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up OpenAI Conversation."""
    await async_migrate_integration(hass)
    await async_setup_services(hass, config)
    return True


async def async_setup_entry(
    hass: HomeAssistant, entry: ExtendedOpenAIConfigEntry
) -> bool:
    """Set up OpenAI Conversation from a config entry."""

    if get_service_type(entry) == SERVICE_TYPE_MCP:
        config = MCPServerConfig.from_dict({**entry.data, "name": entry.title})
        auth = (
            await async_get_mcp_oauth_auth(hass, entry)
            if config.use_oauth and CONF_TOKEN in entry.data
            else None
        )
        entry.runtime_data = MCPClient(config, auth=auth)
        entry.async_on_unload(entry.add_update_listener(update_listener))
        return True

    try:
        client = await get_authenticated_client(
            hass=hass,
            api_key=entry.data[CONF_API_KEY],
            base_url=entry.data.get(CONF_BASE_URL),
            api_version=entry.data.get(CONF_API_VERSION),
            organization=entry.data.get(CONF_ORGANIZATION),
            skip_authentication=entry.data.get(
                CONF_SKIP_AUTHENTICATION, DEFAULT_SKIP_AUTHENTICATION
            ),
            api_provider=entry.data.get(CONF_API_PROVIDER, DEFAULT_API_PROVIDER),
        )
    except AuthenticationError as err:
        _LOGGER.error("Invalid API key: %s", err)
        return False
    except OpenAIError as err:
        raise ConfigEntryNotReady(err) from err

    entry.runtime_data = client

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(update_listener))

    await async_setup_templates(hass)
    return True


async def async_unload_entry(
    hass: HomeAssistant, entry: ExtendedOpenAIConfigEntry
) -> bool:
    """Unload OpenAI."""
    if get_service_type(entry) == SERVICE_TYPE_MCP:
        await cast(MCPClient, entry.runtime_data).aclose()
        return True
    await async_unload_templates(hass)
    return bool(await hass.config_entries.async_unload_platforms(entry, PLATFORMS))


async def update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_migrate_integration(hass: HomeAssistant) -> None:
    """Migrate integration entry structure."""

    # Make sure we get enabled config entries first
    entries = sorted(
        (
            entry
            for entry in hass.config_entries.async_entries(DOMAIN)
            if get_service_type(entry) != SERVICE_TYPE_MCP
        ),
        key=lambda e: e.disabled_by is not None,
    )
    if not any(entry.version == 1 for entry in entries):
        return

    for entry in entries:
        _LOGGER.warning(
            "Migrating Extended OpenAI Conversation config entry %s from version %s to version 2",
            entry.entry_id,
            entry.version,
        )
        subentry = ConfigSubentry(
            data=entry.options,
            subentry_type="conversation",
            title=entry.title,
            unique_id=None,
        )
        hass.config_entries.async_add_subentry(entry, subentry)
        hass.config_entries.async_update_entry(
            entry, title=entry.title, options={}, version=2
        )
