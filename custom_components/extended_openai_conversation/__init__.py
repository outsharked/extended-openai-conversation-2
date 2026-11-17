"""The OpenAI Conversation integration."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from functools import partial
import logging
from typing import Any, cast

from openai import AsyncClient
from openai._exceptions import AuthenticationError, OpenAIError

from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import CONF_API_KEY, CONF_TOKEN, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.typing import ConfigType

from .const import (
    CONF_API_PROVIDER,
    CONF_API_VERSION,
    CONF_BASE_URL,
    CONF_EMBEDDINGS,
    CONF_ORGANIZATION,
    CONF_RERANKER,
    CONF_RERANKER_MODEL,
    CONF_RERANKER_URL,
    CONF_SKIP_AUTHENTICATION,
    DEFAULT_API_PROVIDER,
    DEFAULT_SKIP_AUTHENTICATION,
    DOMAIN,
    SERVICE_TYPE_MCP,
)
from .context_management import EmbeddingMemoryStore, RerankerClient
from .helpers import get_authenticated_client, get_service_type
from .mcp_client import MCPClient, MCPServerConfig
from .mcp_oauth import async_get_mcp_oauth_auth
from .runtime import OpenAIServiceRuntime
from .services import async_setup_services
from .template import async_setup_templates, async_unload_templates

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.AI_TASK, Platform.CONVERSATION]
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)
_CONTEXT_MEMORY_CLEANUP_INTERVAL = timedelta(hours=1)

type ExtendedOpenAIConfigEntry = ConfigEntry[OpenAIServiceRuntime | MCPClient]


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
        try:
            auth = (
                await async_get_mcp_oauth_auth(hass, entry)
                if config.use_oauth and CONF_TOKEN in entry.data
                else None
            )
        except ConfigEntryAuthFailed, ConfigEntryNotReady:
            raise
        except (KeyError, TypeError, ValueError) as err:
            raise ConfigEntryAuthFailed(
                "Stored MCP OAuth configuration is invalid"
            ) from err
        entry.runtime_data = MCPClient(config, auth=auth)
        entry.async_on_unload(
            entry.add_update_listener(
                partial(
                    _mcp_update_listener,
                    previous_data=dict(entry.data),
                    previous_title=entry.title,
                )
            )
        )
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

    embedding_client: AsyncClient | None = None
    embedding_memory: EmbeddingMemoryStore | None = None
    reranker_client: RerankerClient | None = None
    embeddings = entry.data.get(CONF_EMBEDDINGS)
    if isinstance(embeddings, Mapping):
        embedding_client = await get_authenticated_client(
            hass=hass,
            api_key=embeddings[CONF_API_KEY],
            base_url=embeddings.get(CONF_BASE_URL),
            api_version=None,
            organization=None,
            api_provider=None,
            skip_authentication=True,
        )
        embedding_memory = EmbeddingMemoryStore(hass, entry.entry_id)
        try:
            await embedding_memory.async_cleanup_expired()
        except Exception:  # pylint: disable=broad-except
            _LOGGER.exception(
                "Unable to clean expired embedding memory; setup will continue"
            )

    reranker = entry.data.get(CONF_RERANKER)
    if isinstance(reranker, Mapping):
        reranker_client = RerankerClient(
            async_get_clientsession(hass),
            endpoint_url=reranker[CONF_RERANKER_URL],
            api_key=reranker[CONF_API_KEY],
            model=reranker[CONF_RERANKER_MODEL],
        )

    entry.runtime_data = OpenAIServiceRuntime(
        client=client,
        embedding_client=embedding_client,
        embedding_memory=embedding_memory,
        reranker_client=reranker_client,
    )
    runtime = entry.runtime_data
    context_memory = runtime.embedding_memory

    if context_memory is not None:

        async def _async_cleanup_context_memory(_: datetime) -> None:
            """Discard expired persistent and in-memory context data."""
            runtime.recall_snapshots.cleanup_expired()
            try:
                await context_memory.async_cleanup_expired()
            except Exception:  # pylint: disable=broad-except
                _LOGGER.exception("Unable to clean expired context memory")

        entry.async_on_unload(
            async_track_time_interval(
                hass,
                _async_cleanup_context_memory,
                _CONTEXT_MEMORY_CLEANUP_INTERVAL,
            )
        )

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


async def _mcp_update_listener(
    hass: HomeAssistant,
    entry: ConfigEntry,
    *,
    previous_data: Mapping[str, Any],
    previous_title: str,
) -> None:
    """Reload an MCP entry unless only its existing OAuth token changed."""
    previous_without_token = {
        key: value for key, value in previous_data.items() if key != CONF_TOKEN
    }
    current_without_token = {
        key: value for key, value in entry.data.items() if key != CONF_TOKEN
    }
    token_only_update = (
        CONF_TOKEN in previous_data
        and CONF_TOKEN in entry.data
        and previous_title == entry.title
        and previous_without_token == current_without_token
    )
    if token_only_update:
        return
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
