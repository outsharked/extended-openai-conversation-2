"""Helper functions for Extended OpenAI Conversation component."""

from __future__ import annotations

from functools import partial
import logging
import re
from typing import Any, cast
from urllib.parse import urlparse

from openai import AsyncAzureOpenAI, AsyncClient, AsyncOpenAI, BadRequestError

from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.template import Template

from .const import (
    ALTERNATE_REQUEST_PARAMS,
    DEFAULT_MODEL_CONFIG,
    MODEL_CONFIG_PATTERNS,
    OPENAI_API_HOST,
    OPENAI_DEFAULT_MODEL_CONFIG,
    OPTIONAL_REQUEST_PARAMS,
    UNSUPPORTED_PARAM_ERROR_CODES,
)

_LOGGER = logging.getLogger(__name__)


AZURE_DOMAIN_PATTERN = r"\.(openai\.azure\.com|azure-api\.net|services\.ai\.azure\.com)"


def get_model_config(model: str, base_url: str | None = None) -> dict[str, bool]:
    """Get model-specific parameter configuration.

    A base_url of None means the default OpenAI endpoint.
    """
    # Check patterns in order; first match wins
    for entry in MODEL_CONFIG_PATTERNS:
        pattern = str(entry["pattern"])
        entry_config = entry["config"]
        if re.search(pattern, model, re.IGNORECASE):
            # Type assertion since we know the structure from MODEL_CONFIG_PATTERNS
            return (
                dict(entry_config)
                if isinstance(entry_config, dict)
                else DEFAULT_MODEL_CONFIG
            )

    if is_openai_url(base_url):
        return OPENAI_DEFAULT_MODEL_CONFIG

    # Default configuration for standard models (gpt-4, gpt-4o, etc.)
    return DEFAULT_MODEL_CONFIG


def get_exposed_entities(hass: HomeAssistant) -> list[dict[str, Any]]:
    """Get exposed entities."""
    states = [
        state
        for state in hass.states.async_all()
        if async_should_expose(hass, conversation.DOMAIN, state.entity_id)
    ]
    entity_registry = er.async_get(hass)
    exposed_entities = []
    for state in states:
        entity_id = state.entity_id
        entity = entity_registry.async_get(entity_id)

        aliases: list[str] = []
        if entity and entity.aliases:
            aliases = [str(a) for a in entity.aliases]

        exposed_entities.append(
            {
                "entity_id": entity_id,
                "name": state.name,
                "state": state.state,
                "aliases": aliases,
            }
        )
    return exposed_entities


def is_azure_url(base_url: str | None) -> bool:
    """Check if the base URL is an Azure OpenAI URL."""
    return bool(base_url and re.search(AZURE_DOMAIN_PATTERN, base_url))


def is_openai_url(base_url: str | None) -> bool:
    """Check if the base URL is OpenAI's own API (None means the default)."""
    return base_url is None or urlparse(base_url).hostname == OPENAI_API_HOST


def convert_to_template(
    settings: Any,
    hass: HomeAssistant,
    template_keys: list[str] | None = None,
) -> None:
    if template_keys is None:
        template_keys = ["data", "event_data", "target", "service"]
    _convert_to_template(settings, template_keys, hass, [])


def _convert_to_template(
    settings: Any,
    template_keys: list[str],
    hass: HomeAssistant,
    parents: list[str],
) -> None:
    if isinstance(settings, dict):
        for key, value in settings.items():
            if isinstance(value, str) and (
                key in template_keys or set(parents).intersection(template_keys)
            ):
                settings[key] = Template(value, hass)
            if isinstance(value, dict):
                parents.append(key)
                _convert_to_template(value, template_keys, hass, parents)
                parents.pop()
            if isinstance(value, list):
                parents.append(key)
                for item in value:
                    _convert_to_template(item, template_keys, hass, parents)
                parents.pop()
    if isinstance(settings, list):
        for setting in settings:
            _convert_to_template(setting, template_keys, hass, parents)


async def get_authenticated_client(
    hass: HomeAssistant,
    api_key: str,
    base_url: str | None,
    api_version: str | None,
    organization: str | None,
    api_provider: str | None,
    skip_authentication: bool = False,
) -> AsyncClient:
    """Validate OpenAI authentication."""

    client: AsyncClient
    if base_url and (is_azure_url(base_url) or api_provider == "azure"):
        client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=base_url,
            api_version=api_version,
            organization=organization,
            # Legacy HTTPX clients are supported at runtime only.
            http_client=cast(Any, get_async_client(hass)),
        )
    else:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            # Legacy HTTPX clients are supported at runtime only.
            http_client=cast(Any, get_async_client(hass)),
        )

    if skip_authentication:
        return client

    response = await hass.async_add_executor_job(
        partial(client.models.list, timeout=10)
    )

    async for _ in response:
        break
    return client


def _remove_unsupported_param(
    kwargs: dict[str, Any], param: str, unsupported_params: set[str]
) -> None:
    """Drop a rejected parameter, moving its value to the alternate if allowed."""
    value = kwargs.pop(param)
    alternate = ALTERNATE_REQUEST_PARAMS.get(param)
    if alternate and alternate not in unsupported_params and alternate not in kwargs:
        kwargs[alternate] = value


async def async_create_chat_completion(
    client: AsyncClient, unsupported_params: set[str], **kwargs: Any
) -> Any:
    """Create a chat completion, retrying without parameters the model rejects.

    Newer OpenAI models reject parameters older ones accepted (max_tokens,
    temperature, top_p, ...) with a 400 naming the parameter. Rather than
    tracking every model's quirks, drop (or swap) the named parameter and
    retry. Rejections are recorded in unsupported_params so callers can share
    them across requests and skip the failing round trip next time.
    """
    for known in unsupported_params:
        if known in kwargs:
            _remove_unsupported_param(kwargs, known, unsupported_params)

    while True:
        try:
            return await client.chat.completions.create(**kwargs)
        except BadRequestError as err:
            param = err.param
            if (
                err.code not in UNSUPPORTED_PARAM_ERROR_CODES
                or param not in OPTIONAL_REQUEST_PARAMS
                or param not in kwargs
            ):
                raise
            _LOGGER.warning(
                "Model %s does not support '%s', retrying without it: %s",
                kwargs.get("model"),
                param,
                err.message,
            )
            unsupported_params.add(param)
            _remove_unsupported_param(kwargs, param, unsupported_params)
