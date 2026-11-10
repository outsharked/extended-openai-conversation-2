"""Template tool for Jinja2 rendering."""

from __future__ import annotations

from typing import Any

import probatio as vol

from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv, llm

from ..const import CONF_ENTITY_FILTER
from .base import Function
from .entity_filter import EntityFilter, validate_entity_filter_config


class TemplateFunction(Function):
    def __init__(self) -> None:
        """Initialize template tool."""
        super().__init__(
            vol.Schema(
                {
                    vol.Required("value_template"): cv.template,
                    vol.Optional("parse_result"): bool,
                    vol.Optional(CONF_ENTITY_FILTER): validate_entity_filter_config,
                }
            )
        )

    async def execute(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> Any:
        entity_filter_config = function_config.get(CONF_ENTITY_FILTER)
        entity_ids = arguments.get("entity_id")
        if entity_filter_config is not None and entity_ids is not None:
            if isinstance(entity_ids, str):
                entity_ids = [
                    entity_id.strip()
                    for entity_id in entity_ids.split(",")
                    if entity_id.strip()
                ]
            if isinstance(entity_ids, list) and all(
                isinstance(entity_id, str) for entity_id in entity_ids
            ):
                EntityFilter.from_config(entity_filter_config).validate(entity_ids)

        return function_config["value_template"].async_render(
            arguments,
            parse_result=function_config.get("parse_result", False),
        )
