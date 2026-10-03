"""Template tool for Jinja2 rendering."""

from __future__ import annotations

from typing import Any

import probatio as vol

from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv, llm

from .base import Function


class TemplateFunction(Function):
    def __init__(self) -> None:
        """Initialize template tool."""
        super().__init__(
            vol.Schema(
                {
                    vol.Required("value_template"): cv.template,
                    vol.Optional("parse_result"): bool,
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
        return function_config["value_template"].async_render(
            arguments,
            parse_result=function_config.get("parse_result", False),
        )
