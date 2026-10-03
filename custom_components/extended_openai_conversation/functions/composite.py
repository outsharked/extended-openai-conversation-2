"""Composite tool for chaining multiple functions."""

from __future__ import annotations

from typing import Any

import probatio as vol

from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv, llm

from .base import Function


class CompositeFunction(Function):
    def __init__(self) -> None:
        """Initialize composite tool."""
        super().__init__(
            vol.Schema(
                {
                    vol.Required("sequence"): vol.All(
                        cv.ensure_list, [self.function_schema]
                    )
                }
            )
        )

    def function_schema(self, function_config: Any) -> dict[str, Any]:
        """Validate a composite function schema."""
        from . import get_function

        if not isinstance(function_config, dict):
            raise vol.Invalid("expected dictionary")

        composite_schema = {vol.Optional("response_variable"): str}
        function = get_function(function_config["type"])

        return dict(function.data_schema.extend(composite_schema)(function_config))

    async def execute(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> Any:
        from . import get_function

        sequence = function_config["sequence"]
        new_arguments = arguments.copy()

        for next_function_config in sequence:
            next_function = get_function(next_function_config["type"])
            result = await next_function.execute(
                hass, next_function_config, new_arguments, llm_context, exposed_entities
            )

            response_variable = next_function_config.get("response_variable")
            if response_variable:
                new_arguments[response_variable] = result

        return result
