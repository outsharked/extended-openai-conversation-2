"""Base classes for functions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import probatio as vol

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from ..exceptions import EntityNotExposed, EntityNotFound, InvalidFunction


class Function(ABC):
    def __init__(self, data_schema: vol.Schema = vol.Schema({})) -> None:
        """Initialize tool."""
        self.data_schema = data_schema.extend({vol.Required("type"): str})

    def validate_schema(self, function_config: dict[str, Any]) -> dict[str, Any]:
        """Validate and convert function configuration using the schema."""
        try:
            result = self.data_schema(function_config)
            return dict(result) if isinstance(result, dict) else {}
        except vol.error.Error as e:
            from . import FUNCTIONS

            function_type = next(
                (key for key, value in FUNCTIONS.items() if value == self),
                "",
            )
            raise InvalidFunction(function_type) from e

    def validate_entity_ids(
        self,
        hass: HomeAssistant,
        entity_ids: list[str],
        exposed_entities: list[dict[str, Any]],
    ) -> None:
        not_found = [
            entity_id for entity_id in entity_ids if hass.states.get(entity_id) is None
        ]
        if not_found:
            raise EntityNotFound(", ".join(not_found))
        exposed_entity_ids = {e["entity_id"] for e in exposed_entities}
        not_exposed = [
            entity_id for entity_id in entity_ids if entity_id not in exposed_entity_ids
        ]
        if not_exposed:
            raise EntityNotExposed(", ".join(not_exposed))

    @abstractmethod
    async def execute(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> Any:
        """Execute function."""
