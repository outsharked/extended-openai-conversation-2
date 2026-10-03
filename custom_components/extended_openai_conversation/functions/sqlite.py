"""SQLite tool for database queries."""

from __future__ import annotations

import logging
import os
import sqlite3
from typing import Any
from urllib import parse

import probatio as vol

from homeassistant.components import recorder
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.template import Template

from .base import Function

_LOGGER = logging.getLogger(__name__)


class SqliteFunction(Function):
    def __init__(self) -> None:
        """Initialize sqlite tool."""
        super().__init__(
            vol.Schema(
                {
                    vol.Optional("query"): str,
                    vol.Optional("db_url"): str,
                    vol.Optional("single"): bool,
                }
            )
        )

    def is_exposed(
        self, entity_id: str, exposed_entities: list[dict[str, Any]]
    ) -> bool:
        return any(
            exposed_entity["entity_id"] == entity_id
            for exposed_entity in exposed_entities
        )

    def is_exposed_entity_in_query(
        self, query: str, exposed_entities: list[dict[str, Any]]
    ) -> bool:
        exposed_entity_ids = list(
            map(lambda e: f"'{e['entity_id']}'", exposed_entities)
        )
        return any(
            exposed_entity_id in query for exposed_entity_id in exposed_entity_ids
        )

    def raise_error(self, msg: str = "Unexpected error occurred.") -> None:
        raise HomeAssistantError(msg)

    def get_default_db_url(self, hass: HomeAssistant) -> str:
        db_file_path = os.path.join(hass.config.config_dir, recorder.DEFAULT_DB_FILE)
        return f"file:{db_file_path}?mode=ro"

    def set_url_read_only(self, url: str) -> str:
        scheme, netloc, path, query_string, fragment = parse.urlsplit(url)
        query_params = parse.parse_qs(query_string)

        query_params["mode"] = ["ro"]
        new_query_string = parse.urlencode(query_params, doseq=True)

        return parse.urlunsplit((scheme, netloc, path, new_query_string, fragment))

    async def execute(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> dict[str, Any] | list[dict[str, Any]]:
        db_url = self.set_url_read_only(
            function_config.get("db_url", self.get_default_db_url(hass))
        )
        query = function_config.get("query", "{{query}}")

        template_arguments = {
            "is_exposed": lambda e: self.is_exposed(e, exposed_entities),
            "is_exposed_entity_in_query": lambda q: self.is_exposed_entity_in_query(
                q, exposed_entities
            ),
            "exposed_entities": exposed_entities,
            "raise": self.raise_error,
        }
        template_arguments.update(arguments)

        q = Template(query, hass).async_render(template_arguments)
        _LOGGER.info("Rendered query: %s", q)

        with sqlite3.connect(db_url, uri=True) as conn:
            cursor = conn.cursor().execute(q)
            names = [description[0] for description in cursor.description]

            if function_config.get("single") is True:
                row = cursor.fetchone()
                return {name: val for name, val in zip(names, row, strict=False)}

            rows = cursor.fetchall()
            result = []
            for row in rows:
                result.append(
                    {name: val for name, val in zip(names, row, strict=False)}
                )
            return result
