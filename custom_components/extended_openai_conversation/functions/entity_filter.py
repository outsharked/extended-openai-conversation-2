"""Entity access filtering for function tools."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import probatio as vol
import regex

from homeassistant.helpers import config_validation as cv

from ..const import (
    CONF_ENTITY_FILTER_DENY_PATTERN,
    CONF_ENTITY_FILTER_ENTITIES,
    CONF_ENTITY_FILTER_MODE,
    CONF_ENTITY_FILTER_REGEX_TIMEOUT,
    DEFAULT_ENTITY_FILTER_REGEX_TIMEOUT,
    ENTITY_FILTER_MODE_EXCLUDE,
    ENTITY_FILTER_MODE_INCLUDE,
)
from ..exceptions import EntityAccessDenied


def _positive_float(value: Any) -> float:
    """Validate a positive floating-point value."""
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise vol.Invalid("value must be a positive number")
    return float(value)


def _regex_pattern(value: Any) -> str:
    """Validate a regular expression pattern."""
    if not isinstance(value, str):
        raise vol.Invalid("value must be a string")
    if value:
        try:
            regex.compile(value)
        except regex.error as err:
            raise vol.Invalid("invalid regular expression") from err
    return value


ENTITY_FILTER_SCHEMA = vol.Schema(
    {
        vol.Optional(
            CONF_ENTITY_FILTER_MODE, default=ENTITY_FILTER_MODE_EXCLUDE
        ): vol.In((ENTITY_FILTER_MODE_EXCLUDE, ENTITY_FILTER_MODE_INCLUDE)),
        vol.Optional(CONF_ENTITY_FILTER_ENTITIES, default=[]): vol.All(
            cv.ensure_list, [cv.entity_id]
        ),
        vol.Optional(CONF_ENTITY_FILTER_DENY_PATTERN, default=""): _regex_pattern,
        vol.Optional(
            CONF_ENTITY_FILTER_REGEX_TIMEOUT,
            default=DEFAULT_ENTITY_FILTER_REGEX_TIMEOUT,
        ): _positive_float,
    }
)


def validate_entity_filter_config(value: Any) -> dict[str, Any]:
    """Validate an entity filter configuration."""
    if not isinstance(value, dict):
        raise vol.Invalid("entity_filter must be a mapping")
    return dict(ENTITY_FILTER_SCHEMA(value))


@dataclass(frozen=True, slots=True)
class EntityFilter:
    """Apply exact-list and regular-expression entity access rules."""

    mode: str
    entities: frozenset[str]
    deny_pattern: str
    regex_timeout: float

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> EntityFilter:
        """Create an entity filter from validated configuration."""
        validated = validate_entity_filter_config(config)
        return cls(
            mode=validated[CONF_ENTITY_FILTER_MODE],
            entities=frozenset(validated[CONF_ENTITY_FILTER_ENTITIES]),
            deny_pattern=validated[CONF_ENTITY_FILTER_DENY_PATTERN],
            regex_timeout=validated[CONF_ENTITY_FILTER_REGEX_TIMEOUT],
        )

    def is_allowed(self, entity_id: str) -> bool:
        """Return whether an entity is allowed by the policy."""
        if self.deny_pattern:
            try:
                if regex.fullmatch(self.deny_pattern,entity_id,timeout=self.regex_timeout):
                    return False
            except TimeoutError:
                return False

        if self.mode == ENTITY_FILTER_MODE_INCLUDE:
            return entity_id in self.entities
        return entity_id not in self.entities

    def validate(self, entity_ids: Collection[str]) -> None:
        """Raise when any entity is denied by the policy."""
        denied = sorted(entity_id for entity_id in entity_ids if not self.is_allowed(entity_id))
        if denied:
            raise EntityAccessDenied(denied)
