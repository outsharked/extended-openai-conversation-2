"""Tests for function entity access filtering."""

import probatio as vol
import pytest

from custom_components.extended_openai_conversation.exceptions import EntityAccessDenied
from custom_components.extended_openai_conversation.functions.entity_filter import (
    EntityFilter,
    validate_entity_filter_config,
)


def test_exclude_mode_denies_listed_entities() -> None:
    """Exclude mode denies only explicitly listed entities."""
    entity_filter = EntityFilter.from_config(
        {"mode": "exclude", "entities": ["lock.front_door"]}
    )

    assert entity_filter.is_allowed("light.living_room")
    assert not entity_filter.is_allowed("lock.front_door")


def test_include_mode_allows_only_listed_entities() -> None:
    """Include mode allows only explicitly listed entities."""
    entity_filter = EntityFilter.from_config(
        {"mode": "include", "entities": ["light.living_room"]}
    )

    assert entity_filter.is_allowed("light.living_room")
    assert not entity_filter.is_allowed("light.bedroom")


def test_deny_pattern_is_independent_of_include_mode() -> None:
    """The deny regex overrides an exact-list allow."""
    entity_filter = EntityFilter.from_config(
        {
            "mode": "include",
            "entities": ["light.protected_room"],
            "deny_pattern": ".*protected.*",
        }
    )

    with pytest.raises(EntityAccessDenied):
        entity_filter.validate(["light.protected_room"])


@pytest.mark.parametrize(
    "config",
    [
        {"mode": "invalid"},
        {"deny_pattern": "["},
        {"regex_timeout": 0},
        {"regex_timeout": True},
    ],
)
def test_invalid_entity_filter_config(config: dict[str, object]) -> None:
    """Invalid policy values are rejected during function validation."""
    with pytest.raises(vol.Invalid):
        validate_entity_filter_config(config)
