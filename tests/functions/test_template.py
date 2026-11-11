"""Tests for TemplateFunction using yaml definitions."""

import pytest

from custom_components.extended_openai_conversation.exceptions import EntityAccessDenied
from custom_components.extended_openai_conversation.functions import TemplateFunction
from homeassistant.core import State
from homeassistant.exceptions import HomeAssistantError
from tests.helpers import prepare_function_tool_from_yaml


class TestTemplateFunctionYaml:
    """Test TemplateFunction using yaml definitions."""

    @pytest.fixture
    def function(self):
        """Create TemplateFunction instance."""
        return TemplateFunction()

    async def test_execute_template_from_yaml(
        self, hass, function, exposed_entities, llm_context
    ):
        """Test template execution from yaml definition."""
        # Load function from yaml
        function_tool = prepare_function_tool_from_yaml("template_example.yaml")
        function_config = function_tool["function"]

        # Mock sensor states for energy monitoring
        def mock_states_get(entity_id):
            states_map = {
                "sensor.house_power": State("sensor.house_power", "1500"),
                "sensor.daily_energy": State("sensor.daily_energy", "12.5"),
            }
            return states_map.get(entity_id)

        hass.states.get = mock_states_get

        # Arguments based on yaml spec parameters
        arguments = {"show_cost": True}

        result = await function.execute(
            hass, function_config, arguments, llm_context, exposed_entities
        )

        # Should return formatted energy summary
        assert "1500" in result or "1500.0" in result  # Power consumption
        assert "12.5" in result  # Daily energy
        assert "$" in result or "1.5" in result  # Cost (12.5 * 0.12 = 1.5)

    async def test_template_without_cost(
        self, hass, function, exposed_entities, llm_context
    ):
        """Test template without cost calculation."""
        function_tool = prepare_function_tool_from_yaml("template_example.yaml")
        function_config = function_tool["function"]

        # Mock sensor states
        def mock_states_get(entity_id):
            states_map = {
                "sensor.house_power": State("sensor.house_power", "800"),
                "sensor.daily_energy": State("sensor.daily_energy", "5.0"),
            }
            return states_map.get(entity_id)

        hass.states.get = mock_states_get

        arguments = {"show_cost": False}
        result = await function.execute(
            hass, function_config, arguments, llm_context, exposed_entities
        )

        # Should not include cost when show_cost is False
        assert "800" in result or "800.0" in result
        assert "5.0" in result or "5" in result

    async def test_entity_filter_denies_get_attributes_target(
        self, hass, function, exposed_entities, llm_context
    ):
        """A filtered get_attributes entity is not rendered."""
        function_config = function.validate_schema(
            {
                "type": "template",
                "value_template": "{{ entity_id | join(',') }}",
                "entity_filter": {
                    "mode": "exclude",
                    "entities": ["sensor.temperature"],
                },
            }
        )

        with pytest.raises(EntityAccessDenied):
            await function.execute(
                hass,
                function_config,
                {"entity_id": ["sensor.temperature"]},
                llm_context,
                exposed_entities,
            )

    async def test_entity_filter_allows_get_attributes_target(
        self, hass, function, exposed_entities, llm_context
    ):
        """An entity selected in include mode can be rendered."""
        function_config = function.validate_schema(
            {
                "type": "template",
                "value_template": "{{ entity_id | join(',') }}",
                "entity_filter": {
                    "mode": "include",
                    "entities": ["sensor.temperature"],
                },
            }
        )

        result = await function.execute(
            hass,
            function_config,
            {"entity_id": ["sensor.temperature"]},
            llm_context,
            exposed_entities,
        )

        assert result == "sensor.temperature"

    async def test_entity_filter_accepts_comma_separated_entity_string(
        self, hass, function, exposed_entities, llm_context
    ):
        """A valid comma-separated string is normalized for policy validation."""
        function_config = function.validate_schema(
            {
                "type": "template",
                "value_template": "{{ entity_id }}",
                "entity_filter": {
                    "mode": "include",
                    "entities": ["sensor.temperature", "sensor.humidity"],
                },
            }
        )

        result = await function.execute(
            hass,
            function_config,
            {"entity_id": "sensor.temperature, sensor.humidity"},
            llm_context,
            exposed_entities,
        )

        assert result == "sensor.temperature, sensor.humidity"

    @pytest.mark.parametrize(
        "entity_ids",
        [
            {"sensor.private_room": True},
            ["sensor.temperature", 123],
        ],
    )
    async def test_entity_filter_rejects_malformed_entity_ids(
        self, hass, function, exposed_entities, llm_context, entity_ids
    ):
        """Malformed entity arguments cannot bypass policy validation."""
        function_config = function.validate_schema(
            {
                "type": "template",
                "value_template": "{{ entity_id }}",
                "entity_filter": {
                    "mode": "exclude",
                    "entities": ["sensor.private_room"],
                },
            }
        )

        with pytest.raises(
            HomeAssistantError,
            match="entity_id must be a string or a list of strings",
        ):
            await function.execute(
                hass,
                function_config,
                {"entity_id": entity_ids},
                llm_context,
                exposed_entities,
            )
