"""Tests for the reserved final_response function argument."""

from collections.abc import AsyncIterator
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.extended_openai_conversation.entity import (
    FINAL_RESPONSE_ARG,
    ExtendedOpenAIBaseLLMEntity,
    _get_final_response,
    _result_has_error,
)
from homeassistant.components import conversation
from homeassistant.helpers import llm

ENTITY_PATH = "custom_components.extended_openai_conversation.entity"

FUNCTION_TOOL = {
    "spec": {
        "name": "execute_services",
        "parameters": {"type": "object", "properties": {}},
    },
    "function": {"type": "native", "name": "execute_service"},
}


def _tool_input(args: object) -> llm.ToolInput:
    return llm.ToolInput(
        id="call_1", tool_name="execute_services", tool_args=args, external=True
    )


def test_get_final_response_returns_stripped_text():
    assert _get_final_response(_tool_input({FINAL_RESPONSE_ARG: " Fatto. "})) == (
        "Fatto."
    )


@pytest.mark.parametrize("value", [None, "", "   ", 42, ["Fatto"]])
def test_get_final_response_ignores_missing_or_invalid(value):
    args = {"list": []}
    if value is not None:
        args[FINAL_RESPONSE_ARG] = value
    assert _get_final_response(_tool_input(args)) is None


def test_get_final_response_ignores_non_dict_arguments():
    assert _get_final_response(_tool_input(["not", "a", "dict"])) is None


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ([{"success": True}, {"success": True}], False),
        ([{"success": True}, {"error": "unavailable"}], True),
        ({"error": "boom"}, True),
        ({"success": True}, False),
        ("Scheduled", False),
        ([[{"error": "nested"}]], True),
    ],
)
def test_result_has_error(result, expected):
    assert _result_has_error(result) is expected


def _chunk(
    *,
    content: str | None = None,
    tool_calls: list[SimpleNamespace] | None = None,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                delta=SimpleNamespace(content=content, tool_calls=tool_calls),
            )
        ],
        usage=None,
    )


def _tool_call_chunks(arguments: dict) -> list[SimpleNamespace]:
    delta = SimpleNamespace(
        index=0,
        id="call_1",
        function=SimpleNamespace(
            name="execute_services", arguments=json.dumps(arguments)
        ),
        extra_content=None,
        model_extra=None,
    )
    return [
        _chunk(tool_calls=[delta]),
        _chunk(finish_reason="tool_calls"),
    ]


def _text_chunks(text: str) -> list[SimpleNamespace]:
    return [_chunk(content=text), _chunk(finish_reason="stop")]


async def _stream(chunks: list[SimpleNamespace]) -> AsyncIterator[SimpleNamespace]:
    for chunk in chunks:
        yield chunk


def _make_entity() -> ExtendedOpenAIBaseLLMEntity:
    entity = ExtendedOpenAIBaseLLMEntity.__new__(ExtendedOpenAIBaseLLMEntity)
    entity.hass = MagicMock()
    entity.hass.async_add_executor_job = AsyncMock(
        side_effect=lambda func, *args: func(*args)
    )
    entity.entity_id = "conversation.test"
    entity.subentry = MagicMock()
    entity.subentry.data = {}
    entity.entry = MagicMock()
    entity.entry.data = {}
    entity.entry.runtime_data = MagicMock()
    entity._unsupported_params = {}
    return entity


def _make_chat_log() -> conversation.ChatLog:
    chat_log = conversation.ChatLog(MagicMock(), "conversation_id")
    chat_log.async_add_user_content(
        conversation.UserContent(content="spegni la luce della cucina")
    )
    return chat_log


async def _run(
    responses: list[list[SimpleNamespace]], function_result: object
) -> tuple[conversation.ChatLog, AsyncMock, AsyncMock]:
    """Run the chat loop with scripted LLM responses and a fake function."""
    entity = _make_entity()
    chat_log = _make_chat_log()
    create = AsyncMock(side_effect=[_stream(chunks) for chunks in responses])
    function = MagicMock()
    function.execute = AsyncMock(return_value=function_result)
    with (
        patch(f"{ENTITY_PATH}.async_create_chat_completion", create),
        patch(f"{ENTITY_PATH}.get_function", return_value=function),
    ):
        await entity._async_handle_chat_log(
            chat_log, function_tools=[FUNCTION_TOOL], exposed_entities=[]
        )
    return chat_log, create, function.execute


async def test_final_response_skips_follow_up_request():
    args = {
        "list": [{"domain": "light", "service": "turn_off"}],
        FINAL_RESPONSE_ARG: "Luce della cucina spenta.",
    }
    chat_log, create, execute = await _run(
        [_tool_call_chunks(args)], [{"success": True}]
    )

    assert create.await_count == 1
    last = chat_log.content[-1]
    assert isinstance(last, conversation.AssistantContent)
    assert last.content == "Luce della cucina spenta."
    # The reserved argument never reaches the function
    executed_args = execute.await_args.args[2]
    assert FINAL_RESPONSE_ARG not in executed_args
    assert executed_args["list"] == args["list"]


async def test_failed_call_falls_back_to_follow_up_request():
    args = {"list": [], FINAL_RESPONSE_ARG: "Luce della cucina spenta."}
    chat_log, create, _ = await _run(
        [_tool_call_chunks(args), _text_chunks("Non ci sono riuscito.")],
        [{"error": "light.cucina is unavailable"}],
    )

    assert create.await_count == 2
    assert chat_log.content[-1].content == "Non ci sono riuscito."


async def test_without_final_response_keeps_follow_up_request():
    chat_log, create, _ = await _run(
        [_tool_call_chunks({"list": []}), _text_chunks("Fatto.")],
        [{"success": True}],
    )

    assert create.await_count == 2
    assert chat_log.content[-1].content == "Fatto."


async def test_final_response_keeps_chat_history_valid():
    args = {"list": [], FINAL_RESPONSE_ARG: "Fatto."}
    chat_log, _, _ = await _run([_tool_call_chunks(args)], [{"success": True}])

    roles = [content.role for content in chat_log.content]
    assert roles == ["system", "user", "assistant", "tool_result", "assistant"]
