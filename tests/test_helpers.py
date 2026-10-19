"""Tests for model parameter selection and unsupported-parameter fallback."""

import json

import httpx2
from openai import AsyncOpenAI, BadRequestError
import pytest

from custom_components.extended_openai_conversation.helpers import (
    async_create_chat_completion,
    get_model_config,
)

OPENAI_URL = "https://api.openai.com/v1"
OLLAMA_URL = "http://localhost:11434/v1"
AZURE_URL = "https://example.openai.azure.com"

# Verbatim error body OpenAI returned for gpt-6.1-sol (issue #22)
MAX_TOKENS_REJECTED = {
    "error": {
        "message": "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",
        "type": "invalid_request_error",
        "param": "max_tokens",
        "code": "unsupported_parameter",
    }
}

COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-6.1-sol",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
}


def _rejection(param: str, code: str = "unsupported_parameter") -> dict:
    return {
        "error": {
            "message": f"Unsupported: '{param}'",
            "type": "invalid_request_error",
            "param": param,
            "code": code,
        }
    }


class FakeOpenAI:
    """Fake OpenAI server that rejects a fixed set of request parameters."""

    def __init__(self, rejected: dict[str, dict]) -> None:
        self.rejected = rejected
        self.requests: list[dict] = []

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        for param, error in self.rejected.items():
            if param in body:
                return httpx2.Response(400, json=error)
        return httpx2.Response(200, json=COMPLETION)

    def client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key="test",
            base_url=OPENAI_URL,
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self.handle)),
        )


def _kwargs(**params) -> dict:
    return {
        "model": "gpt-6.1-sol",
        "messages": [{"role": "user", "content": "hi"}],
        **params,
    }


@pytest.mark.parametrize(
    "model",
    [
        "gpt-6.1-sol",
        "gpt-5",
        "gpt-5.2-mini",
        "gpt-10",
        "o1",
        "o3-mini",
        "o5",
        "openai/gpt-6",
    ],
)
def test_reasoning_models_use_max_completion_tokens(model):
    config = get_model_config(model, OLLAMA_URL)

    assert config["supports_max_completion_tokens"]
    assert not config["supports_max_tokens"]
    assert config["supports_reasoning_effort"]
    assert not config["supports_temperature"]


@pytest.mark.parametrize(
    "model", ["gpt-4o", "gpt-4.1-mini", "gpt-3.5-turbo", "chatgpt-4o-latest"]
)
def test_older_models_are_not_reasoning_models(model):
    assert not get_model_config(model, OLLAMA_URL)["supports_reasoning_effort"]


@pytest.mark.parametrize("base_url", [None, OPENAI_URL])
def test_unknown_model_on_openai_uses_max_completion_tokens(base_url):
    config = get_model_config("some-future-model", base_url)

    assert config["supports_max_completion_tokens"]
    assert not config["supports_max_tokens"]
    assert config["supports_temperature"]


@pytest.mark.parametrize("base_url", [OLLAMA_URL, AZURE_URL])
def test_unknown_model_elsewhere_uses_max_tokens(base_url):
    config = get_model_config("qwen3:8b", base_url)

    assert config["supports_max_tokens"]
    assert not config["supports_max_completion_tokens"]


async def test_rejected_max_tokens_retries_with_max_completion_tokens():
    server = FakeOpenAI({"max_tokens": MAX_TOKENS_REJECTED})
    unsupported: set[str] = set()

    response = await async_create_chat_completion(
        server.client(), unsupported, **_kwargs(max_tokens=500)
    )

    assert response.choices[0].message.content == "ok"
    assert len(server.requests) == 2
    assert "max_tokens" not in server.requests[1]
    assert server.requests[1]["max_completion_tokens"] == 500
    assert unsupported == {"max_tokens"}


async def test_remembered_rejection_is_applied_before_first_request():
    server = FakeOpenAI({"max_tokens": MAX_TOKENS_REJECTED})

    await async_create_chat_completion(
        server.client(), {"max_tokens"}, **_kwargs(max_tokens=500)
    )

    assert len(server.requests) == 1
    assert server.requests[0]["max_completion_tokens"] == 500


async def test_unsupported_sampling_value_is_dropped():
    server = FakeOpenAI({"temperature": _rejection("temperature", "unsupported_value")})
    unsupported: set[str] = set()

    await async_create_chat_completion(
        server.client(), unsupported, **_kwargs(temperature=0.5, top_p=1)
    )

    assert "temperature" not in server.requests[-1]
    assert server.requests[-1]["top_p"] == 1
    assert unsupported == {"temperature"}


async def test_both_token_params_rejected_gives_up_without_looping():
    server = FakeOpenAI(
        {
            "max_tokens": MAX_TOKENS_REJECTED,
            "max_completion_tokens": _rejection("max_completion_tokens"),
        }
    )

    await async_create_chat_completion(
        server.client(), set(), **_kwargs(max_tokens=500)
    )

    assert len(server.requests) == 3
    assert "max_tokens" not in server.requests[-1]
    assert "max_completion_tokens" not in server.requests[-1]


async def test_other_bad_requests_are_raised_without_retry():
    server = FakeOpenAI({"messages": _rejection("messages", "invalid_value")})

    with pytest.raises(BadRequestError):
        await async_create_chat_completion(server.client(), set(), **_kwargs())

    assert len(server.requests) == 1


async def test_rejected_param_outside_allowlist_is_not_dropped():
    server = FakeOpenAI({"tools": _rejection("tools")})

    with pytest.raises(BadRequestError):
        await async_create_chat_completion(
            server.client(),
            set(),
            **_kwargs(tools=[{"type": "function", "function": {"name": "f"}}]),
        )

    assert len(server.requests) == 1
