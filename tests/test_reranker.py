"""Tests for the standalone reranker client."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.extended_openai_conversation.context_management import (
    RerankerAuthenticationError,
    RerankerClient,
    RerankerConnectionError,
    RerankerResponseError,
    truncate_reranker_document,
)


def _session_response(payload, *, status: int = 200):
    """Return a mocked session and response context manager."""
    response = MagicMock(status=status)
    response.json = AsyncMock(return_value=payload)
    request_context = MagicMock()
    request_context.__aenter__ = AsyncMock(return_value=response)
    request_context.__aexit__ = AsyncMock(return_value=None)
    session = MagicMock()
    session.post.return_value = request_context
    return session


def test_truncate_reranker_document_uses_bounded_readable_excerpt():
    """Candidate truncation respects the estimate and a nearby boundary."""
    assert truncate_reranker_document("alpha beta gamma", 3) == "alpha beta"
    assert truncate_reranker_document("short", None) == "short"

    with pytest.raises(ValueError, match="must be positive"):
        truncate_reranker_document("text", 0)


async def test_reranker_maps_scores_to_original_documents_and_filters():
    """Results are validated, sorted, filtered, and mapped by index."""
    session = _session_response(
        {
            "results": [
                {"index": 0, "relevance_score": 0.2},
                {"index": 2, "relevance_score": 0.7},
                {"index": 1, "relevance_score": 0.9},
            ]
        }
    )
    client = RerankerClient(
        session,
        endpoint_url=" https://reranker.example.com/v1/rerank ",
        api_key="secret",
        model=" reranker-model ",
    )

    results = await client.async_rerank(
        "query",
        ["alpha beta gamma", "second", "third"],
        max_results=2,
        min_score=0.75,
        max_candidate_tokens=3,
    )

    assert [(result.index, result.score, result.document) for result in results] == [
        (1, 0.9, "second")
    ]
    session.post.assert_called_once_with(
        "https://reranker.example.com/v1/rerank",
        headers={"Authorization": "Bearer secret"},
        json={
            "model": "reranker-model",
            "query": "query",
            "documents": ["alpha beta", "second", "third"],
            "top_n": 2,
        },
    )


async def test_reranker_caps_request_to_available_candidates():
    """A caller ceiling above the candidate count does not inflate top_n."""
    session = _session_response({"results": [{"index": 0, "relevance_score": 1}]})
    client = RerankerClient(
        session,
        endpoint_url="https://reranker.example.com/rerank",
        api_key="secret",
        model="model",
    )

    results = await client.async_rerank("query", ["only"], max_results=20)

    assert results[0].score == 1.0
    assert session.post.call_args.kwargs["json"]["top_n"] == 1


async def test_reranker_skips_request_without_candidates():
    """An empty candidate set returns immediately."""
    session = MagicMock()
    client = RerankerClient(
        session,
        endpoint_url="https://reranker.example.com/rerank",
        api_key="secret",
        model="model",
    )

    assert await client.async_rerank("query", []) == []
    session.post.assert_not_called()


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (401, RerankerAuthenticationError),
        (403, RerankerAuthenticationError),
        (404, RerankerConnectionError),
        (500, RerankerConnectionError),
        (400, RerankerResponseError),
    ],
)
async def test_reranker_classifies_http_failures(status, error_type):
    """HTTP failures retain their authentication/connection distinction."""
    client = RerankerClient(
        _session_response({}, status=status),
        endpoint_url="https://reranker.example.com/rerank",
        api_key="secret",
        model="model",
    )

    with pytest.raises(error_type):
        await client.async_rerank("query", ["candidate"])


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"results": []},
        {"results": ["invalid"]},
        {"results": [{"index": 2, "relevance_score": 0.5}]},
        {"results": [{"index": 0, "relevance_score": "high"}]},
        {
            "results": [
                {"index": 0, "relevance_score": 0.5},
                {"index": 0, "relevance_score": 0.4},
            ]
        },
    ],
)
async def test_reranker_rejects_malformed_results(payload):
    """Bad indices, scores, duplicates, and result shapes are rejected."""
    client = RerankerClient(
        _session_response(payload),
        endpoint_url="https://reranker.example.com/rerank",
        api_key="secret",
        model="model",
    )

    with pytest.raises(RerankerResponseError):
        await client.async_rerank("query", ["candidate"])


async def test_reranker_rejects_invalid_runtime_limits():
    """Runtime callers cannot submit ambiguous result or score limits."""
    client = RerankerClient(
        MagicMock(),
        endpoint_url="https://reranker.example.com/rerank",
        api_key="secret",
        model="model",
    )

    with pytest.raises(ValueError, match="max_results"):
        await client.async_rerank("query", ["candidate"], max_results=0)
    with pytest.raises(ValueError, match="min_score"):
        await client.async_rerank("query", ["candidate"], min_score=float("nan"))


@pytest.mark.parametrize(
    ("endpoint_url", "model", "message"),
    [
        ("not-a-url", "model", "endpoint URL"),
        ("https://reranker.example.com/rerank", "  ", "model"),
    ],
)
def test_reranker_rejects_invalid_configuration(endpoint_url, model, message):
    """The standalone client rejects unusable endpoint configuration."""
    with pytest.raises(ValueError, match=message):
        RerankerClient(
            MagicMock(),
            endpoint_url=endpoint_url,
            api_key="secret",
            model=model,
        )
