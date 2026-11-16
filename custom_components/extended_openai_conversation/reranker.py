"""Client for OpenAI-compatible reranker endpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any
from urllib.parse import urlparse

from aiohttp import ClientError, ClientSession

from homeassistant.exceptions import HomeAssistantError

_ESTIMATED_CHARACTERS_PER_TOKEN = 4


class RerankerError(HomeAssistantError):
    """Base error raised by the reranker client."""


class RerankerAuthenticationError(RerankerError):
    """Raised when a reranker endpoint rejects its credentials."""


class RerankerConnectionError(RerankerError):
    """Raised when a reranker endpoint cannot be reached."""


class RerankerResponseError(RerankerError):
    """Raised when a reranker endpoint returns an invalid response."""


@dataclass(frozen=True, slots=True)
class RerankerResult:
    """One reranked candidate mapped back to its original document."""

    index: int
    score: float
    document: str


def truncate_reranker_document(document: str, max_estimated_tokens: int | None) -> str:
    """Bound one reranker candidate using the integration's token estimate."""
    if max_estimated_tokens is None:
        return document
    if max_estimated_tokens < 1:
        raise ValueError("max_estimated_tokens must be positive")

    max_characters = max_estimated_tokens * _ESTIMATED_CHARACTERS_PER_TOKEN
    if len(document) <= max_characters:
        return document

    truncated = document[:max_characters]
    split_at = max(
        truncated.rfind(separator) for separator in ("\n", "\t", " ")
    )
    if split_at >= max_characters // 2:
        truncated = truncated[:split_at]
    return truncated.rstrip()


class RerankerClient:
    """Call and validate a configured reranker endpoint."""

    def __init__(
        self,
        session: ClientSession,
        *,
        endpoint_url: str,
        api_key: str,
        model: str,
    ) -> None:
        """Initialize the client from validated endpoint configuration."""
        endpoint_url = endpoint_url.strip()
        parsed_url = urlparse(endpoint_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("A valid reranker endpoint URL is required")
        model = model.strip()
        if not model:
            raise ValueError("A reranker model is required")

        self.endpoint_url = endpoint_url
        self.model = model
        self._session = session
        self._headers = {"Authorization": f"Bearer {api_key}"}

    async def async_rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        max_results: int | None = None,
        min_score: float | None = None,
        max_candidate_tokens: int | None = None,
    ) -> list[RerankerResult]:
        """Rerank candidate documents and return validated scored results."""
        original_documents = list(documents)
        if not original_documents:
            return []

        result_limit = len(original_documents) if max_results is None else max_results
        if isinstance(result_limit, bool) or not isinstance(result_limit, int):
            raise ValueError("max_results must be a positive integer")
        if result_limit < 1:
            raise ValueError("max_results must be a positive integer")
        result_limit = min(result_limit, len(original_documents))

        if min_score is not None and not math.isfinite(min_score):
            raise ValueError("min_score must be finite")

        request_documents = [
            truncate_reranker_document(document, max_candidate_tokens)
            for document in original_documents
        ]
        result = await self._async_request(query, request_documents, result_limit)
        parsed_results = self._parse_results(result, original_documents)
        if min_score is not None:
            parsed_results = [
                item for item in parsed_results if item.score >= min_score
            ]
        return parsed_results[:result_limit]

    async def _async_request(self, query: str, documents: list[str], result_limit: int) -> Any:
        """Send one request and return its decoded JSON body."""
        try:
            async with self._session.post(
                self.endpoint_url,
                headers=self._headers,
                json={
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                    "top_n": result_limit,
                },
            ) as response:
                if response.status in {401, 403}:
                    raise RerankerAuthenticationError("The reranker endpoint rejected the credentials.")
                if response.status == 404 or response.status >= 500:
                    raise RerankerConnectionError(f"The reranker endpoint returned HTTP {response.status}.")
                if response.status >= 400:
                    raise RerankerResponseError(f"The reranker endpoint rejected the request with HTTP {response.status}.")
                try:
                    return await response.json(content_type=None)
                except ValueError as err:
                    raise RerankerResponseError("The reranker endpoint returned invalid JSON.") from err
        except (RerankerAuthenticationError, RerankerConnectionError, RerankerResponseError):
            raise
        except (ClientError, TimeoutError) as err:
            raise RerankerConnectionError("Unable to connect to the reranker endpoint.") from err

    @staticmethod
    def _parse_results(response: Any, documents: list[str]) -> list[RerankerResult]:
        """Validate results and map indices to the original documents."""
        if not isinstance(response, Mapping):
            raise RerankerResponseError("The reranker endpoint returned an invalid response.")
        raw_results = response.get("results")
        if not isinstance(raw_results, list) or not raw_results:
            raise RerankerResponseError("The reranker endpoint returned no results.")

        results: list[RerankerResult] = []
        seen_indices: set[int] = set()
        for raw_result in raw_results:
            if not isinstance(raw_result, Mapping):
                raise RerankerResponseError("The reranker endpoint returned an invalid result.")
            index = raw_result.get("index")
            score = raw_result.get("relevance_score")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(documents)
                or index in seen_indices
                or isinstance(score, bool)
                or not isinstance(score, int | float)
                or not math.isfinite(score)
            ):
                raise RerankerResponseError("The reranker endpoint returned an invalid result.")
            seen_indices.add(index)
            results.append(
                RerankerResult(
                    index=index,
                    score=float(score),
                    document=documents[index],
                )
            )

        results.sort(key=lambda result: result.score, reverse=True)
        return results
