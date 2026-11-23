"""Normalize configured conversation-preservation methods."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from ..const import (
    CONTEXT_PRESERVATION_METHOD_COMPACT,
    CONTEXT_PRESERVATION_METHOD_EMBEDDINGS,
    CONTEXT_PRESERVATION_METHOD_KEYWORD,
    CONTEXT_PRESERVATION_METHODS,
)


def context_preservation_methods(value: Any) -> frozenset[str]:
    """Return configured methods, including values saved by earlier builds."""
    if isinstance(value, str):
        return {
            "clear": frozenset(),
            "compact": frozenset({CONTEXT_PRESERVATION_METHOD_COMPACT}),
            "retrieve": frozenset({CONTEXT_PRESERVATION_METHOD_EMBEDDINGS}),
            "retrieve_compact": frozenset(
                {
                    CONTEXT_PRESERVATION_METHOD_COMPACT,
                    CONTEXT_PRESERVATION_METHOD_EMBEDDINGS,
                }
            ),
        }.get(value, frozenset())
    if not isinstance(value, Collection):
        return frozenset()
    return frozenset(
        method
        for method in value
        if isinstance(method, str) and method in CONTEXT_PRESERVATION_METHODS
    )


def archives_context(methods: Collection[str]) -> bool:
    """Return whether either lossless archive method is enabled."""
    return bool(
        {
            CONTEXT_PRESERVATION_METHOD_EMBEDDINGS,
            CONTEXT_PRESERVATION_METHOD_KEYWORD,
        }.intersection(methods)
    )
