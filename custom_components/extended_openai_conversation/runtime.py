"""Runtime data for configured AI endpoints."""

from dataclasses import dataclass, field

from openai import AsyncClient

from .context_management import (
    EmbeddingMemoryStore,
    RecallSnapshotStore,
    RerankerClient,
)


@dataclass(slots=True)
class OpenAIServiceRuntime:
    """Clients and local state shared by an AI endpoint's entities."""

    client: AsyncClient
    embedding_client: AsyncClient | None = None
    embedding_memory: EmbeddingMemoryStore | None = None
    reranker_client: RerankerClient | None = None
    recall_snapshots: RecallSnapshotStore = field(default_factory=RecallSnapshotStore)
