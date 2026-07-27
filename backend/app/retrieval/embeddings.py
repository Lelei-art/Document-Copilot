"""Local Ollama query embedding for live retrieval."""

from __future__ import annotations

from ollama import Client

from app.config import settings

_client = Client(host=settings.ollama_base_url)


def embed_query(text: str) -> list[float]:
    """Generate a query embedding locally using Ollama."""

    response = _client.embed(
        model=settings.embedding_model,
        input=text,
    )

    embedding = response["embeddings"][0]

    if len(embedding) != settings.embedding_dimensions:
        raise ValueError(
            f"Expected embedding dimension {settings.embedding_dimensions}, "
            f"got {len(embedding)}"
        )

    return embedding
