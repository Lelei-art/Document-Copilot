"""Local Ollama query embedding for live retrieval."""

from __future__ import annotations

from ollama import Client

MODEL_NAME = "nomic-embed-text"

_client = Client(host="http://localhost:11434")


def embed_query(text: str) -> list[float]:
    """Generate a query embedding locally using Ollama."""

    response = _client.embed(
        model=MODEL_NAME,
        input=text,
    )

    embedding = response["embeddings"][0]

    if len(embedding) != 768:
        raise ValueError(f"Expected embedding dimension 768, got {len(embedding)}")

    return embedding
