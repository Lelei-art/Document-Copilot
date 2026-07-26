"""Local Ollama embedding generation for document chunks."""

from __future__ import annotations

from ollama import Client

EMBED_BATCH_SIZE = 100
MODEL_NAME = "nomic-embed-text"

_client = Client(host="http://localhost:11434")


def embed_texts(
    texts: list[str],
    *,
    batch_size: int = EMBED_BATCH_SIZE,
) -> list[list[float]]:
    """
    Generate embeddings locally using Ollama.

    Uses the nomic-embed-text model (768 dimensions),
    matching the Supabase pgvector column.
    """

    if not texts:
        return []

    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]

        for text in batch:
            response = _client.embed(
                model=MODEL_NAME,
                input=text,
            )

            embedding = response["embeddings"][0]

            if len(embedding) != 768:
                raise ValueError(
                    f"Expected embedding dimension 768, got {len(embedding)}"
                )

            vectors.append(embedding)

    return vectors
