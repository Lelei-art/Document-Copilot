"""Local Ollama embedding generation for document chunks."""

from __future__ import annotations

from ollama import Client

from app.config import settings

EMBED_BATCH_SIZE = 100

_client = Client(host=settings.ollama_base_url)


def embed_texts(
    texts: list[str],
    *,
    batch_size: int = EMBED_BATCH_SIZE,
) -> list[list[float]]:
    """
    Generate embeddings locally using Ollama.

    Uses the configured Ollama embedding model, usually nomic-embed-text.
    """

    if not texts:
        return []

    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]

        for text in batch:
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

            vectors.append(embedding)

    return vectors
