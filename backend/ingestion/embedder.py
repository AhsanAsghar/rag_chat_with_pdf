"""
Wraps OpenAI's embeddings endpoint. Used both to embed document
chunks at ingestion time and to embed the user's question at query
time — both must use the same model so vectors live in the same
space and similarity search is meaningful.
"""
from typing import List
from openai import OpenAI
from ..config import settings

_client = OpenAI(api_key=settings.openai_api_key)

# OpenAI allows batching multiple inputs in one call, which is much
# faster and cheaper than embedding one chunk at a time.
_BATCH_SIZE = 100


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embeds a list of texts and returns their vectors, in the same order."""
    if not texts:
        return []

    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        response = _client.embeddings.create(
            model=settings.embedding_model,
            input=batch,
        )
        all_embeddings.extend([item.embedding for item in response.data])

    return all_embeddings


def embed_query(query: str) -> List[float]:
    """Embeds a single query string (e.g. the user's question)."""
    return embed_texts([query])[0]
