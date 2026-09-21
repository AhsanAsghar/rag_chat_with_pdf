"""
Given a user's question, embeds it and fetches the top-k most
relevant chunks from the vector store. This is the "R" in RAG.
"""
from dataclasses import dataclass
from typing import List, Optional
from ..ingestion.embedder import embed_query
from . import vector_store
from ..config import settings


@dataclass
class RetrievedChunk:
    document_id: str
    filename: str
    page: int
    chunk_index: int
    text: str
    similarity_score: float


def retrieve(
    question: str,
    document_ids: Optional[List[str]] = None,
    top_k: Optional[int] = None,
) -> List[RetrievedChunk]:
    query_embedding = embed_query(question)

    results = vector_store.query(
        query_embedding=query_embedding,
        top_k=top_k or settings.top_k,
        document_ids=document_ids,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved: List[RetrievedChunk] = []
    for text, meta, distance in zip(documents, metadatas, distances):
        # Chroma returns cosine *distance*; convert to a 0-1 similarity
        # score that's more intuitive to display to the user.
        similarity = max(0.0, 1.0 - distance)
        retrieved.append(
            RetrievedChunk(
                document_id=meta.get("document_id", ""),
                filename=meta.get("filename", "unknown"),
                page=meta.get("page", 0),
                chunk_index=meta.get("chunk_index", 0),
                text=text,
                similarity_score=round(similarity, 4),
            )
        )

    return retrieved
