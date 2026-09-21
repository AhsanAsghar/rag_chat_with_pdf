"""
Wraps a persisted ChromaDB collection. All embedding storage and
similarity search go through this module so the rest of the app
never touches Chroma's API directly — makes it easy to swap vector
stores later (e.g. to FAISS or Pinecone) without touching ingestion
or retrieval logic.
"""
from typing import List, Dict, Any, Optional
import chromadb
from ..config import settings

_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
_collection = _client.get_or_create_collection(
    name=settings.chroma_collection_name,
    metadata={"hnsw:space": "cosine"},
)


def add_chunks(
    ids: List[str],
    embeddings: List[List[float]],
    documents: List[str],
    metadatas: List[Dict[str, Any]],
) -> None:
    """Stores chunk text + embedding + metadata (doc id, filename, page, etc.)."""
    _collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def query(
    query_embedding: List[float],
    top_k: int,
    document_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Runs a similarity search. If document_ids is provided, restricts
    the search to chunks belonging to those documents only (useful
    when a user wants to ask questions about one specific upload).
    """
    where_filter = None
    if document_ids:
        where_filter = {"document_id": {"$in": document_ids}}

    return _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
    )


def list_documents() -> List[Dict[str, Any]]:
    """
    Returns distinct documents currently stored, with their chunk
    counts, by scanning stored metadata. Fine at small/medium scale;
    a production system would maintain a separate documents table.
    """
    all_items = _collection.get(include=["metadatas"])
    metadatas = all_items.get("metadatas", [])

    doc_map: Dict[str, Dict[str, Any]] = {}
    for meta in metadatas:
        doc_id = meta.get("document_id")
        if doc_id is None:
            continue
        if doc_id not in doc_map:
            doc_map[doc_id] = {
                "document_id": doc_id,
                "filename": meta.get("filename", "unknown"),
                "num_chunks": 0,
            }
        doc_map[doc_id]["num_chunks"] += 1

    return list(doc_map.values())


def delete_document(document_id: str) -> None:
    """Removes all chunks belonging to a specific document."""
    _collection.delete(where={"document_id": document_id})


def reset_all() -> None:
    """Wipes the entire collection. Used by the /reset endpoint."""
    global _collection
    _client.delete_collection(settings.chroma_collection_name)
    _collection = _client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )
