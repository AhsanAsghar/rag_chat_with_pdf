"""
Request/response models shared between the API layer and the
frontend. Keeping these in one place means the FastAPI docs
(/docs) are always accurate and the frontend knows exactly what
shape to expect back.
"""
from typing import List, Optional
from pydantic import BaseModel


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    num_chunks: int
    num_pages: Optional[int] = None
    message: str


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    num_chunks: int


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]


class AskRequest(BaseModel):
    question: str
    # Optional: restrict retrieval to specific documents. If empty,
    # searches across everything that's been ingested.
    document_ids: Optional[List[str]] = None
    # Optional: prior turns for follow-up questions, e.g.
    # [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    history: Optional[List[dict]] = None


class SourceChunk(BaseModel):
    document_id: str
    filename: str
    page: Optional[int] = None
    chunk_index: int
    text_snippet: str
    similarity_score: float


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]


class ResetResponse(BaseModel):
    message: str
