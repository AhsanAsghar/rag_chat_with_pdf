"""
FastAPI entrypoint. Exposes the full RAG pipeline as a small REST
API that the Streamlit frontend (or any other client) talks to:

  POST   /upload            -> ingest a PDF/md/txt file
  POST   /ask                -> ask a question, get an answer + sources
  GET    /documents          -> list ingested documents
  DELETE /documents/{doc_id} -> remove a single document
  POST   /reset               -> wipe the entire vector store
"""
import os
import uuid
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .models.schemas import (
    UploadResponse,
    DocumentListResponse,
    DocumentInfo,
    AskRequest,
    AskResponse,
    SourceChunk,
    ResetResponse,
)
from .ingestion.loader import load_document
from .ingestion.chunker import chunk_pages
from .ingestion.embedder import embed_texts
from .retrieval import vector_store
from .retrieval.retriever import retrieve
from .generation.prompt_builder import build_messages
from .generation.llm_client import generate_answer

settings.validate()
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.chroma_persist_dir, exist_ok=True)

app = FastAPI(
    title="Chat with your PDF/Notes — RAG API",
    description="Upload documents, then ask questions grounded in their content.",
    version="1.0.0",
)

# Wide-open CORS since this is a local/portfolio project talking to a
# local Streamlit frontend. Lock this down to specific origins before
# deploying anything public-facing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "RAG API is running."}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    allowed_extensions = (".pdf", ".md", ".txt")
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}",
        )

    document_id = str(uuid.uuid4())
    saved_path = os.path.join(settings.upload_dir, f"{document_id}_{file.filename}")

    # Enforce max upload size while streaming to disk.
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    size = 0
    with open(saved_path, "wb") as out_file:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                out_file.close()
                os.remove(saved_path)
                raise HTTPException(
                    status_code=400,
                    detail=f"File exceeds max upload size of {settings.max_upload_size_mb}MB.",
                )
            out_file.write(chunk)

    try:
        # 1. Extract text
        pages = load_document(saved_path, file.filename)

        # 2. Chunk it
        chunks = chunk_pages(
            pages,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        if not chunks:
            raise HTTPException(status_code=400, detail="No text could be extracted from this file.")

        # 3. Embed chunks
        chunk_texts = [c.text for c in chunks]
        embeddings = embed_texts(chunk_texts)

        # 4. Store in vector DB with metadata for citations
        ids = [f"{document_id}_{c.chunk_index}" for c in chunks]
        metadatas = [
            {
                "document_id": document_id,
                "filename": file.filename,
                "page": c.page,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]
        vector_store.add_chunks(
            ids=ids,
            embeddings=embeddings,
            documents=chunk_texts,
            metadatas=metadatas,
        )

    except ValueError as e:
        os.remove(saved_path)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        if os.path.exists(saved_path):
            os.remove(saved_path)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    return UploadResponse(
        document_id=document_id,
        filename=file.filename,
        num_chunks=len(chunks),
        num_pages=len(pages),
        message=f"Successfully ingested '{file.filename}' into {len(chunks)} chunks.",
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chunks = retrieve(
        question=request.question,
        document_ids=request.document_ids,
    )

    messages = build_messages(
        question=request.question,
        chunks=chunks,
        history=request.history,
    )

    try:
        answer = generate_answer(messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {str(e)}")

    sources = [
        SourceChunk(
            document_id=c.document_id,
            filename=c.filename,
            page=c.page if c.page else None,
            chunk_index=c.chunk_index,
            text_snippet=(c.text[:220] + "...") if len(c.text) > 220 else c.text,
            similarity_score=c.similarity_score,
        )
        for c in chunks
    ]

    return AskResponse(answer=answer, sources=sources)


@app.get("/documents", response_model=DocumentListResponse)
def list_documents():
    docs = vector_store.list_documents()
    return DocumentListResponse(
        documents=[DocumentInfo(**d) for d in docs]
    )


@app.delete("/documents/{document_id}", response_model=ResetResponse)
def delete_document(document_id: str):
    vector_store.delete_document(document_id)

    # Also clean up the file on disk, if present.
    for fname in os.listdir(settings.upload_dir):
        if fname.startswith(document_id):
            os.remove(os.path.join(settings.upload_dir, fname))

    return ResetResponse(message=f"Document {document_id} removed.")


@app.post("/reset", response_model=ResetResponse)
def reset_everything():
    vector_store.reset_all()
    shutil.rmtree(settings.upload_dir, ignore_errors=True)
    os.makedirs(settings.upload_dir, exist_ok=True)
    return ResetResponse(message="All documents and embeddings cleared.")
