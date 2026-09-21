"""
Central place for all environment-driven configuration. Every other
module imports from here instead of calling os.getenv() directly,
so there's exactly one source of truth for config values.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- OpenAI ---
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # --- Vector store ---
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./data/vector_db")
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "documents")

    # --- Chunking ---
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "2000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # --- Retrieval ---
    top_k: int = int(os.getenv("TOP_K", "4"))

    # --- Uploads ---
    upload_dir: str = os.getenv("UPLOAD_DIR", "./data/uploads")
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))

    # --- Server ---
    backend_host: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    backend_port: int = int(os.getenv("BACKEND_PORT", "8000"))

    def validate(self):
        """Fail loudly and early if required secrets are missing,
        instead of surfacing a confusing error on the first request."""
        if not self.openai_api_key or self.openai_api_key == "your_openai_api_key_here":
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env "
                "and add your OpenAI API key before starting the server."
            )


settings = Settings()
