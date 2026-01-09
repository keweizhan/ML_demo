from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    env: str = os.getenv("ENV", "local")
    docs_dir: str = os.getenv("RAG_DOCS_DIR", "docs")
    index_path: str = os.getenv("RAG_INDEX_PATH", "artifacts/rag_index.joblib")
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
    top_k: int = int(os.getenv("RAG_TOP_K", "4"))


settings = Settings()


def abs_path(p: str) -> Path:
    return Path(p).resolve()
