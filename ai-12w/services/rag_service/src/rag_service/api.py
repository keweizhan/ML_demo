from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import settings
from .rag import RagIndex, load_index, search

app = FastAPI(title="rag_service")

_index: RagIndex | None = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=10)


class ContextHit(BaseModel):
    source: str
    chunk_id: int
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    contexts: list[ContextHit]


def _get_index() -> RagIndex:
    global _index
    if _index is not None:
        return _index
    path = Path(settings.index_path)
    if not path.exists():
        raise FileNotFoundError(
            f"RAG index not found at {path}. Run: uv run python -m rag_service.ingest"
        )
    _index = load_index(path)
    return _index


@app.get("/health")
def health() -> dict[str, Any]:
    p = Path(settings.index_path)
    return {
        "status": "ok",
        "env": settings.env,
        "index_path": settings.index_path,
        "index_exists": p.exists(),
        "index_loaded": _index is not None,
        "docs_dir": settings.docs_dir,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        index = _get_index()
    except FileNotFoundError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e

    top_k = req.top_k or settings.top_k
    hits = search(index, req.question, top_k=top_k)

    contexts = [
        ContextHit(source=c.source, chunk_id=c.chunk_id, score=score, text=c.text)
        for c, score in hits
    ]

    # “最小 RAG”回答策略：先返回检索到的上下文 + 简单汇总句式（Day8 再接 LLM）
    if not contexts:
        answer = "I couldn't find relevant context in the indexed documents."
    else:
        answer = (
            "Based on the retrieved documents, the most relevant parts are shown below. "
            "You can refine the question for a more specific match."
        )

    return AskResponse(answer=answer, contexts=contexts)
