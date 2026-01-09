from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class Chunk:
    source: str  # filename
    chunk_id: int
    text: str


@dataclass(frozen=True)
class RagIndex:
    vectorizer: TfidfVectorizer
    matrix: Any  # sparse matrix
    chunks: list[Chunk]


def _read_text_files(docs_dir: Path) -> list[tuple[str, str]]:
    files: list[tuple[str, str]] = []
    for p in sorted(docs_dir.rglob("*")):
        if p.is_dir():
            continue
        if p.suffix.lower() not in {".txt", ".md"}:
            continue
        try:
            files.append((p.name, p.read_text(encoding="utf-8")))
        except UnicodeDecodeError:
            # fallback
            files.append((p.name, p.read_text(encoding="latin-1")))
    return files


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    n = len(text)
    step = max(1, chunk_size - overlap)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        start += step
    return chunks


def build_index(docs_dir: Path, chunk_size: int, overlap: int) -> RagIndex:
    raw_files = _read_text_files(docs_dir)

    chunks: list[Chunk] = []
    for fname, content in raw_files:
        parts = _chunk_text(content, chunk_size=chunk_size, overlap=overlap)
        for i, part in enumerate(parts):
            chunks.append(Chunk(source=fname, chunk_id=i, text=part))

    # TF-IDF baseline: fast, local, deterministic
    vectorizer = TfidfVectorizer(max_features=50000)
    matrix = vectorizer.fit_transform([c.text for c in chunks])

    return RagIndex(vectorizer=vectorizer, matrix=matrix, chunks=chunks)


def save_index(index: RagIndex, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(index, path)


def load_index(path: Path) -> RagIndex:
    return joblib.load(path)


def search(index: RagIndex, question: str, top_k: int) -> list[tuple[Chunk, float]]:
    q = question.strip()
    if not q:
        return []
    q_vec = index.vectorizer.transform([q])  # (1, d)
    # cosine sim for TF-IDF = normalized dot product; vectorizer output is L2-normalized by default
    scores = (index.matrix @ q_vec.T).toarray().ravel()  # (n,)
    if scores.size == 0:
        return []
    top_idx = scores.argsort()[::-1][:top_k]
    return [(index.chunks[i], float(scores[i])) for i in top_idx]
