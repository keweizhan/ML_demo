from __future__ import annotations

from pathlib import Path

from .config import settings
from .rag import build_index, save_index


def main() -> None:
    docs_dir = Path(settings.docs_dir)
    index_path = Path(settings.index_path)

    index = build_index(docs_dir, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    save_index(index, index_path)

    print("✅ RAG index saved")
    print(f"  docs_dir  -> {docs_dir.resolve()}")
    print(f"  index     -> {index_path.resolve()}")
    print(f"  chunks    -> {len(index.chunks)}")


if __name__ == "__main__":
    main()
