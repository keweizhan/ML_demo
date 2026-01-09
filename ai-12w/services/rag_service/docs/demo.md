# rag_service demo docs

How to start the service:
1) uv sync
2) uv run python -m rag_service.ingest
3) uv run uvicorn rag_service.api:app --host 0.0.0.0 --port 8001 --reload

The /ask endpoint retrieves top_k chunks from docs and returns contexts.
