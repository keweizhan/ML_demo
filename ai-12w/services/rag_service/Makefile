.PHONY: help sync fmt lint test ingest run clean

help:
	@echo "Targets:"
	@echo "  make sync   - install deps (uv)"
	@echo "  make fmt    - format (ruff)"
	@echo "  make lint   - lint (ruff)"
	@echo "  make test   - pytest"
	@echo "  make ingest - build rag index -> artifacts/"
	@echo "  make run    - run api (reload)"
	@echo "  make clean  - remove caches/artifacts"

sync:
	uv sync

fmt:
	uv run ruff format .

lint:
	uv run ruff check .

test:
	uv run pytest

ingest:
	uv run python -m rag_service.ingest

run:
	uv run uvicorn rag_service.api:app --host $${RAG_HOST:-0.0.0.0} --port $${RAG_PORT:-8001} --reload

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build *.egg-info artifacts/rag_index.joblib