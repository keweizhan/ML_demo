.PHONY: help sync fmt lint type test train run clean docker-build docker-run

help:
	@echo "Targets:"
	@echo "  make sync         - install deps (uv)"
	@echo "  make fmt          - format (ruff)"
	@echo "  make lint         - lint (ruff)"
	@echo "  make type         - typecheck (mypy)"
	@echo "  make test         - pytest"
	@echo "  make train        - train model -> artifacts/"
	@echo "  make run          - run api (reload)"
	@echo "  make clean        - remove caches/artifacts"
	@echo "  make docker-build - build docker image"
	@echo "  make docker-run   - run docker container"

sync:
	uv sync

fmt:
	uv run ruff format .

lint:
	uv run ruff check .

type:
	uv run mypy src

test:
	uv run pytest

train:
	uv run python -m ml_service.train

run:
	uv run uvicorn ml_service.api:app --host $${ML_HOST:-0.0.0.0} --port $${ML_PORT:-8000} --reload

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build *.egg-info artifacts

docker-build:
	docker build -t ml_service:dev .

docker-run:
	docker run --rm -p 8000:8000 ml_service:dev
