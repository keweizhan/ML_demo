FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# install uv (dependency manager)
RUN pip install --no-cache-dir uv

# copy project metadata first (better layer caching)
COPY pyproject.toml uv.lock README.md ./

# copy source code
COPY src ./src

# if you already trained locally, bake artifacts into the image for demo
COPY artifacts ./artifacts

# install deps (no dev deps in image)
RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "ml_service.api:app", "--host", "0.0.0.0", "--port", "8000"]