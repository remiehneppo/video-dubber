FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIDEO_DUBBER_WORKSPACE=/app/workspace

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY cli.py ./
COPY dubber ./dubber
COPY web ./web
COPY schemas ./schemas
COPY config.example.yaml ./.env.example ./
COPY docs ./docs

RUN pip install --no-cache-dir -e .

EXPOSE 8080
CMD ["dubber", "web", "--workspace", "/app/workspace", "--host", "0.0.0.0", "--port", "8080"]
