
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/data/models

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY src/ ./src/

EXPOSE 8000 8501
VOLUME ["/data"]

CMD ["sh", "-c", "exec uvicorn src.companion_api:app --host 0.0.0.0 --port \"${PORT:-8000}\""]
