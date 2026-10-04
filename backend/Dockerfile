# Backend: FastAPI + ONNX Runtime (CPU). Build context = repository root (see docker-compose.yml).
# Only installs dependencies and copies code: no datasets, no training, no models.
# The ONNX models are mounted read-only at /app/models at run time (docker-compose.yml).
FROM python:3.11.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODELS_DIR=/app/models \
    SAMPLES_DIR=/app/backend/samples

WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Application code + the three small shared modules (same corruption code as training).
COPY src/__init__.py src/config.py src/corruptions.py src/
COPY backend/__init__.py backend/__init__.py
COPY backend/app backend/app
COPY backend/samples backend/samples

# Run as a non-root user.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
