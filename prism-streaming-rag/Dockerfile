FROM python:3.12-slim

LABEL maintainer="PRISM Research"
LABEL description="PRISM Adaptive Streaming RAG"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY configs/ ./configs/
COPY benchmarks/ ./benchmarks/
COPY experiments/ ./experiments/
COPY scripts/ ./scripts/
COPY docs/ ./docs/

# Create data directories
RUN mkdir -p data/processed data/faiss_index results/telemetry results/benchmark

# Environment
ENV PRISM_CONFIG_PATH=configs/config.yaml
ENV PYTHONPATH=/app/src

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "prism.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
