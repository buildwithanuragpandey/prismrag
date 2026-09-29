.PHONY: setup install ingest index test benchmark demo server clean help

PYTHON := python
SRC := src
PRISM_DIR := .

help:
	@echo "PRISM Adaptive Streaming RAG"
	@echo "=========================="
	@echo "make setup       Install all dependencies"
	@echo "make ingest      Download corpus and process chunks"
	@echo "make index       Build FAISS + BM25 indexes"
	@echo "make test        Run all unit tests"
	@echo "make benchmark   Run full benchmark suite (G1-G6 + all metrics)"
	@echo "make demo        Start the API server + frontend"
	@echo "make server      Start the API server only"
	@echo "make ablation    Run ablation studies"
	@echo "make clean       Remove build artifacts"

setup:
	$(PYTHON) -m pip install -r requirements.txt
	@echo "Setup complete."

ingest:
	@echo "Ingesting corpus..."
	$(PYTHON) scripts/ingest_corpus.py

index:
	@echo "Building indexes..."
	$(PYTHON) scripts/build_indexes.py

test:
	@echo "Running unit tests..."
	$(PYTHON) -m pytest tests/unit/ -v --tb=short

test-all:
	$(PYTHON) -m pytest tests/ -v --tb=short

benchmark:
	@echo "Running PRISM benchmark..."
	$(PYTHON) experiments/run_benchmark.py

ablation:
	@echo "Running ablation studies..."
	$(PYTHON) experiments/ablation/run_ablation.py

server:
	@echo "Starting PRISM API server..."
	$(PYTHON) -m uvicorn src.prism.api.app:app --host 0.0.0.0 --port 8000 --reload

demo:
	@echo "Starting demo..."
	$(PYTHON) -m uvicorn src.prism.api.app:app --host 0.0.0.0 --port 8000 &
	@echo "API: http://localhost:8000"
	@echo "Docs: http://localhost:8000/docs"
	@echo "Frontend: open frontend/index.html in browser"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf data/faiss_index 2>/dev/null || true
	@echo "Cleaned."
