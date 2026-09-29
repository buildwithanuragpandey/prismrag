import pytest
from fastapi.testclient import TestClient
from adaptive_agentic_rag.api.app import create_app
from adaptive_agentic_rag.api.service import RAGService


def test_health_endpoint():
    app = create_app(service=RAGService(nodes=None))
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_readiness_endpoint_not_ready():
    service = RAGService(nodes=None)
    app = create_app(service=service)
    with TestClient(app) as client:
        response = client.get("/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"
        assert data["pipeline_loaded"] is False


def test_readiness_endpoint_ready():
    class DummyNodes:
        pass

    service = RAGService(nodes=DummyNodes())
    app = create_app(service=service)
    with TestClient(app) as client:
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["pipeline_loaded"] is True
        assert data["qdrant_collection"] == "multihop_chunks_v2"


def test_system_info_endpoint():
    service = RAGService(nodes=None)
    app = create_app(service=service)
    with TestClient(app) as client:
        response = client.get("/v1/system")
        assert response.status_code == 200
        data = response.json()
        assert data["project"] == "Adaptive Agentic RAG with Hybrid Retrieval, Reranking & Self-Correction"
        assert data["api_version"] == "1.0.0"
        assert data["architecture_version"] == "V2-A (Frozen Canonical)"
        assert data["embedding_model"] == "Qwen/Qwen3-Embedding-0.6B"
        assert data["qdrant_collection"] == "multihop_chunks_v2"
        assert data["corpus_chunks"] == 8173
