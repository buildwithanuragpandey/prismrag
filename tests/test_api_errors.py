import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from adaptive_agentic_rag.api.app import create_app
from adaptive_agentic_rag.api.service import RAGService
from adaptive_agentic_rag.api.errors import InferenceError


def test_empty_query_validation_error():
    service = RAGService(nodes=MagicMock())
    app = create_app(service=service)
    with TestClient(app) as client:
        response = client.post("/v1/query", json={"query": ""})
        assert response.status_code == 422


def test_whitespace_query_validation_error():
    service = RAGService(nodes=MagicMock())
    app = create_app(service=service)
    with TestClient(app) as client:
        response = client.post("/v1/query", json={"query": "   \n\t  "})
        assert response.status_code == 422


def test_missing_query_field_validation_error():
    service = RAGService(nodes=MagicMock())
    app = create_app(service=service)
    with TestClient(app) as client:
        response = client.post("/v1/query", json={"include_trace": True})
        assert response.status_code == 422


def test_pipeline_unavailable_error():
    service = RAGService(nodes=None)
    app = create_app(service=service)
    with TestClient(app) as client:
        response = client.post("/v1/query", json={"query": "Sample test query"})
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "PIPELINE_UNAVAILABLE"


def test_inference_failure_error_handling():
    service = RAGService(nodes=MagicMock())

    async def fail_query(request):
        raise InferenceError("Mocked GPU out of memory exception")

    service.query = fail_query
    app = create_app(service=service)
    with TestClient(app) as client:
        response = client.post("/v1/query", json={"query": "Trigger failure"})
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "INFERENCE_FAILURE"
        assert "Mocked GPU out of memory" in data["error"]["message"]
