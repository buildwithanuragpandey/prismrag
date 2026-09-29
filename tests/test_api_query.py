import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from adaptive_agentic_rag.api.app import create_app
from adaptive_agentic_rag.api.service import RAGService
from adaptive_agentic_rag.api.schemas import (
    QueryResponse,
    CitationResponse,
    SourceItem,
    RetryInfo,
    TimingInfo,
    TraceInfo,
)


@pytest.fixture
def mock_rag_service():
    service = RAGService(nodes=MagicMock())

    async def fake_query(request):
        abstained = "unknown" in request.query.lower()
        answer_text = (
            "I don't have enough evidence in the provided sources to answer reliably."
            if abstained
            else "Google announced new search features [1]."
        )
        direct_ans = "UNKNOWN" if abstained else "Yes"
        citations = [] if abstained else [
            CitationResponse(
                citation_id=1,
                document_id="doc_100_chunk_0",
                source="TechCrunch",
                title="Google Search Updates",
                url="https://techcrunch.com/google-search",
                supporting_text="Google today announced search features.",
                entailment_score=0.92,
            )
        ]
        sources = [
            SourceItem(
                document_id="doc_100_chunk_0",
                source="TechCrunch",
                title="Google Search Updates",
                url="https://techcrunch.com/google-search",
            )
        ]
        timing = TimingInfo(
            total_ms=120.5,
            route_ms=5.2,
            retrieval_ms=30.1,
            retry_ms=None,
            generation_ms=85.2,
        )
        trace = None
        if request.include_trace:
            trace = TraceInfo(
                route="comparison_query",
                retrieval_strategy="hybrid",
                retrieved_candidate_count=10,
                evidence_sufficient_initially=not abstained,
                evidence_sufficient_final=not abstained,
                retry_attempted=False,
                retry_target_sources=[],
                retry_rescued=False,
                grounded_claim_count=1 if not abstained else 0,
                relevant_claim_count=1 if not abstained else 0,
                semantic_verifier_decision="UNKNOWN" if abstained else "SUPPORTED",
                citation_valid=True,
                abstention_reason="Insufficient initial evidence" if abstained else None,
                timing=timing,
            )

        return QueryResponse(
            request_id=request.request_id or "req_generated_123",
            query=request.query,
            answer=answer_text,
            direct_answer=direct_ans,
            abstained=abstained,
            citations=citations,
            sources=sources,
            route="comparison_query",
            retry=RetryInfo(attempted=False, rescued=False, target_sources=[]),
            latency_ms=120.5,
            timing=timing,
            trace=trace,
        )

    service.query = fake_query
    return service


def test_query_successful_answer(mock_rag_service):
    app = create_app(service=mock_rag_service)
    with TestClient(app) as client:
        response = client.post(
            "/v1/query",
            json={
                "query": "Did Google release updates?",
                "request_id": "custom_req_001",
                "include_trace": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == "custom_req_001"
        assert response.headers["X-Request-ID"] == "custom_req_001"
        assert data["abstained"] is False
        assert data["direct_answer"] == "Yes"
        assert len(data["citations"]) == 1
        assert data["citations"][0]["citation_id"] == 1
        assert data["citations"][0]["source"] == "TechCrunch"
        assert data["trace"] is None


def test_query_abstention_returns_200(mock_rag_service):
    app = create_app(service=mock_rag_service)
    with TestClient(app) as client:
        response = client.post(
            "/v1/query",
            json={"query": "Ask an unknown question requiring abstention"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["abstained"] is True
        assert data["direct_answer"] == "UNKNOWN"
        assert len(data["citations"]) == 0


def test_query_include_trace(mock_rag_service):
    app = create_app(service=mock_rag_service)
    with TestClient(app) as client:
        response = client.post(
            "/v1/query",
            json={
                "query": "Did Google release updates?",
                "include_trace": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["trace"] is not None
        assert data["trace"]["retrieval_strategy"] == "hybrid"
        assert data["trace"]["citation_valid"] is True
        assert data["trace"]["timing"]["total_ms"] == 120.5
