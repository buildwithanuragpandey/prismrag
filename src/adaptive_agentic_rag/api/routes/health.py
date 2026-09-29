from fastapi import APIRouter, Depends, status, Response
from adaptive_agentic_rag.api.schemas import HealthResponse, ReadyResponse
from adaptive_agentic_rag.api.service import RAGService
from adaptive_agentic_rag.api.dependencies import get_rag_service

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Process Liveness Probe",
    description="Lightweight endpoint to confirm the FastAPI server process is running.",
)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Pipeline Readiness Probe",
    description="Confirms whether the RAG pipeline models and vector store connections are initialized.",
)
async def ready(
    response: Response,
    service: RAGService = Depends(get_rag_service),
) -> ReadyResponse:
    readiness = service.get_readiness()
    if not readiness.pipeline_loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return readiness
