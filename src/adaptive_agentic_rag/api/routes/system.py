from fastapi import APIRouter, Depends
from adaptive_agentic_rag.api.schemas import SystemInfoResponse
from adaptive_agentic_rag.api.service import RAGService
from adaptive_agentic_rag.api.dependencies import get_rag_service

router = APIRouter(prefix="/v1", tags=["System"])


@router.get(
    "/system",
    response_model=SystemInfoResponse,
    summary="System Architecture Metadata",
    description="Returns metadata about frozen models, devices, and the canonical Qdrant collection.",
)
async def system_info(
    service: RAGService = Depends(get_rag_service),
) -> SystemInfoResponse:
    return service.get_system_info()
