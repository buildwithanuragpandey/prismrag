import logging
from fastapi import APIRouter, Depends, Response
from adaptive_agentic_rag.api.schemas import QueryRequest, QueryResponse, ErrorResponse
from adaptive_agentic_rag.api.service import RAGService
from adaptive_agentic_rag.api.dependencies import get_rag_service

logger = logging.getLogger("adaptive_agentic_rag.api.query")

router = APIRouter(prefix="/v1", tags=["Inference"])


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        200: {"description": "Successful RAG query response (includes safe fail-closed abstentions)."},
        422: {"model": ErrorResponse, "description": "Validation error for empty or invalid query payload."},
        500: {"model": ErrorResponse, "description": "Internal server/inference execution error."},
        503: {"model": ErrorResponse, "description": "Pipeline unavailable or uninitialized."},
    },
    summary="Execute Multi-Hop RAG Query",
    description=(
        "Executes a user query through the frozen Adaptive Agentic RAG pipeline, "
        "including query routing, hybrid retrieval, cross-encoder reranking, "
        "evidence grading, adaptive retry, grounded generation, and semantic verification."
    ),
)
async def query_endpoint(
    request: QueryRequest,
    response: Response,
    service: RAGService = Depends(get_rag_service),
) -> QueryResponse:
    query_resp = await service.query(request)
    response.headers["X-Request-ID"] = query_resp.request_id

    logger.info(
        "Query completed: request_id=%s, route=%s, abstained=%s, latency=%.2fms",
        query_resp.request_id,
        query_resp.route,
        query_resp.abstained,
        query_resp.latency_ms,
    )
    return query_resp
