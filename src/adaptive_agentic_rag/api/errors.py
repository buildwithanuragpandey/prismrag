from typing import Any
from fastapi import Request, status
from fastapi.responses import JSONResponse
from adaptive_agentic_rag.api.schemas import ErrorResponse, ErrorDetail


class RAGAPIError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Any | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details


class PipelineUnavailableError(RAGAPIError):
    def __init__(self, message: str = "The RAG inference pipeline is not initialized or unavailable."):
        super().__init__(
            code="PIPELINE_UNAVAILABLE",
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


class InvalidQueryError(RAGAPIError):
    def __init__(self, message: str = "The supplied query is invalid."):
        super().__init__(
            code="INVALID_QUERY",
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )


class InferenceError(RAGAPIError):
    def __init__(self, message: str = "An error occurred during pipeline inference execution.", details: Any | None = None):
        super().__init__(
            code="INFERENCE_FAILURE",
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
        )


async def rag_api_error_handler(request: Request, exc: RAGAPIError) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorDetail(
            code=exc.code,
            message=exc.message,
            details=exc.details,
        )
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=payload.model_dump(),
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorDetail(
            code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred during request processing.",
            details=str(exc) if not isinstance(exc, AssertionError) else None,
        )
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=payload.model_dump(),
    )
