import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from adaptive_agentic_rag.api.errors import RAGAPIError, rag_api_error_handler, generic_exception_handler
from adaptive_agentic_rag.api.service import RAGService
from adaptive_agentic_rag.api.routes.health import router as health_router
from adaptive_agentic_rag.api.routes.system import router as system_router
from adaptive_agentic_rag.api.routes.query import router as query_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("adaptive_agentic_rag.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager for RAG model initialization and cleanup.
    """
    logger.info("Initializing Adaptive Agentic RAG service...")
    service = getattr(app.state, "rag_service", None)
    if service is None:
        service = RAGService()
        app.state.rag_service = service

    try:
        service.initialize()
        logger.info("Adaptive Agentic RAG service successfully ready for inference.")
        yield
    finally:
        logger.info("Shutting down Adaptive Agentic RAG service...")
        service.close()
        logger.info("Service shutdown complete.")


def create_app(
    service: RAGService | None = None,
    allow_cors: bool = True,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """
    Factory creating the production FastAPI application instance.
    """
    app = FastAPI(
        title="Adaptive Agentic RAG API",
        version="1.0.0",
        description=(
            "Production-oriented HTTP inference API for Adaptive Agentic RAG "
            "with Hybrid Retrieval, Reranking, Evidence Gating, and Semantic Verification."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan if service is None else None,
    )

    if service is not None:
        app.state.rag_service = service

    # CORS configuration
    if allow_cors:
        default_origins = [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            "http://localhost:5500",
            "http://127.0.0.1:5500",
        ]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins or default_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )

    # Register exception handlers
    app.add_exception_handler(RAGAPIError, rag_api_error_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # Include route modules
    app.include_router(health_router)
    app.include_router(system_router)
    app.include_router(query_router)

    # Mount demo static interface if present
    from pathlib import Path
    from starlette.staticfiles import StaticFiles
    demo_dir = Path(__file__).resolve().parents[3] / "demo"
    if demo_dir.is_dir():
        app.mount("/demo", StaticFiles(directory=str(demo_dir), html=True), name="demo")

    return app


# Default app instance for standard ASGI servers
app = create_app()
