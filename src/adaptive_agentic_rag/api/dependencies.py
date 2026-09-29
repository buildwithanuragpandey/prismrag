from fastapi import Request
from adaptive_agentic_rag.api.service import RAGService


def get_rag_service(request: Request) -> RAGService:
    service: RAGService | None = getattr(request.app.state, "rag_service", None)
    if service is None:
        return RAGService(nodes=None)
    return service
