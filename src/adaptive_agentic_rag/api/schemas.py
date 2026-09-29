from typing import Any
from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        description="The user query to be answered by Adaptive Agentic RAG.",
        min_length=1,
        max_length=2000,
        examples=["Did Google and The Verge report the same outcome on antitrust negotiations?"],
    )
    request_id: str | None = Field(
        default=None,
        description="Optional unique client request identifier. Auto-generated if omitted.",
        examples=["req_12345"],
    )
    include_trace: bool = Field(
        default=False,
        description="Whether to include safe internal engineering observability trace.",
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_blank(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Query cannot be empty or whitespace-only.")
        return trimmed


class CitationResponse(BaseModel):
    citation_id: int = Field(..., description="1-indexed numeric citation identifier ([1], [2], ...).")
    document_id: str = Field(..., description="Unique corpus document identifier.")
    source: str | None = Field(default=None, description="Publisher/source name.")
    title: str | None = Field(default=None, description="Article title.")
    url: str | None = Field(default=None, description="Source URL if available.")
    supporting_text: str | None = Field(default=None, description="Specific supporting premise snippet.")
    entailment_score: float | None = Field(default=None, description="NLI entailment confidence score.")


class SourceItem(BaseModel):
    document_id: str
    source: str | None = None
    title: str | None = None
    url: str | None = None


class RetryInfo(BaseModel):
    attempted: bool = Field(..., description="Whether a retrieval retry was executed.")
    rescued: bool = Field(..., description="Whether evidence was successfully rescued during retry.")
    target_sources: list[str] = Field(default_factory=list, description="Explicit publisher sources targeted during retry.")


class TimingInfo(BaseModel):
    total_ms: float = Field(..., description="Total end-to-end processing time in milliseconds.")
    route_ms: float | None = Field(default=None, description="Query routing latency in milliseconds.")
    retrieval_ms: float | None = Field(default=None, description="Retrieval latency in milliseconds.")
    retry_ms: float | None = Field(default=None, description="Retry retrieval latency in milliseconds.")
    generation_ms: float | None = Field(default=None, description="Generation and verification latency in milliseconds.")


class TraceInfo(BaseModel):
    route: str | None = None
    retrieval_strategy: str | None = None
    retrieved_candidate_count: int = 0
    evidence_sufficient_initially: bool | None = None
    evidence_sufficient_final: bool | None = None
    retry_attempted: bool = False
    retry_target_sources: list[str] = Field(default_factory=list)
    retry_rescued: bool = False
    grounded_claim_count: int = 0
    relevant_claim_count: int = 0
    semantic_verifier_decision: str | None = None
    citation_valid: bool = True
    abstention_reason: str | None = None
    timing: TimingInfo | None = None


class QueryResponse(BaseModel):
    request_id: str = Field(..., description="Request identifier.")
    query: str = Field(..., description="Normalized user query.")
    answer: str = Field(..., description="Full generated answer with citations, or abstention message.")
    direct_answer: str | None = Field(default=None, description="Concise direct answer proposition or UNKNOWN.")
    abstained: bool = Field(..., description="True if the pipeline safely abstained due to insufficient or ungrounded evidence.")
    citations: list[CitationResponse] = Field(default_factory=list, description="Entailed, verified inline citations.")
    sources: list[SourceItem] = Field(default_factory=list, description="Unique source documents referenced.")
    route: str | None = Field(default=None, description="Routing taxonomy category.")
    retry: RetryInfo = Field(..., description="Adaptive retry execution details.")
    latency_ms: float = Field(..., description="Total pipeline latency in milliseconds.")
    timing: TimingInfo | None = Field(default=None, description="Detailed stage timings.")
    trace: TraceInfo | None = Field(default=None, description="Observability trace (present only when include_trace=True).")


class HealthResponse(BaseModel):
    status: str = Field(default="ok", description="Process health status.")


class ReadyResponse(BaseModel):
    status: str = Field(default="ready", description="Inference readiness status.")
    pipeline_loaded: bool = Field(..., description="Whether the RAG execution graph is initialized.")
    qdrant_collection: str = Field(..., description="Canonical Qdrant collection name.")
    models_initialized: bool = Field(..., description="Whether embedding, reranking, and generation models are loaded.")
    device: str = Field(..., description="Active compute device (e.g., cuda:0 or cpu).")


class SystemInfoResponse(BaseModel):
    project: str = "Adaptive Agentic RAG"
    api_version: str = "1.0.0"
    architecture_version: str = "V2-A (Frozen Canonical)"
    embedding_model: str
    reranker_model: str
    generator_model: str
    grounding_model: str
    qdrant_collection: str
    corpus_file: str
    corpus_chunks: int
    device: str
    cuda_available: bool
    python_version: str
    torch_version: str


class ErrorDetail(BaseModel):
    code: str = Field(..., description="Machine-readable error code.")
    message: str = Field(..., description="Human-readable error description.")
    details: Any | None = Field(default=None, description="Optional structured diagnostic details.")


class ErrorResponse(BaseModel):
    error: ErrorDetail
