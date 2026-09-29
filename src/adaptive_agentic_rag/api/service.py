import asyncio
import logging
import sys
import time
import uuid
from typing import Any
import anyio
import torch

from adaptive_agentic_rag.orchestration.graph import route_after_evidence
from adaptive_agentic_rag.orchestration.nodes import RAGNodes
from adaptive_agentic_rag.api.errors import PipelineUnavailableError, InferenceError
from adaptive_agentic_rag.api.schemas import (
    QueryRequest,
    QueryResponse,
    CitationResponse,
    SourceItem,
    RetryInfo,
    TimingInfo,
    TraceInfo,
    SystemInfoResponse,
    ReadyResponse,
)

logger = logging.getLogger("adaptive_agentic_rag.api")


class RAGService:
    """
    Production Application Service wrapping the frozen Adaptive Agentic RAG pipeline.
    """

    def __init__(self, nodes: RAGNodes | None = None, max_concurrent_requests: int = 1):
        self.nodes = nodes
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._initialized_at = time.time()

    @property
    def is_ready(self) -> bool:
        return self.nodes is not None

    def initialize(self) -> None:
        if self.nodes is None:
            logger.info("Initializing canonical RAGNodes stack...")
            self.nodes = RAGNodes()
            logger.info("Canonical RAGNodes stack successfully initialized.")

    def close(self) -> None:
        if self.nodes is not None:
            logger.info("Closing RAGNodes resources...")
            self.nodes.close()
            self.nodes = None

    async def query(self, request: QueryRequest) -> QueryResponse:
        if not self.is_ready or self.nodes is None:
            raise PipelineUnavailableError("RAG pipeline is not initialized or ready.")

        req_id = request.request_id or f"req_{uuid.uuid4().hex[:12]}"
        query_text = request.query.strip()

        async with self._semaphore:
            try:
                response = await anyio.to_thread.run_sync(
                    self._execute_pipeline,
                    self.nodes,
                    query_text,
                    req_id,
                    request.include_trace,
                )
                return response
            except Exception as e:
                logger.exception("Inference execution failed for request %s: %s", req_id, str(e))
                raise InferenceError(message=f"Pipeline execution failed: {str(e)}", details={"request_id": req_id})

    @staticmethod
    def _execute_pipeline(
        nodes: RAGNodes,
        query: str,
        request_id: str,
        include_trace: bool,
    ) -> QueryResponse:
        t_start = time.perf_counter()

        state = {
            "original_query": query,
            "current_query": query,
            "retry_count": 0,
            "max_retries": 1,
            "retry_target_sources": [],
        }

        # 1. Route
        t_route_start = time.perf_counter()
        state.update(nodes.route_query(state))
        t_route_ms = (time.perf_counter() - t_route_start) * 1000.0

        # 2. Retrieve & Context Attempt 0
        t_ret_start = time.perf_counter()
        state.update(nodes.retrieve(state))
        t_ret_ms = (time.perf_counter() - t_ret_start) * 1000.0

        state.update(nodes.build_context(state))
        state.update(nodes.grade_evidence(state))

        initial_route = route_after_evidence(state)
        initial_ev_suff = state.get("evidence_sufficient")
        rewrite_attempted = False
        rewrite_rescued = False
        t_retry_ms = 0.0

        if initial_route == "rewrite":
            rewrite_attempted = True
            state.update(nodes.rewrite_query(state))
            t_retry_start = time.perf_counter()
            state.update(nodes.retrieve(state))
            t_retry_ms = (time.perf_counter() - t_retry_start) * 1000.0
            state.update(nodes.build_context(state))
            state.update(nodes.grade_evidence(state))
            rewrite_rescued = bool(state.get("evidence_sufficient"))

        final_evidence_sufficient = bool(state.get("evidence_sufficient"))

        # 3. Generate
        t_gen_start = time.perf_counter()
        gen_result = nodes.generator.generate(
            query=query,
            context=state.get("context"),
            evidence_sufficient=final_evidence_sufficient,
        )
        t_gen_ms = (time.perf_counter() - t_gen_start) * 1000.0

        # 4. Grade Answer
        nodes.answer_grader.grade(
            query=query,
            generation_result=gen_result,
            evidence_sufficient=final_evidence_sufficient,
        )

        t_total_ms = (time.perf_counter() - t_start) * 1000.0

        # Map Citations and Sources
        context_items = state["context"].items if state.get("context") else []
        cid_to_item = {it.citation_id: it for it in context_items}
        doc_id_to_item = {it.document_id: it for it in context_items}

        # Build citations for cited IDs
        citations: list[CitationResponse] = []
        for cid in gen_result.cited_ids:
            if cid in cid_to_item:
                it = cid_to_item[cid]
                # Find supporting text and entailment score from claim_support_list if available
                supp_text = None
                ent_score = None
                for claim_sup in getattr(gen_result, "claim_support_list", []):
                    if getattr(claim_sup, "citation_id", None) == cid:
                        supp_text = getattr(claim_sup, "supporting_text", None)
                        ent_score = getattr(claim_sup, "entailment_score", None)
                        break

                citations.append(
                    CitationResponse(
                        citation_id=cid,
                        document_id=it.document_id,
                        source=it.source,
                        title=it.title,
                        url=it.url,
                        supporting_text=supp_text,
                        entailment_score=ent_score,
                    )
                )

        # Unique sources present in context
        sources: list[SourceItem] = []
        seen_docs = set()
        for it in context_items:
            if it.document_id not in seen_docs:
                seen_docs.add(it.document_id)
                sources.append(
                    SourceItem(
                        document_id=it.document_id,
                        source=it.source,
                        title=it.title,
                        url=it.url,
                    )
                )

        timing = TimingInfo(
            total_ms=round(t_total_ms, 2),
            route_ms=round(t_route_ms, 2),
            retrieval_ms=round(t_ret_ms, 2),
            retry_ms=round(t_retry_ms, 2) if rewrite_attempted else None,
            generation_ms=round(t_gen_ms, 2),
        )

        trace = None
        if include_trace:
            trace = TraceInfo(
                route=state.get("query_type"),
                retrieval_strategy=state.get("retrieval_strategy"),
                retrieved_candidate_count=len(state.get("retrieved_results", [])),
                evidence_sufficient_initially=initial_ev_suff,
                evidence_sufficient_final=final_evidence_sufficient,
                retry_attempted=rewrite_attempted,
                retry_target_sources=state.get("retry_target_sources", []),
                retry_rescued=rewrite_rescued,
                grounded_claim_count=len(getattr(gen_result, "grounded_claims", [])),
                relevant_claim_count=len(getattr(gen_result, "claim_support_list", [])),
                semantic_verifier_decision="UNKNOWN" if gen_result.direct_answer == "UNKNOWN" and gen_result.draft_direct_answer != "UNKNOWN" else "SUPPORTED",
                citation_valid=gen_result.citation_valid,
                abstention_reason=None if not gen_result.abstained else ("Insufficient initial evidence" if not final_evidence_sufficient else "Semantic conclusion safety abstention"),
                timing=timing,
            )

        direct_ans = gen_result.direct_answer
        if gen_result.abstained and (direct_ans is None or not direct_ans.strip()):
            direct_ans = "UNKNOWN"

        return QueryResponse(
            request_id=request_id,
            query=query,
            answer=gen_result.answer,
            direct_answer=direct_ans,
            abstained=gen_result.abstained,
            citations=citations,
            sources=sources,
            route=state.get("query_type"),
            retry=RetryInfo(
                attempted=rewrite_attempted,
                rescued=rewrite_rescued,
                target_sources=state.get("retry_target_sources", []),
            ),
            latency_ms=round(t_total_ms, 2),
            timing=timing,
            trace=trace,
        )

    def get_readiness(self) -> ReadyResponse:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return ReadyResponse(
            status="ready" if self.is_ready else "not_ready",
            pipeline_loaded=self.is_ready,
            qdrant_collection="multihop_chunks_v2",
            models_initialized=self.is_ready,
            device=f"{device}:{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "cpu",
        )

    def get_system_info(self) -> SystemInfoResponse:
        device_str = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        return SystemInfoResponse(
            project="Adaptive Agentic RAG with Hybrid Retrieval, Reranking & Self-Correction",
            api_version="1.0.0",
            architecture_version="V2-A (Frozen Canonical)",
            embedding_model="Qwen/Qwen3-Embedding-0.6B",
            reranker_model="BAAI/bge-reranker-base",
            generator_model="Qwen/Qwen2.5-1.5B-Instruct",
            grounding_model="cross-encoder/nli-deberta-v3-small",
            qdrant_collection="multihop_chunks_v2",
            corpus_file="data/processed/processed_corpus_v2.json",
            corpus_chunks=8173,
            device=device_str,
            cuda_available=torch.cuda.is_available(),
            python_version=sys.version.split()[0],
            torch_version=torch.__version__,
        )
