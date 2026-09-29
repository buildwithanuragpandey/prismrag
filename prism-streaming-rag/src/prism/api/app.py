"""
PRISM FastAPI Application

Endpoints:
  POST   /sessions                     Create a new session
  POST   /sessions/{id}/chunks         Submit a transcript chunk
  POST   /sessions/{id}/constraint     Submit a late constraint
  GET    /sessions/{id}/state          Get current query state
  GET    /sessions/{id}/evidence       Get evidence store
  GET    /sessions/{id}/answers        Get all answer versions
  GET    /sessions/{id}/telemetry      Get session telemetry
  GET    /sessions/{id}/history        Get controller decision history
  WS     /ws/{session_id}             WebSocket: stream chunks live
  POST   /simulate                     Run a demo scenario
  GET    /benchmark/baselines          Run all baselines on a query
  GET    /health                       Health check
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project src is importable
_PRISM_SRC = Path(__file__).parents[2]
if str(_PRISM_SRC) not in sys.path:
    sys.path.insert(0, str(_PRISM_SRC))

from prism.config import load_config, PRISMConfig
from prism.streaming import TranscriptChunk, TranscriptSimulator, get_demo_scenario
from prism.telemetry import TelemetryBus, TelemetryEvent, EventType, get_telemetry_bus

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="PRISM Adaptive Streaming RAG",
    description=(
        "Samsung PRISM Research Project — "
        "Adaptive retrieval controller with utility-aware streaming RAG."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# CORS
config = load_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = Path(__file__).parents[3] / "frontend"
if (frontend_dir / "index.html").exists():
    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(frontend_dir / "index.html")

    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

# ──────────────────────────────────────────────────────────────
# Dependency: Session Manager
# ──────────────────────────────────────────────────────────────

_session_manager = None


def get_session_manager():
    global _session_manager
    if _session_manager is None:
        _session_manager = _build_session_manager(config)
    return _session_manager


def _build_session_manager(cfg: PRISMConfig):
    """Build the SessionManager with all injected dependencies."""
    from prism.models import EmbeddingModel, create_llm_adapter
    from prism.controller import (
        AdaptiveRetrievalController,
        ControllerConfig,
        IntentStabilityTracker,
        RetrievalUtilityEstimator,
        SemanticNoveltyDetector,
        WeightedLinearPolicy,
        QueryStateTracker,
    )
    from prism.intents import MultiIntentDecomposer
    from prism.session import SessionManager

    bus = get_telemetry_bus(cfg.telemetry.log_dir)

    # Embedding model
    try:
        embedding_model = EmbeddingModel(
            model_name=cfg.models.embedding.name,
            device=cfg.models.embedding.device,
        )
        embedding_fn = embedding_model
    except Exception as exc:
        logger.warning("Could not load embedding model: %s — novelty will use fallback", exc)
        embedding_fn = None

    # Novelty detector
    novelty_detector = None
    if embedding_fn is not None:
        novelty_detector = SemanticNoveltyDetector(
            embedding_fn=embedding_fn,
            high_threshold=cfg.controller.novelty.high_novelty_threshold,
            low_threshold=cfg.controller.novelty.low_novelty_threshold,
        )

    # Stability tracker
    stability_tracker = IntentStabilityTracker(
        stable_window=cfg.controller.intent_stability.stable_window,
        max_intents=cfg.controller.intent_stability.max_intents,
    )

    # Utility estimator
    w = cfg.controller.utility_weights
    policy = WeightedLinearPolicy(
        w_novelty=w.novelty,
        w_intent_stability=w.intent_stability,
        w_completeness=w.completeness,
        w_evidence_gap=w.evidence_gap,
        w_retrieval_cost=w.retrieval_cost,
        min_retrieval_interval_s=cfg.controller.min_retrieval_interval_s,
    )
    utility_estimator = RetrievalUtilityEstimator(policy=policy)

    # Controller
    ctrl_cfg = ControllerConfig.from_config(cfg)
    controller = AdaptiveRetrievalController(
        utility_estimator=utility_estimator,
        config=ctrl_cfg,
    )

    # Intent decomposer
    llm_fn = None
    try:
        llm_adapter = create_llm_adapter(cfg)
        llm_fn = llm_adapter.generate
    except Exception as exc:
        logger.warning("LLM not available: %s. Generation will be disabled.", exc)

    intent_decomposer = MultiIntentDecomposer(
        backend="llm" if llm_fn else "heuristic",
        llm_fn=llm_fn,
        max_intents=cfg.controller.intent_stability.max_intents,
    )

    # Retriever (optional — skip if corpus not built)
    retriever = None
    delta_retriever = None
    try:
        from prism.retrieval import PRISMHybridRetriever, DeltaRetriever
        corpus_path = cfg.corpus.processed_corpus_path
        if Path(corpus_path).exists() and embedding_fn is not None:
            retriever = PRISMHybridRetriever(
                corpus_path=corpus_path,
                embedding_fn=embedding_fn,
                faiss_index_path=cfg.vectorstore.faiss.index_path,
                faiss_metadata_path=cfg.vectorstore.faiss.metadata_path,
                top_k_bm25=cfg.retrieval.top_k_bm25,
                top_k_dense=cfg.retrieval.top_k_dense,
                top_k_rrf=cfg.retrieval.top_k_rrf,
                top_k_reranked=cfg.retrieval.top_k_reranked,
                rerank_enabled=cfg.models.reranker.enabled,
                reranker_model=cfg.models.reranker.name,
                rrf_k=cfg.retrieval.rrf_k,
            )
            delta_retriever = DeltaRetriever(retriever)
            logger.info("Retriever initialized.")
        else:
            logger.warning(
                "Corpus not found at %s. Retrieval disabled — run `make ingest` first.", corpus_path
            )
    except Exception as exc:
        logger.warning("Retriever init failed: %s", exc)

    return SessionManager(
        query_state_tracker=QueryStateTracker(),
        novelty_detector=novelty_detector,
        stability_tracker=stability_tracker,
        utility_estimator=utility_estimator,
        controller=controller,
        intent_decomposer=intent_decomposer,
        retriever=retriever,
        delta_retriever=delta_retriever,
        llm_fn=llm_fn,
        telemetry_bus=bus,
    )


# ──────────────────────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────────────────────

class CreateSessionResponse(BaseModel):
    session_id: str
    message: str = "Session created"


class ChunkRequest(BaseModel):
    text: str
    chunk_index: int = 0
    is_final: bool = False


class ConstraintRequest(BaseModel):
    new_text: str


class SimulateRequest(BaseModel):
    scenario: str = "workshop_planning"
    interval_s: float = 0.1      # Fast for API demo
    generate_answers: bool = True


# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0", "project": "PRISM Adaptive Streaming RAG"}


@app.post("/sessions", response_model=CreateSessionResponse)
async def create_session():
    manager = get_session_manager()
    session_id = manager.create_session()
    return CreateSessionResponse(session_id=session_id)


@app.post("/sessions/{session_id}/chunks")
async def process_chunk(session_id: str, req: ChunkRequest):
    manager = get_session_manager()

    import time
    chunk = TranscriptChunk(
        chunk_index=req.chunk_index,
        text=req.text,
        delta_text=req.text,
        session_id=session_id,
        timestamp=time.time(),
        is_final=req.is_final,
    )

    result = await manager.process_chunk(session_id, chunk)
    return result


@app.post("/sessions/{session_id}/constraint")
async def process_constraint(session_id: str, req: ConstraintRequest):
    manager = get_session_manager()
    result = await manager.process_late_constraint(session_id, req.new_text)
    return result


@app.get("/sessions/{session_id}/state")
async def get_state(session_id: str):
    manager = get_session_manager()
    state = manager.state_tracker._states.get(session_id)
    if state is None:
        raise HTTPException(404, "Session not found")
    return state.to_dict()


@app.get("/sessions/{session_id}/evidence")
async def get_evidence(session_id: str):
    manager = get_session_manager()
    session = manager.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    return {
        "summary": session.evidence_store.summary(),
        "items": [e.to_dict() for e in session.evidence_store.get_all()],
    }


@app.get("/sessions/{session_id}/answers")
async def get_answers(session_id: str):
    manager = get_session_manager()
    session = manager.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    return {
        "session_id": session_id,
        "version_count": session.version_manager.version_count,
        "versions": [v.to_dict() for v in session.version_manager.get_all_versions()],
        "latest": session.latest_answer.to_dict() if session.latest_answer else None,
    }


@app.get("/sessions/{session_id}/history")
async def get_history(session_id: str):
    manager = get_session_manager()
    history = manager.state_tracker.get_history(session_id)
    return {"session_id": session_id, "history": history}


@app.get("/sessions/{session_id}/telemetry")
async def get_telemetry(session_id: str):
    bus = get_telemetry_bus()
    events = bus.get_session_events(session_id)
    return {
        "session_id": session_id,
        "event_count": len(events),
        "events": [e.to_dict() for e in events[-100:]],  # Last 100
        "decision_log": bus.get_decision_log(session_id),
    }


@app.post("/simulate")
async def simulate(req: SimulateRequest):
    """Run a full demo scenario end-to-end."""
    manager = get_session_manager()

    try:
        sim = get_demo_scenario(req.scenario, interval_s=req.interval_s)
    except ValueError as e:
        raise HTTPException(400, str(e))

    session_id = manager.create_session()
    results = []

    async for chunk in sim.stream():
        result = await manager.process_chunk(session_id, chunk)
        results.append(result)

    return {
        "session_id": session_id,
        "scenario": req.scenario,
        "chunk_count": len(results),
        "results": results,
    }


# ──────────────────────────────────────────────────────────────
# WebSocket
# ──────────────────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_stream(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for live transcript streaming.

    Client sends JSON: {"text": "...", "chunk_index": 0, "is_final": false}
    Server responds with controller decision + retrieval result.
    """
    await websocket.accept()
    manager = get_session_manager()

    # Ensure session exists
    if manager.get_session(session_id) is None:
        manager.create_session()

    # Subscribe to telemetry for this session
    async def send_telemetry(event: TelemetryEvent):
        try:
            if event.session_id == session_id:
                await websocket.send_text(
                    json.dumps({"type": "telemetry", "event": event.to_dict()})
                )
        except Exception:
            pass

    bus = get_telemetry_bus()
    bus.add_async_subscriber(send_telemetry)

    try:
        import time
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)

            chunk = TranscriptChunk(
                chunk_index=data.get("chunk_index", 0),
                text=data.get("text", ""),
                delta_text=data.get("delta_text", data.get("text", "")),
                session_id=session_id,
                timestamp=time.time(),
                is_final=data.get("is_final", False),
            )

            result = await manager.process_chunk(session_id, chunk)
            await websocket.send_text(
                json.dumps({"type": "chunk_result", "data": result})
            )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: %s", session_id)
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
        await websocket.close(code=1011)
