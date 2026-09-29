"""
PRISM Session Manager

Orchestrates the full session lifecycle:
  - Streaming transcript processing
  - Query state updates
  - Controller decisions
  - Retrieval (hybrid / delta)
  - Evidence management
  - Answer versioning
  - Telemetry emission

This is the central coordinator — every other component is injected
into the SessionManager via dependency injection.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from prism.controller import (
    AdaptiveRetrievalController,
    ControllerConfig,
    ControllerDecision,
    DetectedIntent,
    IntentStabilityTracker,
    QueryState,
    QueryStateTracker,
    RetrievalAction,
    RetrievalUtilityEstimator,
    SemanticNoveltyDetector,
    UtilitySignals,
    estimate_completeness,
)
from prism.evidence import EvidenceStore
from prism.intents import MultiIntentDecomposer
from prism.retrieval import DeltaRetriever, PRISMHybridRetriever
from prism.session.versioning import (
    AnswerVersion,
    AnswerVersionManager,
    ClaimImpactAnalyzer,
    Claim,
    DetectedConstraint,
    LateConstraintDetector,
)
from prism.streaming import TranscriptChunk
from prism.telemetry import EventType, TelemetryBus, TelemetryEvent

logger = logging.getLogger(__name__)


class Session:
    """
    A single user session state container.
    Created per-session by SessionManager.
    """

    def __init__(self, session_id: str, bus: TelemetryBus) -> None:
        self.session_id = session_id
        self.bus = bus
        self.evidence_store = EvidenceStore(session_id)
        self.version_manager = AnswerVersionManager(session_id)
        self.start_time = time.time()

        # Emit session start
        self.bus.emit(
            TelemetryEvent(
                event_type=EventType.SESSION_START,
                session_id=session_id,
                payload={"start_time": self.start_time},
            )
        )

    @property
    def latest_answer(self) -> Optional[AnswerVersion]:
        return self.version_manager.latest


class SessionManager:
    """
    Orchestrates the full PRISM adaptive streaming RAG pipeline.

    Injected dependencies (all configurable):
      - query_state_tracker
      - novelty_detector
      - stability_tracker
      - utility_estimator
      - controller
      - intent_decomposer
      - retriever
      - delta_retriever
      - constraint_detector
      - impact_analyzer
      - llm_fn (callable for generation)
      - telemetry_bus

    Usage:
        manager = SessionManager(...)
        session_id = manager.create_session()
        async for event in manager.process_chunk(session_id, chunk):
            send_to_frontend(event)
    """

    def __init__(
        self,
        query_state_tracker: Optional[QueryStateTracker] = None,
        novelty_detector: Optional[SemanticNoveltyDetector] = None,
        stability_tracker: Optional[IntentStabilityTracker] = None,
        utility_estimator: Optional[RetrievalUtilityEstimator] = None,
        controller: Optional[AdaptiveRetrievalController] = None,
        intent_decomposer: Optional[MultiIntentDecomposer] = None,
        retriever: Optional[PRISMHybridRetriever] = None,
        delta_retriever: Optional[DeltaRetriever] = None,
        constraint_detector: Optional[LateConstraintDetector] = None,
        impact_analyzer: Optional[ClaimImpactAnalyzer] = None,
        llm_fn: Optional[Callable[[str, str], str]] = None,
        telemetry_bus: Optional[TelemetryBus] = None,
    ) -> None:
        self.state_tracker = query_state_tracker or QueryStateTracker()
        self.novelty_detector = novelty_detector
        self.stability_tracker = stability_tracker or IntentStabilityTracker()
        self.utility_estimator = utility_estimator or RetrievalUtilityEstimator()
        self.controller = controller or AdaptiveRetrievalController(
            utility_estimator=self.utility_estimator
        )
        self.intent_decomposer = intent_decomposer or MultiIntentDecomposer()
        self.retriever = retriever
        self.delta_retriever = delta_retriever
        self.constraint_detector = constraint_detector or LateConstraintDetector()
        self.impact_analyzer = impact_analyzer or ClaimImpactAnalyzer()
        self.llm_fn = llm_fn
        self.bus = telemetry_bus or TelemetryBus()

        self._sessions: Dict[str, Session] = {}

    def create_session(self) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = Session(session_id=session_id, bus=self.bus)
        logger.info("Session created: %s", session_id)
        return session_id

    def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    async def process_chunk(
        self, session_id: str, chunk: TranscriptChunk
    ) -> Dict[str, Any]:
        """
        Process a single transcript chunk.

        Returns a dict describing what happened:
          - controller decision
          - retrieval result (if retrieved)
          - updated state
        """
        session = self.get_session(session_id)
        if session is None:
            session_id = self.create_session()
            session = self._sessions[session_id]

        t_start = time.time()

        # 1. Update query state from chunk
        state = self.state_tracker.update_from_chunk(
            session_id=session_id,
            chunk_index=chunk.chunk_index,
            current_text=chunk.text,
        )

        # Emit transcript chunk event
        await self.bus.emit_async(
            TelemetryEvent(
                event_type=EventType.TRANSCRIPT_CHUNK,
                session_id=session_id,
                payload=chunk.to_dict(),
            )
        )

        # 2. Compute signals
        # Novelty
        novelty = 0.0
        if self.novelty_detector is not None:
            novelty = self.novelty_detector.compute(
                current_text=chunk.text,
                previous_text=state.previous_text,
            )
        else:
            # Fallback: simple length ratio novelty
            delta_len = len(chunk.delta_text.split())
            novelty = min(1.0, delta_len / max(1, len(chunk.text.split())))

        # Intent decomposition
        intents: List[DetectedIntent] = self.intent_decomposer.decompose(chunk.text)

        # Stability
        stability = self.stability_tracker.update(
            session_id=session_id,
            intent_labels=[i.label for i in intents],
        )

        # Completeness
        completeness = estimate_completeness(chunk.text, state.previous_text)

        # Evidence coverage
        evidence_coverage = session.evidence_store.estimate_coverage(chunk.text)
        evidence_gap = max(0.0, 1.0 - evidence_coverage)

        # Utility
        utility_result = self.utility_estimator.compute_from_components(
            novelty_score=novelty,
            intent_stability=stability,
            completeness=completeness,
            evidence_gap=evidence_gap,
            retrieval_cost=0.1,
            seconds_since_last_retrieval=state.seconds_since_last_retrieval(),
            chunk_index=chunk.chunk_index,
        )

        # 3. Apply signals to state
        self.state_tracker.apply_signals(
            state=state,
            novelty_score=novelty,
            intent_stability=stability,
            completeness=completeness,
            retrieval_utility=utility_result.utility,
            intents=intents,
        )

        # Emit signals
        await self.bus.emit_async(
            TelemetryEvent(
                event_type=EventType.UTILITY_COMPUTED,
                session_id=session_id,
                payload=utility_result.to_dict(),
                latency_ms=(time.time() - t_start) * 1000,
            )
        )

        # 4. Controller decision
        decision = self.controller.decide(
            state=state,
            evidence_coverage=evidence_coverage,
        )

        # Record snapshot with decision
        state.snapshot(
            action=decision.action,
            action_reason=decision.reason,
            retrieval_id=decision.retrieval_id,
        )

        # Emit decision
        await self.bus.emit_async(
            TelemetryEvent(
                event_type=EventType.RETRIEVAL_DECISION,
                session_id=session_id,
                payload=decision.to_dict(),
                latency_ms=(time.time() - t_start) * 1000,
            )
        )

        result: Dict[str, Any] = {
            "session_id": session_id,
            "chunk_index": chunk.chunk_index,
            "action": decision.action.value,
            "reason": decision.reason,
            "signals": {
                "novelty": round(novelty, 4),
                "intent_stability": round(stability, 4),
                "completeness": round(completeness, 4),
                "evidence_gap": round(evidence_gap, 4),
                "utility": round(utility_result.utility, 4),
            },
            "intents": [i.to_dict() for i in intents],
            "retrieval_result": None,
            "answer": None,
        }

        # 5. Retrieve if decided
        if decision.action == RetrievalAction.RETRIEVE and self.retriever:
            retrieval_result = await self._do_retrieval(
                session=session,
                state=state,
                decision=decision,
                intents=intents,
            )
            result["retrieval_result"] = retrieval_result
            state.record_retrieval(decision.retrieval_id, chunk.text)

            # 6. Generate answer
            if self.llm_fn and session.evidence_store.get_all():
                answer = await self._generate_answer(session=session, query=chunk.text)
                result["answer"] = answer.to_dict() if answer else None

        return result

    async def _do_retrieval(
        self,
        session: Session,
        state: QueryState,
        decision: ControllerDecision,
        intents: List[DetectedIntent],
    ) -> Dict[str, Any]:
        """Execute retrieval and add to evidence store."""
        t_start = time.time()
        sub_queries = [i.sub_query for i in intents if i.sub_query] or [state.current_text]

        await self.bus.emit_async(
            TelemetryEvent(
                event_type=EventType.RETRIEVAL_START,
                session_id=session.session_id,
                payload={"retrieval_id": decision.retrieval_id, "sub_queries": sub_queries},
            )
        )

        if len(sub_queries) > 1 and hasattr(self.retriever, "retrieve_parallel"):
            merged, sub_results = await self.retriever.retrieve_parallel(sub_queries)
            retrieval_result = merged
        else:
            retrieval_result = await self.retriever.retrieve_async(sub_queries[0])

        # Add to evidence store
        new_ev = session.evidence_store.add_from_retrieval(
            retrieval_id=decision.retrieval_id,
            query=state.current_text,
            chunks=retrieval_result.chunks,
            chunk_index=state.chunk_index,
        )

        latency_ms = (time.time() - t_start) * 1000
        await self.bus.emit_async(
            TelemetryEvent(
                event_type=EventType.RETRIEVAL_COMPLETE,
                session_id=session.session_id,
                payload={
                    "retrieval_id": decision.retrieval_id,
                    "chunk_count": len(retrieval_result.chunks),
                    "new_evidence_count": len(new_ev),
                },
                latency_ms=latency_ms,
            )
        )

        return retrieval_result.to_dict()

    async def _generate_answer(
        self, session: Session, query: str
    ) -> Optional[AnswerVersion]:
        """Generate an answer from evidence store."""
        if not self.llm_fn:
            return None

        context = session.evidence_store.to_context_string()
        system_prompt = (
            "You are a precise research assistant. Answer ONLY using the provided evidence. "
            "Cite evidence using [DOC-ID, CHUNK-ID] format inline. "
            "If the evidence is insufficient, say so explicitly. "
            "Do not use outside knowledge."
        )
        user_prompt = f"Evidence:\n{context}\n\nQuestion: {query}"

        try:
            t_start = time.time()
            text = await asyncio.get_event_loop().run_in_executor(
                None, self.llm_fn, system_prompt, user_prompt
            )
            latency_ms = (time.time() - t_start) * 1000

            # Extract simple claims (split by sentence for now)
            claim_sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
            claims = [
                Claim(
                    claim_id=str(uuid.uuid4()),
                    text=s,
                    is_grounded=True,
                    grounding_score=0.7,
                )
                for s in claim_sentences[:10]
            ]

            version = session.version_manager.add_version(
                text=text,
                claims=claims,
                evidence_ids=[e.evidence_id for e in session.evidence_store.get_all()],
                context_used=context[:500],
                trigger="initial" if session.version_manager.version_count == 0 else "retrieval",
            )

            await self.bus.emit_async(
                TelemetryEvent(
                    event_type=EventType.ANSWER_VERSION,
                    session_id=session.session_id,
                    payload={"version": version.version, "trigger": version.trigger},
                    latency_ms=latency_ms,
                )
            )

            return version
        except Exception as exc:
            logger.error("Answer generation failed: %s", exc)
            return None

    async def process_late_constraint(
        self, session_id: str, new_text: str
    ) -> Dict[str, Any]:
        """
        Process new user input after an answer has been given.
        Detects constraints, runs delta retrieval, refines answer.
        """
        session = self.get_session(session_id)
        if session is None:
            return {"error": "Session not found"}

        prev = session.latest_answer
        if prev is None:
            return {"error": "No previous answer to refine"}

        # Detect constraints
        constraints = self.constraint_detector.detect(
            new_text=new_text,
            previous_text=prev.text,
            previous_claims=prev.claims,
        )

        if not constraints:
            return {
                "session_id": session_id,
                "constraint_detected": False,
                "message": "No new constraints detected — no refinement needed",
            }

        await self.bus.emit_async(
            TelemetryEvent(
                event_type=EventType.CONSTRAINT_DETECTED,
                session_id=session_id,
                payload={"constraints": [c.to_dict() for c in constraints]},
            )
        )

        # For each constraint: find affected claims + run delta retrieval
        all_new_chunks = []
        for constraint in constraints:
            affected, unaffected = self.impact_analyzer.analyze(constraint, prev.claims)

            await self.bus.emit_async(
                TelemetryEvent(
                    event_type=EventType.CLAIM_IMPACT,
                    session_id=session_id,
                    payload={
                        "constraint": constraint.to_dict(),
                        "affected_count": len(affected),
                        "unaffected_count": len(unaffected),
                    },
                )
            )

            # Delta retrieval
            if self.delta_retriever:
                delta_result = await self.delta_retriever.retrieve_delta_async(
                    original_query=prev.text[:200],
                    new_constraint=constraint.text,
                    existing_chunk_ids=session.evidence_store.get_chunk_ids(),
                )
                all_new_chunks.extend(delta_result.new_chunks)

                await self.bus.emit_async(
                    TelemetryEvent(
                        event_type=EventType.DELTA_RETRIEVAL_COMPLETE,
                        session_id=session_id,
                        payload=delta_result.to_dict(),
                    )
                )

                # Add new chunks to evidence
                if delta_result.new_chunks:
                    session.evidence_store.add_from_retrieval(
                        retrieval_id=delta_result.delta_retrieval_id,
                        query=constraint.text,
                        chunks=delta_result.new_chunks,
                        chunk_index=-1,
                        intent_label="delta_refinement",
                    )

        # Regenerate answer V2
        new_answer = None
        if self.llm_fn:
            new_answer = await self._generate_answer(session=session, query=new_text)

        return {
            "session_id": session_id,
            "constraint_detected": True,
            "constraints": [c.to_dict() for c in constraints],
            "new_chunks_retrieved": len(all_new_chunks),
            "answer": new_answer.to_dict() if new_answer else None,
        }
