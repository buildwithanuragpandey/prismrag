"""
Unit tests for PRISM core components (no external dependencies required).

Tests:
  - TranscriptSimulator
  - QueryStateTracker
  - SemanticNoveltyDetector (with mock embedding)
  - IntentStabilityTracker
  - QueryCompletenessEstimator
  - RetrievalUtilityEstimator (WeightedLinearPolicy)
  - AdaptiveRetrievalController (WAIT/RETRIEVE/SUPPRESS)
  - MultiIntentDecomposer (heuristic)
  - EvidenceStore
  - AnswerVersionManager
  - LateConstraintDetector
  - ClaimImpactAnalyzer
  - CitationValidator
  - GroundingChecker
  - PRISM gates G1-G6
  - Baselines
"""

import sys
import os
from pathlib import Path
import asyncio

# Ensure src is importable
_PRISM_SRC = Path(__file__).parents[2] / "src"
if str(_PRISM_SRC) not in sys.path:
    sys.path.insert(0, str(_PRISM_SRC))

import pytest


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def mock_embedding(text: str):
    """Simple mock embedding: one-hot vector based on text length."""
    dim = 64
    vec = [0.0] * dim
    for i, ch in enumerate(text[:dim]):
        vec[i % dim] += ord(ch) / 255.0
    # Normalize
    norm = sum(x**2 for x in vec) ** 0.5
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


# ──────────────────────────────────────────────────────────────
# Streaming Simulator
# ──────────────────────────────────────────────────────────────

class TestTranscriptSimulator:
    def test_cumulative_chunks(self):
        from prism.streaming import TranscriptSimulator
        fragments = ["Hello", "Hello world", "Hello world how are you"]
        sim = TranscriptSimulator(fragments, interval_s=0.0, cumulative=True)
        chunks = list(sim.stream_sync())
        assert len(chunks) == 3
        assert chunks[0].text == "Hello"
        assert chunks[1].text == "Hello world"
        assert chunks[2].text == "Hello world how are you"
        assert chunks[2].is_final

    def test_incremental_chunks(self):
        from prism.streaming import TranscriptSimulator
        fragments = ["Hello", " world", " goodbye"]
        sim = TranscriptSimulator(fragments, interval_s=0.0, cumulative=False)
        chunks = list(sim.stream_sync())
        assert "Hello" in chunks[0].text
        assert chunks[2].is_final

    def test_session_id_assigned(self):
        from prism.streaming import TranscriptSimulator
        sim = TranscriptSimulator(["test"], interval_s=0.0)
        chunks = list(sim.stream_sync())
        assert chunks[0].session_id != ""

    def test_demo_scenario(self):
        from prism.streaming import get_demo_scenario
        sim = get_demo_scenario("workshop_planning", interval_s=0.0)
        chunks = list(sim.stream_sync())
        assert len(chunks) == 4
        assert chunks[-1].is_final


# ──────────────────────────────────────────────────────────────
# Query State Tracker
# ──────────────────────────────────────────────────────────────

class TestQueryStateTracker:
    def test_creates_session(self):
        from prism.controller import QueryStateTracker
        tracker = QueryStateTracker()
        state = tracker.get_or_create("sess-1")
        assert state.session_id == "sess-1"
        assert state.current_text == ""

    def test_update_from_chunk(self):
        from prism.controller import QueryStateTracker
        tracker = QueryStateTracker()
        state = tracker.update_from_chunk("sess-2", 0, "Hello world")
        assert state.current_text == "Hello world"
        assert state.chunk_index == 0
        state2 = tracker.update_from_chunk("sess-2", 1, "Hello world again")
        assert state2.previous_text == "Hello world"
        assert state2.current_text == "Hello world again"

    def test_apply_signals(self):
        from prism.controller import QueryStateTracker, DetectedIntent
        tracker = QueryStateTracker()
        state = tracker.update_from_chunk("sess-3", 0, "venue for 30 people")
        intents = [DetectedIntent(label="venue_selection", sub_query="venue 30 people")]
        tracker.apply_signals(state, 0.8, 0.7, 0.5, 0.9, intents)
        assert state.novelty_score == 0.8
        assert state.intent_stability == 0.7
        assert state.completeness == 0.5

    def test_snapshot_stored(self):
        from prism.controller import QueryStateTracker, RetrievalAction
        tracker = QueryStateTracker()
        state = tracker.update_from_chunk("sess-4", 0, "test query")
        state.snapshot(RetrievalAction.WAIT, "testing")
        assert len(state.history) == 1
        assert state.history[0].action == RetrievalAction.WAIT


# ──────────────────────────────────────────────────────────────
# Semantic Novelty Detector
# ──────────────────────────────────────────────────────────────

class TestSemanticNoveltyDetector:
    def test_first_chunk_is_novel(self):
        from prism.controller import SemanticNoveltyDetector
        det = SemanticNoveltyDetector(mock_embedding)
        score = det.compute("hello world", "")
        assert score == 1.0

    def test_identical_text_is_not_novel(self):
        from prism.controller import SemanticNoveltyDetector
        det = SemanticNoveltyDetector(mock_embedding)
        score = det.compute("hello world", "hello world")
        assert score == 0.0

    def test_different_text_has_novelty(self):
        from prism.controller import SemanticNoveltyDetector
        det = SemanticNoveltyDetector(mock_embedding)
        score = det.compute(
            "I need to plan a workshop for 30 people",
            "I need to plan"
        )
        assert 0.0 < score <= 1.0

    def test_empty_text(self):
        from prism.controller import SemanticNoveltyDetector
        det = SemanticNoveltyDetector(mock_embedding)
        assert det.compute("", "anything") == 0.0


# ──────────────────────────────────────────────────────────────
# Intent Stability Tracker
# ──────────────────────────────────────────────────────────────

class TestIntentStabilityTracker:
    def test_no_stability_with_one_chunk(self):
        from prism.controller import IntentStabilityTracker
        tracker = IntentStabilityTracker(stable_window=2)
        tracker.update("s1", ["venue_selection"])
        assert tracker.compute_stability("s1") == 0.0  # Need >= 2 chunks

    def test_stable_with_same_intent(self):
        from prism.controller import IntentStabilityTracker
        tracker = IntentStabilityTracker(stable_window=2)
        tracker.update("s2", ["venue_selection"])
        score = tracker.update("s2", ["venue_selection"])
        assert score == 1.0

    def test_low_stability_different_intents(self):
        from prism.controller import IntentStabilityTracker
        tracker = IntentStabilityTracker(stable_window=2)
        tracker.update("s3", ["venue_selection"])
        score = tracker.update("s3", ["cancellation_policy"])
        assert score < 1.0

    def test_is_stable(self):
        from prism.controller import IntentStabilityTracker
        tracker = IntentStabilityTracker(stable_window=2)
        tracker.update("s4", ["intent_a"])
        tracker.update("s4", ["intent_a"])
        assert tracker.is_stable("s4", threshold=0.5)


# ──────────────────────────────────────────────────────────────
# Completeness Estimator
# ──────────────────────────────────────────────────────────────

class TestCompletenessEstimator:
    def test_empty_returns_zero(self):
        from prism.controller.completeness import estimate_completeness
        assert estimate_completeness("") == 0.0

    def test_short_text_low_completeness(self):
        from prism.controller.completeness import estimate_completeness
        score = estimate_completeness("plan")
        assert score < 0.4

    def test_long_query_higher_completeness(self):
        from prism.controller.completeness import estimate_completeness
        score = estimate_completeness(
            "I need to plan a customer workshop for 30 people including cancellation policy and catering"
        )
        assert score > 0.5


# ──────────────────────────────────────────────────────────────
# Retrieval Utility Estimator
# ──────────────────────────────────────────────────────────────

class TestRetrievalUtilityEstimator:
    def test_high_signals_give_high_utility(self):
        from prism.controller import RetrievalUtilityEstimator
        est = RetrievalUtilityEstimator()
        result = est.compute_from_components(
            novelty_score=0.9,
            intent_stability=0.9,
            completeness=0.8,
            evidence_gap=0.9,
        )
        assert result.utility > 0.5

    def test_low_signals_give_low_utility(self):
        from prism.controller import RetrievalUtilityEstimator
        est = RetrievalUtilityEstimator()
        result = est.compute_from_components(
            novelty_score=0.0,
            intent_stability=0.0,
            completeness=0.0,
            evidence_gap=0.0,
        )
        assert result.utility < 0.2

    def test_utility_clamped_0_1(self):
        from prism.controller import RetrievalUtilityEstimator
        est = RetrievalUtilityEstimator()
        result = est.compute_from_components(
            novelty_score=2.0,   # Out of bounds input
            intent_stability=2.0,
            completeness=2.0,
            evidence_gap=2.0,
        )
        assert 0.0 <= result.utility <= 1.0

    def test_breakdown_keys_present(self):
        from prism.controller import RetrievalUtilityEstimator
        est = RetrievalUtilityEstimator()
        result = est.compute_from_components()
        assert "novelty_contribution" in result.breakdown


# ──────────────────────────────────────────────────────────────
# Adaptive Controller
# ──────────────────────────────────────────────────────────────

class TestAdaptiveController:
    def _make_state(self, text, novelty=0.8, stability=0.7, completeness=0.6):
        from prism.controller import QueryStateTracker, DetectedIntent
        tracker = QueryStateTracker()
        import uuid
        sid = str(uuid.uuid4())
        state = tracker.update_from_chunk(sid, 0, text)
        tracker.apply_signals(state, novelty, stability, completeness, 0.8, [
            DetectedIntent(label="test_intent", sub_query=text)
        ])
        return state

    def test_wait_on_low_completeness(self):
        from prism.controller import AdaptiveRetrievalController, ControllerConfig, RetrievalAction
        ctrl = AdaptiveRetrievalController(
            config=ControllerConfig(min_completeness_to_retrieve=0.9)
        )
        state = self._make_state("plan", novelty=0.1, stability=0.3, completeness=0.1)
        decision = ctrl.decide(state)
        assert decision.action == RetrievalAction.WAIT

    def test_retrieve_on_high_signals(self):
        from prism.controller import AdaptiveRetrievalController, ControllerConfig, RetrievalAction
        ctrl = AdaptiveRetrievalController(
            config=ControllerConfig(retrieve_threshold=0.5, min_completeness_to_retrieve=0.3)
        )
        state = self._make_state(
            "venue for 30 people cancellation catering",
            novelty=0.9, stability=0.9, completeness=0.7
        )
        decision = ctrl.decide(state, evidence_coverage=0.0)
        assert decision.action == RetrievalAction.RETRIEVE

    def test_suppress_on_reformulation(self):
        from prism.controller import AdaptiveRetrievalController, RetrievalAction
        ctrl = AdaptiveRetrievalController()
        state = self._make_state("please repeat your previous answer in bullet points")
        decision = ctrl.decide(state)
        assert decision.action == RetrievalAction.SUPPRESS

    def test_suppress_when_evidence_sufficient(self):
        from prism.controller import AdaptiveRetrievalController, ControllerConfig, RetrievalAction
        ctrl = AdaptiveRetrievalController(
            config=ControllerConfig(evidence_sufficient_threshold=0.70)
        )
        state = self._make_state("same question", novelty=0.05)
        decision = ctrl.decide(state, evidence_coverage=0.95)
        assert decision.action == RetrievalAction.SUPPRESS

    def test_decision_has_all_fields(self):
        from prism.controller import AdaptiveRetrievalController
        ctrl = AdaptiveRetrievalController()
        state = self._make_state("test query")
        decision = ctrl.decide(state)
        d = decision.to_dict()
        assert "action" in d
        assert "reason" in d
        assert "novelty_score" in d
        assert "retrieval_utility" in d

    def test_retrieve_id_set_when_retrieve(self):
        from prism.controller import AdaptiveRetrievalController, ControllerConfig, RetrievalAction
        ctrl = AdaptiveRetrievalController(
            config=ControllerConfig(retrieve_threshold=0.1, min_completeness_to_retrieve=0.0)
        )
        state = self._make_state("test query", novelty=1.0, stability=1.0, completeness=0.9)
        decision = ctrl.decide(state, evidence_coverage=0.0)
        if decision.action == RetrievalAction.RETRIEVE:
            assert decision.retrieval_id is not None
        else:
            assert decision.retrieval_id is None


# ──────────────────────────────────────────────────────────────
# Multi-Intent Decomposer
# ──────────────────────────────────────────────────────────────

class TestMultiIntentDecomposer:
    def test_single_intent(self):
        from prism.intents import MultiIntentDecomposer
        d = MultiIntentDecomposer()
        intents = d.decompose("I need a venue")
        assert len(intents) >= 1
        assert any(i.label == "venue_selection" for i in intents)

    def test_multi_intent(self):
        from prism.intents import MultiIntentDecomposer
        d = MultiIntentDecomposer()
        intents = d.decompose(
            "I need a venue for 30 people with catering and cancellation policy"
        )
        labels = [i.label for i in intents]
        assert "venue_selection" in labels or "capacity" in labels

    def test_empty_returns_empty(self):
        from prism.intents import MultiIntentDecomposer
        d = MultiIntentDecomposer()
        intents = d.decompose("")
        assert intents == []

    def test_max_intents_respected(self):
        from prism.intents import MultiIntentDecomposer
        d = MultiIntentDecomposer(max_intents=2)
        intents = d.decompose(
            "venue catering cancellation budget travel speakers"
        )
        assert len(intents) <= 2


# ──────────────────────────────────────────────────────────────
# Evidence Store
# ──────────────────────────────────────────────────────────────

class TestEvidenceStore:
    def _make_chunk(self):
        from prism.retrieval import RetrievedChunk
        return RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            text="The venue has a capacity of 50 people.",
            score=0.85,
            rerank_score=0.90,
            metadata={"title": "Venue Guide", "source": "venue.com"},
        )

    def test_add_evidence(self):
        from prism.evidence import EvidenceStore
        store = EvidenceStore("sess-ev-1")
        chunk = self._make_chunk()
        items = store.add_from_retrieval("ret-1", "venue query", [chunk], 0)
        assert len(items) == 1
        assert items[0].chunk_id == "chunk-1"

    def test_deduplication(self):
        from prism.evidence import EvidenceStore
        store = EvidenceStore("sess-ev-2")
        chunk = self._make_chunk()
        items1 = store.add_from_retrieval("ret-1", "query", [chunk], 0)
        items2 = store.add_from_retrieval("ret-2", "query", [chunk], 1)  # Same chunk
        assert len(items1) == 1
        assert len(items2) == 0  # Duplicate filtered

    def test_coverage_estimate(self):
        from prism.evidence import EvidenceStore
        store = EvidenceStore("sess-ev-3")
        chunk = self._make_chunk()
        store.add_from_retrieval("ret-1", "venue", [chunk], 0)
        coverage = store.estimate_coverage("venue capacity people")
        assert coverage > 0.0

    def test_get_chunk_ids(self):
        from prism.evidence import EvidenceStore
        store = EvidenceStore("sess-ev-4")
        chunk = self._make_chunk()
        store.add_from_retrieval("ret-1", "query", [chunk], 0)
        ids = store.get_chunk_ids()
        assert "chunk-1" in ids


# ──────────────────────────────────────────────────────────────
# Answer Versioning
# ──────────────────────────────────────────────────────────────

class TestAnswerVersioning:
    def _make_claims(self, texts):
        from prism.session import Claim
        import uuid
        return [Claim(claim_id=str(uuid.uuid4()), text=t) for t in texts]

    def test_add_version(self):
        from prism.session import AnswerVersionManager
        mgr = AnswerVersionManager("sess-v-1")
        claims = self._make_claims(["Paris is the capital.", "France is in Europe."])
        v = mgr.add_version("Paris is the capital of France.", claims, ["ev-1"])
        assert v.version == 1
        assert mgr.version_count == 1

    def test_version_diff(self):
        from prism.session import AnswerVersionManager
        mgr = AnswerVersionManager("sess-v-2")
        claims1 = self._make_claims(["Claim A.", "Claim B."])
        mgr.add_version("V1 answer", claims1, [])
        claims2 = self._make_claims(["Claim A.", "Claim C."])  # B removed, C added
        v2 = mgr.add_version("V2 answer", claims2, [], trigger="refinement")
        assert "Claim C." in v2.changed_claims
        assert "Claim A." in v2.unchanged_claims


# ──────────────────────────────────────────────────────────────
# Late Constraint Detector
# ──────────────────────────────────────────────────────────────

class TestLateConstraintDetector:
    def _make_claims(self, texts):
        from prism.session import Claim
        import uuid
        return [Claim(claim_id=str(uuid.uuid4()), text=t) for t in texts]

    def test_detects_international_constraint(self):
        from prism.session import LateConstraintDetector
        det = LateConstraintDetector()
        claims = self._make_claims(["Travel reimbursement is $100."])
        constraints = det.detect(
            new_text="Actually the trip was international",
            previous_text="",
            previous_claims=claims,
        )
        assert len(constraints) > 0
        assert any(c.constraint_type == "geographic" for c in constraints)

    def test_no_constraint_on_identical(self):
        from prism.session import LateConstraintDetector
        det = LateConstraintDetector()
        constraints = det.detect("same text", "same text", [])
        assert len(constraints) == 0


# ──────────────────────────────────────────────────────────────
# Grounding Checker
# ──────────────────────────────────────────────────────────────

class TestGroundingChecker:
    def _make_store(self):
        from prism.evidence import EvidenceStore
        from prism.retrieval import RetrievedChunk
        store = EvidenceStore("grounding-sess")
        chunk = RetrievedChunk(
            chunk_id="c1", document_id="d1",
            text="Paris is the capital city of France located in Western Europe.",
            score=0.9,
        )
        store.add_from_retrieval("r1", "q", [chunk], 0)
        return store

    def test_supported_claim(self):
        from prism.generation import GroundingChecker
        checker = GroundingChecker(min_overlap=0.1)
        store = self._make_store()
        result = checker.check_claim("Paris is the capital of France", store)
        assert result.is_supported

    def test_unsupported_claim(self):
        from prism.generation import GroundingChecker
        checker = GroundingChecker(min_overlap=0.4)
        store = self._make_store()
        result = checker.check_claim("Tokyo is an island nation", store)
        assert not result.is_supported


# ──────────────────────────────────────────────────────────────
# Citation Validator
# ──────────────────────────────────────────────────────────────

class TestCitationValidator:
    def _make_store(self):
        from prism.evidence import EvidenceStore
        from prism.retrieval import RetrievedChunk
        store = EvidenceStore("citation-sess")
        chunk = RetrievedChunk(
            chunk_id="chunk-abc", document_id="doc-xyz",
            text="Test content.", score=0.8
        )
        store.add_from_retrieval("r1", "q", [chunk], 0)
        return store

    def test_valid_citation(self):
        from prism.generation import CitationValidator
        val = CitationValidator()
        store = self._make_store()
        result = val.validate("The answer is X [chunk-abc].", store)
        assert result.valid
        assert len(result.valid_citations) == 1

    def test_invalid_citation(self):
        from prism.generation import CitationValidator
        val = CitationValidator()
        store = self._make_store()
        result = val.validate("The answer is X [nonexistent-id].", store)
        assert not result.valid
        assert len(result.invalid_citations) == 1


# ──────────────────────────────────────────────────────────────
# PRISM Gates
# ──────────────────────────────────────────────────────────────

class TestPRISMGates:
    def test_g1_reproducibility_pass(self):
        from benchmarks.gates.prism_gates import gate_g1_reproducibility
        decisions = [{"action": "RETRIEVE"}, {"action": "WAIT"}]
        result = gate_g1_reproducibility(decisions, decisions)
        assert result.passed
        assert result.score == 1.0

    def test_g1_reproducibility_fail(self):
        from benchmarks.gates.prism_gates import gate_g1_reproducibility
        run1 = [{"action": "RETRIEVE"}, {"action": "WAIT"}]
        run2 = [{"action": "WAIT"}, {"action": "RETRIEVE"}]
        result = gate_g1_reproducibility(run1, run2)
        assert not result.passed

    def test_g2_early_retrieval(self):
        from benchmarks.gates.prism_gates import gate_g2_early_retrieval
        log = [
            {"action": "RETRIEVE", "chunk_index": 1},
            {"action": "WAIT", "chunk_index": 2},
        ]
        result = gate_g2_early_retrieval(log, total_chunks=5, min_early_rate=0.3)
        assert result.passed

    def test_g3_multi_intent(self):
        from benchmarks.gates.prism_gates import gate_g3_multi_intent
        cases = [
            {
                "expected_intents": ["venue", "catering"],
                "detected_intents": ["venue", "catering"],
            }
        ]
        result = gate_g3_multi_intent(cases, min_f1=0.5)
        assert result.passed
        assert result.score == 1.0

    def test_g4_grounding(self):
        from benchmarks.gates.prism_gates import gate_g4_factual_grounding
        reports = [{"grounding_ratio": 0.8}, {"grounding_ratio": 0.9}]
        result = gate_g4_factual_grounding(reports, min_grounding_ratio=0.7)
        assert result.passed

    def test_g6_telemetry(self):
        from benchmarks.gates.prism_gates import gate_g6_telemetry
        events = [
            {"event_type": "retrieval_decision"},
            {"event_type": "retrieval_complete"},
            {"event_type": "evidence_added"},
            {"event_type": "generation_complete"},
        ]
        result = gate_g6_telemetry(events)
        assert result.passed


# ──────────────────────────────────────────────────────────────
# Baselines
# ──────────────────────────────────────────────────────────────

class TestBaselines:
    def _make_state(self, text):
        from prism.controller import QueryStateTracker
        import uuid
        sid = str(uuid.uuid4())
        tracker = QueryStateTracker()
        return tracker.update_from_chunk(sid, 0, text)

    def test_conventional_rag_waits(self):
        from benchmarks.baselines.controllers import ConventionalRAG
        from prism.controller import RetrievalAction
        ctrl = ConventionalRAG()
        state = self._make_state("partial query")
        decision = ctrl.decide(state, is_final=False)
        assert decision.action == RetrievalAction.WAIT

    def test_conventional_rag_retrieves_on_final(self):
        from benchmarks.baselines.controllers import ConventionalRAG
        from prism.controller import RetrievalAction
        ctrl = ConventionalRAG()
        state = self._make_state("complete query")
        decision = ctrl.decide(state, is_final=True)
        assert decision.action == RetrievalAction.RETRIEVE

    def test_naive_streaming_always_retrieves(self):
        from benchmarks.baselines.controllers import NaiveStreamingRAG
        from prism.controller import RetrievalAction
        ctrl = NaiveStreamingRAG()
        for text in ["I", "need", "a venue"]:
            state = self._make_state(text)
            decision = ctrl.decide(state)
            assert decision.action == RetrievalAction.RETRIEVE

    def test_fixed_threshold_waits_below(self):
        from benchmarks.baselines.controllers import FixedThresholdRAG
        from prism.controller import RetrievalAction
        ctrl = FixedThresholdRAG(completeness_threshold=0.99)  # Very high
        state = self._make_state("I need")
        decision = ctrl.decide(state)
        assert decision.action == RetrievalAction.WAIT


# ──────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────

class TestMetrics:
    def test_recall_at_k(self):
        from benchmarks.metrics.metrics import recall_at_k
        retrieved = ["doc-1", "doc-2", "doc-3", "doc-4", "doc-5"]
        relevant = {"doc-1", "doc-3"}
        assert recall_at_k(retrieved, relevant, k=5) == 1.0
        assert recall_at_k(retrieved, relevant, k=1) == 0.5

    def test_precision_at_k(self):
        from benchmarks.metrics.metrics import precision_at_k
        retrieved = ["doc-1", "doc-x", "doc-3"]
        relevant = {"doc-1", "doc-3"}
        assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(2/3)

    def test_ndcg_perfect(self):
        from benchmarks.metrics.metrics import ndcg_at_k
        retrieved = ["doc-1", "doc-2"]
        relevant = {"doc-1", "doc-2"}
        assert ndcg_at_k(retrieved, relevant, k=2) == pytest.approx(1.0)

    def test_retrieval_savings(self):
        from benchmarks.metrics.metrics import retrieval_savings
        assert retrieval_savings(3, 10) == pytest.approx(0.7)
        assert retrieval_savings(10, 10) == 0.0
        assert retrieval_savings(0, 10) == 1.0

    def test_refinement_efficiency(self):
        from benchmarks.metrics.metrics import refinement_efficiency
        assert refinement_efficiency(50.0, 200.0) == pytest.approx(0.25)

    def test_answer_stability(self):
        from benchmarks.metrics.metrics import answer_stability
        assert answer_stability(8, 10) == 0.8
        assert answer_stability(0, 0) == 1.0
