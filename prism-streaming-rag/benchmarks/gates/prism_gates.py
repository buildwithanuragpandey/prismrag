"""
PRISM Evaluation Gates G1-G6

Each gate is a function that takes experimental results and returns
a pass/fail verdict with details.

G1 — Reproducibility:
  Same inputs → same controller decisions

G2 — Early Retrieval:
  System retrieves before utterance is complete

G3 — Multi-Intent Identification:
  Correctly identifies multiple intents in compound queries

G4 — Factual Grounding:
  All answer claims are grounded in retrieved evidence

G5 — Session Refinement:
  Late constraints trigger targeted delta retrieval + answer update
  without full pipeline restart

G6 — Telemetry / Observability:
  Every pipeline event produces structured telemetry
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GateResult:
    gate: str
    passed: bool
    score: float           # 0-1
    details: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate": self.gate,
            "passed": self.passed,
            "score": round(self.score, 4),
            "details": self.details,
            "evidence": self.evidence,
        }


# ──────────────────────────────────────────────────────────────
# G1: Reproducibility
# ──────────────────────────────────────────────────────────────

def gate_g1_reproducibility(
    run1_decisions: List[Dict],
    run2_decisions: List[Dict],
) -> GateResult:
    """
    G1: Reproducibility
    Same transcript chunks → same controller decisions across two identical runs.
    """
    if not run1_decisions or not run2_decisions:
        return GateResult("G1_Reproducibility", False, 0.0, "No decisions to compare")

    n = min(len(run1_decisions), len(run2_decisions))
    matches = sum(
        1 for a, b in zip(run1_decisions[:n], run2_decisions[:n])
        if a.get("action") == b.get("action")
    )
    score = matches / n
    passed = score >= 0.95  # Allow tiny floating point differences

    return GateResult(
        gate="G1_Reproducibility",
        passed=passed,
        score=score,
        details=f"{matches}/{n} decisions match across runs",
        evidence={"match_count": matches, "total": n},
    )


# ──────────────────────────────────────────────────────────────
# G2: Early Retrieval
# ──────────────────────────────────────────────────────────────

def gate_g2_early_retrieval(
    decision_log: List[Dict],
    total_chunks: int,
    min_early_rate: float = 0.30,
) -> GateResult:
    """
    G2: Early Retrieval
    The system must retrieve before the final chunk on at least min_early_rate of queries.
    """
    if not decision_log:
        return GateResult("G2_EarlyRetrieval", False, 0.0, "No decisions logged")

    retrieves = [d for d in decision_log if d.get("action") == "RETRIEVE"]
    if not retrieves:
        return GateResult("G2_EarlyRetrieval", False, 0.0, "No retrievals performed")

    # Early = retrieved before last chunk
    early = [r for r in retrieves if r.get("chunk_index", total_chunks) < total_chunks - 1]
    rate = len(early) / len(retrieves)
    passed = rate >= min_early_rate

    return GateResult(
        gate="G2_EarlyRetrieval",
        passed=passed,
        score=rate,
        details=f"Early retrievals: {len(early)}/{len(retrieves)} (rate={rate:.2f}, min={min_early_rate})",
        evidence={"early_count": len(early), "total_retrieves": len(retrieves)},
    )


# ──────────────────────────────────────────────────────────────
# G3: Multi-Intent Identification
# ──────────────────────────────────────────────────────────────

def gate_g3_multi_intent(
    test_cases: List[Dict],
    min_f1: float = 0.50,
) -> GateResult:
    """
    G3: Multi-Intent Identification
    For compound queries, the system must identify multiple intents.

    test_cases: [{"query": str, "expected_intents": [str], "detected_intents": [str]}]
    """
    if not test_cases:
        return GateResult("G3_MultiIntent", False, 0.0, "No test cases")

    f1_scores = []
    for case in test_cases:
        expected = set(case.get("expected_intents", []))
        detected = set(case.get("detected_intents", []))
        if not expected:
            continue

        tp = len(expected & detected)
        precision = tp / len(detected) if detected else 0.0
        recall = tp / len(expected)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    if not f1_scores:
        return GateResult("G3_MultiIntent", False, 0.0, "No valid test cases")

    avg_f1 = sum(f1_scores) / len(f1_scores)
    passed = avg_f1 >= min_f1

    return GateResult(
        gate="G3_MultiIntent",
        passed=passed,
        score=avg_f1,
        details=f"Average intent F1: {avg_f1:.3f} (min={min_f1})",
        evidence={"n_cases": len(f1_scores), "avg_f1": avg_f1},
    )


# ──────────────────────────────────────────────────────────────
# G4: Factual Grounding
# ──────────────────────────────────────────────────────────────

def gate_g4_factual_grounding(
    grounding_reports: List[Dict],
    min_grounding_ratio: float = 0.70,
) -> GateResult:
    """
    G4: Factual Grounding
    At least min_grounding_ratio of claims must be grounded in evidence.

    grounding_reports: [{"grounding_ratio": float, ...}]
    """
    if not grounding_reports:
        return GateResult("G4_FactualGrounding", False, 0.0, "No grounding reports")

    ratios = [r.get("grounding_ratio", 0.0) for r in grounding_reports]
    avg = sum(ratios) / len(ratios)
    passed = avg >= min_grounding_ratio

    return GateResult(
        gate="G4_FactualGrounding",
        passed=passed,
        score=avg,
        details=f"Average grounding ratio: {avg:.3f} (min={min_grounding_ratio})",
        evidence={"n_answers": len(ratios), "avg_grounding": avg},
    )


# ──────────────────────────────────────────────────────────────
# G5: Session Refinement
# ──────────────────────────────────────────────────────────────

def gate_g5_session_refinement(
    refinement_events: List[Dict],
    min_answer_stability: float = 0.60,
    max_full_retrieval_rate: float = 0.20,
) -> GateResult:
    """
    G5: Session Refinement
    When new constraints arrive, the system must:
    1. Perform delta retrieval (not full retrieval) in most cases
    2. Preserve unaffected claims (answer stability >= threshold)

    refinement_events: [{"used_delta": bool, "answer_stability": float}]
    """
    if not refinement_events:
        return GateResult("G5_SessionRefinement", False, 0.0, "No refinement events")

    delta_count = sum(1 for e in refinement_events if e.get("used_delta", False))
    full_count = len(refinement_events) - delta_count
    delta_rate = delta_count / len(refinement_events)

    stability_values = [e.get("answer_stability", 0.0) for e in refinement_events]
    avg_stability = sum(stability_values) / len(stability_values) if stability_values else 0.0

    passed = (
        delta_rate >= (1.0 - max_full_retrieval_rate)
        and avg_stability >= min_answer_stability
    )

    score = (delta_rate + min(1.0, avg_stability / min_answer_stability)) / 2.0

    return GateResult(
        gate="G5_SessionRefinement",
        passed=passed,
        score=round(score, 4),
        details=(
            f"Delta retrieval rate: {delta_rate:.2f}, "
            f"Answer stability: {avg_stability:.3f}"
        ),
        evidence={
            "delta_rate": delta_rate,
            "avg_stability": avg_stability,
            "n_events": len(refinement_events),
        },
    )


# ──────────────────────────────────────────────────────────────
# G6: Telemetry / Observability
# ──────────────────────────────────────────────────────────────

REQUIRED_EVENT_TYPES = {
    "retrieval_decision",
    "retrieval_complete",
    "evidence_added",
    "generation_complete",
}


def gate_g6_telemetry(
    session_events: List[Dict],
) -> GateResult:
    """
    G6: Telemetry / Observability
    Every required event type must appear in the session telemetry.
    """
    if not session_events:
        return GateResult("G6_Telemetry", False, 0.0, "No telemetry events found")

    found_types = {e.get("event_type", "") for e in session_events}
    missing = REQUIRED_EVENT_TYPES - found_types
    covered = REQUIRED_EVENT_TYPES & found_types

    score = len(covered) / len(REQUIRED_EVENT_TYPES)
    passed = len(missing) == 0

    return GateResult(
        gate="G6_Telemetry",
        passed=passed,
        score=score,
        details=(
            f"Event types found: {len(covered)}/{len(REQUIRED_EVENT_TYPES)}. "
            + (f"Missing: {missing}" if missing else "All required events present.")
        ),
        evidence={
            "found_types": list(found_types),
            "missing_types": list(missing),
            "total_events": len(session_events),
        },
    )


# ──────────────────────────────────────────────────────────────
# Run all gates
# ──────────────────────────────────────────────────────────────

def run_all_gates(results: Dict[str, Any]) -> Dict[str, GateResult]:
    """
    Run all G1-G6 gates from a results dict.
    Returns gate_name → GateResult mapping.
    """
    gates = {}

    if "run1_decisions" in results and "run2_decisions" in results:
        gates["G1"] = gate_g1_reproducibility(
            results["run1_decisions"], results["run2_decisions"]
        )

    if "decision_log" in results and "total_chunks" in results:
        gates["G2"] = gate_g2_early_retrieval(
            results["decision_log"], results["total_chunks"]
        )

    if "intent_test_cases" in results:
        gates["G3"] = gate_g3_multi_intent(results["intent_test_cases"])

    if "grounding_reports" in results:
        gates["G4"] = gate_g4_factual_grounding(results["grounding_reports"])

    if "refinement_events" in results:
        gates["G5"] = gate_g5_session_refinement(results["refinement_events"])

    if "session_events" in results:
        gates["G6"] = gate_g6_telemetry(results["session_events"])

    return gates
