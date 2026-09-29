import pytest

from adaptive_agentic_rag.generation.structured_conclusion_verifier import (
    AnswerTypeGuard,
    RequirementCoverageGuard,
    StructuredConclusionVerifier,
    StructuredVerificationResult,
)
from adaptive_agentic_rag.generation.relation_aware_answer_resolver import (
    RelationAwareAnswerResolver,
)


# ============================================================
# AnswerTypeGuard Tests
# ============================================================

def test_answer_type_guard_detects_publisher_collision_on_organization_query():
    question = (
        "Which organization, discussed in articles from 'The Roar | Sports Writers Blog', "
        "is likely to receive support from Super Rugby franchises?"
    )
    draft_answer = "The Roar | Sports Writers Blog"
    context_sources = ["The Roar | Sports Writers Blog"]

    has_collision, reason = AnswerTypeGuard.check(
        question=question,
        draft_direct_answer=draft_answer,
        context_sources=context_sources,
    )

    assert has_collision is True
    assert "Publisher-as-answer collision" in reason


def test_answer_type_guard_detects_collision_on_alias():
    question = "Which company is facing antitrust scrutiny in reports by The Verge?"
    draft_answer = "The Verge"
    context_sources = ["The Verge | Tech News and Media Network"]

    has_collision, _ = AnswerTypeGuard.check(
        question=question,
        draft_direct_answer=draft_answer,
        context_sources=context_sources,
    )

    assert has_collision is True


def test_answer_type_guard_allows_legitimate_news_source_entity_queries():
    question = (
        "Between the report from 'The Roar | Sports Writers Blog' and 'Sporting News', "
        "which news source reported a larger lead difference?"
    )
    draft_answer = "The Roar | Sports Writers Blog"
    context_sources = ["The Roar | Sports Writers Blog", "Sporting News"]

    has_collision, reason = AnswerTypeGuard.check(
        question=question,
        draft_direct_answer=draft_answer,
        context_sources=context_sources,
    )

    assert has_collision is False
    assert "explicitly requests a news source" in reason


def test_answer_type_guard_allows_valid_target_entities():
    question = "Which organization is encouraged to reinstate funding?"
    draft_answer = "Rugby Australia"
    context_sources = ["The Roar | Sports Writers Blog"]

    has_collision, _ = AnswerTypeGuard.check(
        question=question,
        draft_direct_answer=draft_answer,
        context_sources=context_sources,
    )

    assert has_collision is False


def test_answer_type_guard_ignores_non_entity_queries():
    question = "Does the article suggest that revenue grew?"
    draft_answer = "Yes"
    context_sources = ["TechCrunch"]

    has_collision, _ = AnswerTypeGuard.check(
        question=question,
        draft_direct_answer=draft_answer,
        context_sources=context_sources,
    )

    assert has_collision is False


# ============================================================
# RequirementCoverageGuard Tests
# ============================================================

def test_requirement_coverage_guard_flags_missing_sources():
    guard = RequirementCoverageGuard()
    question = (
        "Does the article from Fortune suggest that Sam Bankman-Fried was responsible, "
        "while the TechCrunch article focuses on Gary Wang's admission of guilt?"
    )
    covered_sources = ["TechCrunch"]

    cov_ok, reason, req_s, cov_s, miss_s = guard.check(
        question=question,
        covered_sources=covered_sources,
    )

    assert cov_ok is False
    assert any("fortune" in m.lower() for m in miss_s)
    assert any("techcrunch" in c.lower() for c in cov_s)


def test_requirement_coverage_guard_passes_when_all_sources_covered():
    guard = RequirementCoverageGuard()
    question = (
        "Does the article from Fortune suggest X, while the TechCrunch article indicates Y?"
    )
    covered_sources = ["Fortune", "TechCrunch"]

    cov_ok, _, req_s, cov_s, miss_s = guard.check(
        question=question,
        covered_sources=covered_sources,
    )

    assert cov_ok is True
    assert len(miss_s) == 0


def test_requirement_coverage_guard_passes_for_single_source_query():
    guard = RequirementCoverageGuard()
    question = "Which company is mentioned by TechCrunch as releasing GPT-4 Turbo?"
    covered_sources = ["TechCrunch"]

    cov_ok, _, req_s, cov_s, miss_s = guard.check(
        question=question,
        covered_sources=covered_sources,
    )

    assert cov_ok is True


# ============================================================
# StructuredConclusionVerifier Orchestration Tests
# ============================================================

def test_structured_conclusion_verifier_exact_consistency_positive():
    verifier = StructuredConclusionVerifier()
    question = (
        "Has the reporting style regarding live score updates and highlights from NFL games by "
        "Sporting News remained consistent between the article featuring \"Jaguars vs. Saints\" and "
        "the one covering \"Chiefs vs. Packers\", considering the excerpts mentioning a player "
        "achieving a first down?"
    )
    draft_answer = "No"
    grounded_facts = [
        "The live score update and highlight excerpt for \"Jaguars vs. Saints\" mentioned a player achieving a first down.",
        "The live score update and highlight excerpt for \"Chiefs vs. Packers\" also mentioned a player achieving a first down.",
    ]
    covered_sources = ["Sporting News"]

    result = verifier.verify(
        query=question,
        draft_direct_answer=draft_answer,
        grounded_facts=grounded_facts,
        covered_sources=covered_sources,
    )

    assert result.applied is True
    assert result.resolved_answer == "Yes"
    assert result.status == "RELATION_RESOLVED"
    assert result.applied_mechanism == "RelationAwareAnswerResolver"


def test_structured_conclusion_verifier_publisher_collision_rejected():
    verifier = StructuredConclusionVerifier()
    question = (
        "Which organization, discussed in articles from 'The Roar | Sports Writers Blog', "
        "is likely to receive support from Super Rugby franchises?"
    )
    draft_answer = "The Roar | Sports Writers Blog"
    grounded_facts = [
        "The Roar discusses support for Rugby Australia to conduct a review.",
    ]
    covered_sources = ["The Roar | Sports Writers Blog"]

    result = verifier.verify(
        query=question,
        draft_direct_answer=draft_answer,
        grounded_facts=grounded_facts,
        covered_sources=covered_sources,
    )

    assert result.applied is True
    assert result.resolved_answer == "UNKNOWN"
    assert result.status == "COLLISION_REJECTED"
    assert result.applied_mechanism == "AnswerTypeGuard"


def test_structured_conclusion_verifier_undercovered_multi_source_rejected():
    verifier = StructuredConclusionVerifier()
    question = (
        "Does the Fortune article suggest that Sam Bankman-Fried was responsible, "
        "while the TechCrunch article focuses on Gary Wang's admission of guilt?"
    )
    draft_answer = "No"
    grounded_facts = [
        "The TechCrunch article focuses on Gary Wang and Caroline Ellison's admission of guilt.",
    ]
    covered_sources = ["TechCrunch"]

    result = verifier.verify(
        query=question,
        draft_direct_answer=draft_answer,
        grounded_facts=grounded_facts,
        covered_sources=covered_sources,
    )

    assert result.applied is True
    assert result.resolved_answer == "UNKNOWN"
    assert result.status == "UNDERCOVERED_ABSTAIN"
    assert result.applied_mechanism == "RequirementCoverageGuard"


def test_structured_conclusion_verifier_valid_draft_preserved():
    verifier = StructuredConclusionVerifier()
    question = "Which company is mentioned by TechCrunch as releasing GPT-4 Turbo?"
    draft_answer = "OpenAI"
    grounded_facts = [
        "TechCrunch mentions OpenAI is releasing GPT-4 Turbo.",
    ]
    covered_sources = ["TechCrunch"]

    result = verifier.verify(
        query=question,
        draft_direct_answer=draft_answer,
        grounded_facts=grounded_facts,
        covered_sources=covered_sources,
    )

    assert result.applied is False
    assert result.resolved_answer == "OpenAI"
    assert result.status == "PRESERVED"
