from types import SimpleNamespace

import numpy as np

from adaptive_agentic_rag.agents.answer_grader import (
    AnswerGrader,
)


class FakeEmbedder:

    def encode_queries(
        self,
        texts,
    ):

        return np.asarray(
            [
                [1.0, 0.0]
            ],
            dtype=float,
        )


    def encode_documents(
        self,
        texts,
    ):

        return np.asarray(
            [
                [1.0, 0.0]
            ],
            dtype=float,
        )


def test_removed_unsupported_draft_claims_do_not_fail_final_answer():

    grader = AnswerGrader(
        embedder=FakeEmbedder()
    )

    generation_result = SimpleNamespace(
        answer=(
            "Yes. [1]\n"
            "- Supported fact. [1]"
        ),
        abstained=False,
        citation_valid=True,
        cited_ids=[1],
        supported_claims=1,
        unsupported_claims=3,
        relevant_claims=1,
    )

    result = grader.grade(
        query="Is the answer yes?",
        generation_result=generation_result,
        evidence_sufficient=True,
    )

    assert result.passed is True

    assert (
        result.supported_claim_ratio
        ==
        1.0
    )


def test_final_answer_without_grounded_claim_fails():

    grader = AnswerGrader(
        embedder=FakeEmbedder()
    )

    generation_result = SimpleNamespace(
        answer="Yes.",
        abstained=False,
        citation_valid=True,
        cited_ids=[],
        supported_claims=0,
        unsupported_claims=0,
        relevant_claims=0,
    )

    result = grader.grade(
        query="Is the answer yes?",
        generation_result=generation_result,
        evidence_sufficient=True,
    )

    assert result.passed is False

    assert (
        result.supported_claim_ratio
        ==
        0.0
    )


def test_insufficient_evidence_abstention_still_passes():

    grader = AnswerGrader(
        embedder=FakeEmbedder()
    )

    generation_result = SimpleNamespace(
        answer="I don't have enough evidence.",
        abstained=True,
        citation_valid=True,
        cited_ids=[],
    )

    result = grader.grade(
        query="Some question",
        generation_result=generation_result,
        evidence_sufficient=False,
    )

    assert result.passed is True

    assert result.correct_abstention is True
