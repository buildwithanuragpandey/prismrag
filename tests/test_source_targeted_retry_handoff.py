from adaptive_agentic_rag.orchestration.nodes import (
    RAGNodes,
)


class FakeQueryRewriter:

    def rewrite(
        self,
        **kwargs,
    ):

        return (
            "The Age Google market practices 2023"
        )


class FakeRetriever:

    def __init__(
        self,
    ):

        self.call = None


    def search(
        self,
        query,
        top_k,
        target_sources=None,
    ):

        self.call = {
            "query":
                query,

            "top_k":
                top_k,

            "target_sources":
                list(
                    target_sources
                    or []
                ),
        }


        return {
            "decision":
                None,

            "results": [
                {
                    "id":
                        "age_chunk",
                }
            ],
        }


def test_rewrite_to_retrieve_preserves_target_sources():

    nodes = (
        object.__new__(
            RAGNodes
        )
    )


    nodes.query_rewriter = (
        FakeQueryRewriter()
    )


    nodes.retriever = (
        FakeRetriever()
    )


    state = {
        "original_query":
            (
                "Has Google's market practices "
                "remained consistent?"
            ),

        "current_query":
            (
                "Has Google's market practices "
                "remained consistent?"
            ),

        "query_type":
            "multihop",

        "retry_count":
            0,

        "retry_target_sources":
            [],

        "evidence_reasons": [
            (
                "required_sources="
                "['The Age', 'The Verge', 'TechCrunch']"
            ),

            (
                "covered_sources="
                "['The Verge', 'TechCrunch']"
            ),

            (
                "missing_sources="
                "['The Age']"
            ),

            (
                "evidence_path="
                "explicit_source_coverage_reject"
            ),
        ],

        "retrieved_results":
            [],

        "context":
            None,

        "evidence_sufficient":
            False,

        "evidence_score":
            0.78,
    }


    # ========================================================
    # Exactly the same semantics as frozen500 evaluator:
    #
    # rewrite update → merge state
    # ========================================================

    state.update(
        nodes.rewrite_query(
            state
        )
    )


    assert (
        state[
            "retry_target_sources"
        ]
        ==
        [
            "The Age",
        ]
    )


    assert (
        state[
            "current_query"
        ]
        ==
        "The Age Google market practices 2023"
    )


    # ========================================================
    # Then retrieve using updated state.
    # ========================================================

    state.update(
        nodes.retrieve(
            state
        )
    )


    assert (
        nodes.retriever.call[
            "target_sources"
        ]
        ==
        [
            "The Age",
        ]
    )


    assert (
        nodes.retriever.call[
            "query"
        ]
        ==
        "The Age Google market practices 2023"
    )