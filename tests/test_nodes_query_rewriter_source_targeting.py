from adaptive_agentic_rag.orchestration.nodes import (
    RAGNodes,
)


class FakeQueryRewriter:

    def __init__(
        self,
    ):

        self.call = None


    def rewrite(
        self,
        **kwargs,
    ):

        self.call = (
            kwargs
        )


        return (
            "The Age Google market practices 2023"
        )


def test_rewrite_node_passes_source_telemetry_before_reset():

    nodes = (
        object.__new__(
            RAGNodes
        )
    )


    fake_rewriter = (
        FakeQueryRewriter()
    )


    nodes.query_rewriter = (
        fake_rewriter
    )


    state = {
        "original_query":
            (
                "Has Google's market portrayal "
                "remained consistent?"
            ),

        "current_query":
            (
                "Has Google's market portrayal "
                "remained consistent?"
            ),

        "query_type":
            "multihop",

        "retry_count":
            0,

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

        "retrieved_results": [
            {
                "document_id":
                    "doc_old"
            }
        ],

        "context":
            object(),

        "evidence_sufficient":
            False,

        "evidence_score":
            0.78,
    }


    update = (
        nodes.rewrite_query(
            state
        )
    )


    assert (
        fake_rewriter.call[
            "required_sources"
        ]
        ==
        [
            "The Age",
            "The Verge",
            "TechCrunch",
        ]
    )


    assert (
        fake_rewriter.call[
            "covered_sources"
        ]
        ==
        [
            "The Verge",
            "TechCrunch",
        ]
    )


    assert (
        fake_rewriter.call[
            "missing_sources"
        ]
        ==
        [
            "The Age",
        ]
    )


    assert (
        update[
            "current_query"
        ]
        ==
        "The Age Google market practices 2023"
    )


    assert (
        update[
            "retry_count"
        ]
        ==
        1
    )


    assert (
        update[
            "retrieved_results"
        ]
        ==
        []
    )


    assert (
        update[
            "context"
        ]
        is None
    )


    assert (
        update[
            "evidence_reasons"
        ]
        ==
        []
    )