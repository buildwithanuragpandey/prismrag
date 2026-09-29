import json
from pathlib import Path

from adaptive_agentic_rag.agents.query_router import (
    QueryRouter,
)

from adaptive_agentic_rag.agents.query_rewriter import (
    QueryRewriter,
)

from adaptive_agentic_rag.agents.evidence_grader import (
    EvidenceGrader,
)

from adaptive_agentic_rag.retrieval.adaptive_retriever import (
    AdaptiveRetriever,
)

from adaptive_agentic_rag.generation.context_builder import (
    ContextBuilder,
)


# ============================================================
# Configuration
# ============================================================

FROZEN_PATH = Path(
    "evaluation/datasets/frozen_eval_500.json"
)


TARGET_IDS = {
    "eval_00d16f93f9b3",  # Sam Altman
    "eval_02f51dafcc4a",  # Google complex
    "eval_04b92669fb3e",  # SBF
}


SYSTEMS = {
    "V1": {
        "collection_name":
            "multihop_chunks",

        "bm25_corpus_path":
            (
                "data/processed/"
                "processed_corpus.json"
            ),
    },

    "V2-A": {
        "collection_name":
            "multihop_chunks_v2",

        "bm25_corpus_path":
            (
                "data/processed/"
                "processed_corpus_v2.json"
            ),
    },
}


# ============================================================
# Helpers
# ============================================================

def unique_preserve_order(
    values,
):

    seen = set()
    output = []

    for value in values:

        if not value:
            continue

        if value in seen:
            continue

        seen.add(value)
        output.append(value)

    return output


def gold_document_ids(
    example,
):

    return unique_preserve_order(

        evidence["document_id"]

        for evidence
        in example.get(
            "evidence",
            []
        )
    )


def document_recall(
    predicted,
    gold,
):

    gold_set = set(
        gold
    )

    if not gold_set:
        return 0.0

    return (
        len(
            set(predicted)
            &
            gold_set
        )
        /
        len(
            gold_set
        )
    )


def extract_result_documents(
    results,
):

    return unique_preserve_order(

        result.get(
            "document_id"
        )

        for result
        in results
    )


def extract_context_documents(
    context,
):

    if context is None:
        return []

    items = (
        getattr(
            context,
            "items",
            []
        )
        or []
    )

    return unique_preserve_order(

        getattr(
            item,
            "document_id",
            None
        )

        for item
        in items
    )


def print_ranked_results(
    results,
    gold,
):

    gold_set = set(
        gold
    )

    for rank, item in enumerate(
        results,
        start=1,
    ):

        document_id = item.get(
            "document_id"
        )

        chunk_id = item.get(
            "id"
        )

        marker = (
            "GOLD"
            if document_id
            in gold_set
            else ""
        )

        print(
            f"{rank:2d}. "
            f"{document_id} "
            f"{chunk_id} "
            f"{marker}"
        )


def print_context(
    context,
    gold,
):

    gold_set = set(
        gold
    )

    items = (
        getattr(
            context,
            "items",
            []
        )
        or []
    )

    for index, item in enumerate(
        items,
        start=1,
    ):

        document_id = getattr(
            item,
            "document_id",
            None
        )

        citation_id = getattr(
            item,
            "citation_id",
            None
        )

        text = getattr(
            item,
            "text",
            ""
        )

        marker = (
            "GOLD"
            if document_id
            in gold_set
            else ""
        )

        print(
            f"{index:2d}. "
            f"citation={citation_id} "
            f"doc={document_id} "
            f"{marker}"
        )

        if text:

            preview = (
                text
                .replace(
                    "\n",
                    " "
                )[:180]
            )

            print(
                "    ",
                preview,
            )


# ============================================================
# One retrieval/evidence attempt
# ============================================================

def run_attempt(
    retriever,
    context_builder,
    evidence_grader,
    *,
    retrieval_query,
    original_query,
    query_type,
    gold,
    label,
):

    print(
        "\n"
        +
        "-" * 100
    )

    print(
        label
    )

    print(
        "-" * 100
    )

    print(
        "Retrieval query:"
    )

    print(
        retrieval_query
    )


    output = retriever.search(
        retrieval_query,
        top_k=10,
    )


    results = output[
        "results"
    ]


    retrieved_documents = (
        extract_result_documents(
            results
        )
    )


    retrieved_recall = (
        document_recall(
            retrieved_documents,
            gold,
        )
    )


    print(
        "\nRETRIEVAL"
    )

    print(
        "Gold recall:",
        round(
            retrieved_recall,
            4,
        )
    )

    print(
        "Unique retrieved docs:",
        len(
            retrieved_documents
        )
    )


    print_ranked_results(
        results,
        gold,
    )


    # ========================================================
    # Context
    # ========================================================

    context = (
        context_builder.build(

            results,

            # Same behavior as LangGraph:
            # compact context against ORIGINAL query.
            query=original_query,
        )
    )


    context_documents = (
        extract_context_documents(
            context
        )
    )


    context_recall = (
        document_recall(
            context_documents,
            gold,
        )
    )


    print(
        "\nCONTEXT"
    )

    print(
        "Gold recall:",
        round(
            context_recall,
            4,
        )
    )

    print(
        "Unique context docs:",
        len(
            context_documents
        )
    )


    print_context(
        context,
        gold,
    )


    # ========================================================
    # Evidence grader
    # ========================================================

    grade = (
        evidence_grader.grade(

            query=
                original_query,

            context=
                context,

            query_type=
                query_type,
        )
    )


    print(
        "\nEVIDENCE GRADE"
    )

    print(
        "Sufficient:",
        grade.sufficient,
    )

    print(
        "Evidence score:",
        grade.evidence_score,
    )

    print(
        "Reasons:"
    )


    for reason in (
        grade.reasons
        or []
    ):

        print(
            " -",
            reason,
        )


    return {
        "sufficient":
            grade.sufficient,

        "retrieved_recall":
            retrieved_recall,

        "context_recall":
            context_recall,
    }


# ============================================================
# Diagnose one system
# ============================================================

def diagnose_system(
    system_name,
    config,
    examples,
):

    print(
        "\n\n"
        +
        "#" * 110
    )

    print(
        system_name
    )

    print(
        "Dense:",
        config[
            "collection_name"
        ]
    )

    print(
        "BM25:",
        config[
            "bm25_corpus_path"
        ]
    )

    print(
        "#" * 110
    )


    retriever = (
        AdaptiveRetriever(

            collection_name=
                config[
                    "collection_name"
                ],

            bm25_corpus_path=
                config[
                    "bm25_corpus_path"
                ],
        )
    )


    router = QueryRouter()

    rewriter = QueryRewriter()

    context_builder = (
        ContextBuilder()
    )

    evidence_grader = (
        EvidenceGrader()
    )


    try:

        for example in examples:

            original_query = (
                example[
                    "question"
                ]
            )

            gold = (
                gold_document_ids(
                    example
                )
            )


            decision = (
                router.route(
                    original_query
                )
            )


            query_type = (
                decision[
                    "query_type"
                ]
            )


            print(
                "\n\n"
                +
                "=" * 110
            )

            print(
                example[
                    "id"
                ]
            )

            print(
                "=" * 110
            )


            print(
                "Question:"
            )

            print(
                original_query
            )


            print(
                "\nGold docs:",
                gold,
            )

            print(
                "Router:",
                query_type,
                "/",
                decision[
                    "retrieval_strategy"
                ],
            )


            first = (
                run_attempt(

                    retriever,
                    context_builder,
                    evidence_grader,

                    retrieval_query=
                        original_query,

                    original_query=
                        original_query,

                    query_type=
                        query_type,

                    gold=
                        gold,

                    label=
                        "ATTEMPT 0 — ORIGINAL QUERY",
                )
            )


            if first[
                "sufficient"
            ]:

                print(
                    "\nNo rewrite required."
                )

                continue


            rewritten_query = (
                rewriter.rewrite(

                    query=
                        original_query,

                    query_type=
                        query_type,

                    attempt=1,
                )
            )


            run_attempt(

                retriever,
                context_builder,
                evidence_grader,

                retrieval_query=
                    rewritten_query,

                original_query=
                    original_query,

                query_type=
                    query_type,

                gold=
                    gold,

                label=
                    "ATTEMPT 1 — REWRITTEN QUERY",
            )


    finally:

        retriever.close()


# ============================================================
# Main
# ============================================================

def main():

    with open(
        FROZEN_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        frozen = json.load(
            file
        )


    examples = [

        example

        for example
        in frozen

        if example[
            "id"
        ]
        in TARGET_IDS
    ]


    examples.sort(
        key=lambda item:
            item[
                "id"
            ]
    )


    print(
        "Target examples:",
        len(
            examples
        )
    )


    if len(
        examples
    ) != len(
        TARGET_IDS
    ):

        raise RuntimeError(
            "Could not find all target examples."
        )


    for system_name, config in (
        SYSTEMS.items()
    ):

        diagnose_system(
            system_name,
            config,
            examples,
        )


if __name__ == "__main__":
    main()