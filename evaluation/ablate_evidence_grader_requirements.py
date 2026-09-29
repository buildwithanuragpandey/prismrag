import json
import math

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from adaptive_agentic_rag.orchestration.nodes import (
    RAGNodes,
)

from adaptive_agentic_rag.agents.evidence_grader import (
    EvidenceGrader,
)

from adaptive_agentic_rag.retrieval.query_decomposer import (
    QueryDecomposer,
)


DATASET_PATH = Path(
    "evaluation/datasets/"
    "frozen_e2e_smoke_20.json"
)


OUTPUT_PATH = Path(
    "evaluation/results/"
    "evidence_grader_requirement_ablation.json"
)


# ============================================================
# Candidate structural policies
#
# IMPORTANT:
#
# These are NOT production thresholds.
#
# They are interpretable offline ablation candidates.
# ============================================================

POLICIES = {
    "strict": {
        "minimum_requirement_coverage": 0.45,
        "required_requirement_fraction": 1.00,
        "minimum_mean_coverage": 0.50,
    },

    "balanced": {
        "minimum_requirement_coverage": 0.40,
        "required_requirement_fraction": 0.75,
        "minimum_mean_coverage": 0.48,
    },

    "relaxed": {
        "minimum_requirement_coverage": 0.35,
        "required_requirement_fraction": 0.67,
        "minimum_mean_coverage": 0.45,
    },
}


# ============================================================
# Requirement diagnostics
# ============================================================

@dataclass
class RequirementSupport:

    requirement: str

    query_terms: list[str]

    critical_terms: list[str]

    best_document_id: str | None

    best_coverage: float

    critical_coverage: float

    required_critical_coverage: float

    anchor_ok: bool


@dataclass
class RequirementAnalysis:

    requirements: list[str]

    supports: list[RequirementSupport]

    global_critical_coverage: float

    global_required_critical_coverage: float

    global_anchor_ok: bool

    unique_documents: int

    chunk_count: int

    required_documents: int

    required_chunks: int


# ============================================================
# Requirement analyzer
#
# Reuses the exact lexical/anchor semantics from EvidenceGrader
# instead of duplicating them.
# ============================================================

class RequirementAnalyzer(
    EvidenceGrader
):

    def __init__(
        self,
    ):

        super().__init__()

        self.decomposer = (
            QueryDecomposer()
        )


    # ========================================================
    # Requirements
    # ========================================================

    def extract_requirements(
        self,
        query: str,
    ) -> list[str]:

        decomposed = (
            self.decomposer.decompose(
                query
            )
        )


        if not decomposed:

            return [
                query
            ]


        # ----------------------------------------------------
        # QueryDecomposer returns:
        #
        # [
        #   original_query,
        #   facet_1,
        #   facet_2,
        #   ...
        # ]
        #
        # For requirement analysis we do NOT want the original
        # giant query to compete with its own facets.
        # ----------------------------------------------------

        if (
            len(
                decomposed
            )
            >
            1
        ):

            requirements = (
                decomposed[
                    1:
                ]
            )

        else:

            requirements = [
                decomposed[
                    0
                ]
            ]


        # ----------------------------------------------------
        # Defensive deduplication
        # ----------------------------------------------------

        output = []

        seen = set()


        for requirement in requirements:

            normalized = (
                " ".join(
                    requirement.split()
                )
            )


            key = (
                normalized.lower()
            )


            if (
                not normalized
                or
                key in seen
            ):

                continue


            seen.add(
                key
            )

            output.append(
                normalized
            )


        return (
            output
            or
            [
                query
            ]
        )


    # ========================================================
    # Aggregate context per document
    #
    # Requirement support should come from one coherent
    # document rather than lexical fragments scattered across
    # the entire context.
    # ========================================================

    @staticmethod
    def document_texts(
        context,
    ) -> dict[str, str]:

        grouped = (
            defaultdict(
                list
            )
        )


        for item in context.items:

            source = (
                getattr(
                    item,
                    "source",
                    "",
                )
                or ""
            )


            title = (
                getattr(
                    item,
                    "title",
                    "",
                )
                or ""
            )


            text = (
                getattr(
                    item,
                    "text",
                    "",
                )
                or ""
            )


            searchable = (
                f"{source}\n"
                f"{title}\n"
                f"{text}"
            )


            grouped[
                item.document_id
            ].append(
                searchable
            )


        return {
            document_id:
                "\n".join(
                    parts
                )

            for (
                document_id,
                parts,
            ) in grouped.items()
        }


    # ========================================================
    # Critical coverage helper
    # ========================================================

    def critical_coverage(
        self,
        text: str,
        critical_terms: set[str],
    ) -> float:

        if not critical_terms:

            return 1.0


        matched = {

            anchor

            for anchor
            in critical_terms

            if self._anchor_present(
                text,
                anchor,
            )
        }


        return (
            len(
                matched
            )
            /
            len(
                critical_terms
            )
        )


    # ========================================================
    # Full analysis
    # ========================================================

    def analyze(
        self,
        query: str,
        context,
        query_type: str,
    ) -> RequirementAnalysis:

        requirements = (
            self.extract_requirements(
                query
            )
        )


        document_texts = (
            self.document_texts(
                context
            )
        )


        (
            required_documents,
            required_chunks,
            _,
        ) = self._requirements(
            query_type
        )


        # ====================================================
        # Global critical-anchor safety remains.
        # ====================================================

        global_critical_terms = (
            self._critical_terms(
                query
            )
        )


        global_critical_coverage = (
            self.critical_coverage(
                context.text,
                global_critical_terms,
            )
        )


        global_required_critical_coverage = (
            self._critical_coverage_requirement(
                query_type,
                global_critical_terms,
            )
        )


        global_anchor_ok = (
            global_critical_coverage
            >=
            global_required_critical_coverage
        )


        supports = []


        # ====================================================
        # Evaluate each requirement against each coherent
        # document and retain the best supporting document.
        # ====================================================

        for requirement in requirements:

            requirement_terms = (
                self._query_terms(
                    requirement
                )
            )


            requirement_critical_terms = (
                self._critical_terms(
                    requirement
                )
            )


            required_critical_coverage = (
                self._critical_coverage_requirement(
                    query_type,
                    requirement_critical_terms,
                )
            )


            best_document_id = None

            best_coverage = 0.0

            best_critical_coverage = 0.0

            best_anchor_ok = (
                not requirement_critical_terms
            )


            # ------------------------------------------------
            # Select document primarily by lexical coverage,
            # then critical-anchor coverage as tie-break.
            # ------------------------------------------------

            best_key = (
                -1.0,
                -1.0,
            )


            for (
                document_id,
                document_text,
            ) in document_texts.items():

                lexical_coverage = (
                    self._coverage(
                        requirement_terms,
                        document_text,
                    )
                )


                critical_coverage = (
                    self.critical_coverage(
                        document_text,
                        requirement_critical_terms,
                    )
                )


                anchor_ok = (
                    critical_coverage
                    >=
                    required_critical_coverage
                )


                candidate_key = (
                    lexical_coverage,
                    critical_coverage,
                )


                if (
                    candidate_key
                    >
                    best_key
                ):

                    best_key = (
                        candidate_key
                    )


                    best_document_id = (
                        document_id
                    )


                    best_coverage = (
                        lexical_coverage
                    )


                    best_critical_coverage = (
                        critical_coverage
                    )


                    best_anchor_ok = (
                        anchor_ok
                    )


            supports.append(
                RequirementSupport(
                    requirement=
                        requirement,

                    query_terms=
                        sorted(
                            requirement_terms
                        ),

                    critical_terms=
                        sorted(
                            requirement_critical_terms
                        ),

                    best_document_id=
                        best_document_id,

                    best_coverage=
                        round(
                            best_coverage,
                            4,
                        ),

                    critical_coverage=
                        round(
                            best_critical_coverage,
                            4,
                        ),

                    required_critical_coverage=
                        round(
                            required_critical_coverage,
                            4,
                        ),

                    anchor_ok=
                        best_anchor_ok,
                )
            )


        unique_documents = len(
            {
                item.document_id
                for item
                in context.items
            }
        )


        return RequirementAnalysis(
            requirements=
                requirements,

            supports=
                supports,

            global_critical_coverage=
                round(
                    global_critical_coverage,
                    4,
                ),

            global_required_critical_coverage=
                round(
                    global_required_critical_coverage,
                    4,
                ),

            global_anchor_ok=
                global_anchor_ok,

            unique_documents=
                unique_documents,

            chunk_count=
                len(
                    context.items
                ),

            required_documents=
                required_documents,

            required_chunks=
                required_chunks,
        )


# ============================================================
# Apply candidate policy
# ============================================================

def apply_policy(
    analysis: RequirementAnalysis,
    policy: dict,
) -> dict:

    requirement_count = len(
        analysis.supports
    )


    if (
        requirement_count
        ==
        0
    ):

        return {
            "sufficient": False,
            "supported_requirements": 0,
            "requirement_count": 0,
            "supported_fraction": 0.0,
            "mean_coverage": 0.0,
            "minimum_coverage": 0.0,
            "required_supported_count": 0,
            "enough_documents": False,
            "enough_chunks": False,
            "global_anchor_ok":
                analysis.global_anchor_ok,
        }


    minimum_requirement_coverage = (
        policy[
            "minimum_requirement_coverage"
        ]
    )


    supported = []


    for support in (
        analysis.supports
    ):

        is_supported = (

            support.best_coverage
            >=
            minimum_requirement_coverage

            and

            support.anchor_ok
        )


        supported.append(
            is_supported
        )


    supported_count = sum(
        1
        for value
        in supported
        if value
    )


    supported_fraction = (
        supported_count
        /
        requirement_count
    )


    coverages = [
        support.best_coverage
        for support
        in analysis.supports
    ]


    mean_coverage = (
        sum(
            coverages
        )
        /
        len(
            coverages
        )
    )


    minimum_coverage = min(
        coverages
    )


    required_supported_count = (
        math.ceil(
            requirement_count
            *
            policy[
                "required_requirement_fraction"
            ]
        )
    )


    enough_documents = (

        analysis.unique_documents

        >=

        analysis.required_documents
    )


    enough_chunks = (

        analysis.chunk_count

        >=

        analysis.required_chunks
    )


    enough_requirements = (

        supported_count

        >=

        required_supported_count
    )


    enough_mean_coverage = (

        mean_coverage

        >=

        policy[
            "minimum_mean_coverage"
        ]
    )


    sufficient = (

        enough_documents

        and

        enough_chunks

        and

        analysis.global_anchor_ok

        and

        enough_requirements

        and

        enough_mean_coverage
    )


    return {
        "sufficient":
            sufficient,

        "supported_requirements":
            supported_count,

        "requirement_count":
            requirement_count,

        "supported_fraction":
            round(
                supported_fraction,
                4,
            ),

        "mean_coverage":
            round(
                mean_coverage,
                4,
            ),

        "minimum_coverage":
            round(
                minimum_coverage,
                4,
            ),

        "required_supported_count":
            required_supported_count,

        "enough_documents":
            enough_documents,

        "enough_chunks":
            enough_chunks,

        "global_anchor_ok":
            analysis.global_anchor_ok,
    }


# ============================================================
# Gold-context diagnostics
# ============================================================

def unique_document_ids(
    context,
) -> list[str]:

    output = []


    for item in context.items:

        if (
            item.document_id
            not in output
        ):

            output.append(
                item.document_id
            )


    return output


def gold_recall(
    predicted_ids,
    gold_ids,
):

    gold = set(
        gold_ids
        or []
    )


    if not gold:

        return None


    predicted = set(
        predicted_ids
        or []
    )


    return (
        len(
            gold
            &
            predicted
        )
        /
        len(
            gold
        )
    )


# ============================================================
# One case
# ============================================================

def run_case(
    nodes,
    analyzer,
    example,
):

    question = (
        example[
            "question"
        ]
    )


    state = {
        "original_query":
            question,

        "current_query":
            question,

        "retry_count":
            0,
    }


    state.update(
        nodes.route_query(
            state
        )
    )


    state.update(
        nodes.retrieve(
            state
        )
    )


    state.update(
        nodes.build_context(
            state
        )
    )


    query_type = (
        state[
            "query_type"
        ]
    )


    context = (
        state[
            "context"
        ]
    )


    # ========================================================
    # V2 baseline
    # ========================================================

    v2_grade = (
        nodes.evidence_grader.grade(
            query=
                question,
            context=
                context,
            query_type=
                query_type,
        )
    )


    # ========================================================
    # Requirement analysis
    # ========================================================

    requirement_analysis = (
        analyzer.analyze(
            query=
                question,
            context=
                context,
            query_type=
                query_type,
        )
    )


    policy_results = {}


    for (
        policy_name,
        policy,
    ) in POLICIES.items():

        policy_results[
            policy_name
        ] = (
            apply_policy(
                analysis=
                    requirement_analysis,
                policy=
                    policy,
            )
        )


    context_document_ids = (
        unique_document_ids(
            context
        )
    )


    gold_document_ids = (
        example.get(
            "evidence_document_ids",
            [],
        )
    )


    context_gold_recall = (
        gold_recall(
            predicted_ids=
                context_document_ids,
            gold_ids=
                gold_document_ids,
        )
    )


    return {
        "id":
            example[
                "id"
            ],

        "question":
            question,

        "question_type":
            example[
                "question_type"
            ],

        "router_query_type":
            query_type,

        "gold_answer":
            example.get(
                "answer"
            ),

        "gold_document_ids":
            gold_document_ids,

        "context_document_ids":
            context_document_ids,

        "context_gold_recall":
            context_gold_recall,

        "context_gold_complete": (
            (
                context_gold_recall
                ==
                1.0
            )
            if (
                context_gold_recall
                is not None
            )
            else None
        ),

        "v2": {
            "sufficient":
                v2_grade.sufficient,

            "score":
                v2_grade.evidence_score,

            "query_term_coverage":
                v2_grade.query_term_coverage,

            "reasons":
                v2_grade.reasons,
        },

        "requirements": [
            {
                "requirement":
                    support.requirement,

                "query_terms":
                    support.query_terms,

                "critical_terms":
                    support.critical_terms,

                "best_document_id":
                    support.best_document_id,

                "best_coverage":
                    support.best_coverage,

                "critical_coverage":
                    support.critical_coverage,

                "required_critical_coverage":
                    support.required_critical_coverage,

                "anchor_ok":
                    support.anchor_ok,
            }

            for support
            in requirement_analysis.supports
        ],

        "global_critical_coverage":
            (
                requirement_analysis
                .global_critical_coverage
            ),

        "global_required_critical_coverage":
            (
                requirement_analysis
                .global_required_critical_coverage
            ),

        "global_anchor_ok":
            (
                requirement_analysis
                .global_anchor_ok
            ),

        "policies":
            policy_results,
    }


# ============================================================
# Summary helpers
# ============================================================

def summarize_decisions(
    records,
    decision_getter,
):

    answerable = [
        record
        for record
        in records
        if (
            record[
                "question_type"
            ]
            !=
            "null_query"
        )
    ]


    null_examples = [
        record
        for record
        in records
        if (
            record[
                "question_type"
            ]
            ==
            "null_query"
        )
    ]


    accepted_answerable = [
        record
        for record
        in answerable
        if (
            decision_getter(
                record
            )
        )
    ]


    rejected_answerable = [
        record
        for record
        in answerable
        if not (
            decision_getter(
                record
            )
        )
    ]


    null_false_accepts = [
        record
        for record
        in null_examples
        if (
            decision_getter(
                record
            )
        )
    ]


    null_rejects = [
        record
        for record
        in null_examples
        if not (
            decision_getter(
                record
            )
        )
    ]


    complete_gold_rejected = [
        record
        for record
        in answerable
        if (
            record[
                "context_gold_complete"
            ]
            is True

            and

            not decision_getter(
                record
            )
        )
    ]


    high_gold_rejected = [
        record
        for record
        in answerable
        if (
            record[
                "context_gold_recall"
            ]
            is not None

            and

            record[
                "context_gold_recall"
            ]
            >=
            0.75

            and

            not decision_getter(
                record
            )
        )
    ]


    by_type = {}


    for question_type in sorted(
        {
            record[
                "question_type"
            ]
            for record
            in records
        }
    ):

        subset = [
            record
            for record
            in records
            if (
                record[
                    "question_type"
                ]
                ==
                question_type
            )
        ]


        accepted = sum(
            1
            for record
            in subset
            if (
                decision_getter(
                    record
                )
            )
        )


        by_type[
            question_type
        ] = {
            "count":
                len(
                    subset
                ),

            "accepted":
                accepted,

            "rejected":
                (
                    len(
                        subset
                    )
                    -
                    accepted
                ),
        }


    return {
        "answerable_accepts":
            len(
                accepted_answerable
            ),

        "answerable_rejects":
            len(
                rejected_answerable
            ),

        "answerable_accept_rate": (
            len(
                accepted_answerable
            )
            /
            len(
                answerable
            )
            if answerable
            else None
        ),

        "complete_gold_context_rejected":
            len(
                complete_gold_rejected
            ),

        "high_gold_context_rejected":
            len(
                high_gold_rejected
            ),

        "null_rejects":
            len(
                null_rejects
            ),

        "null_false_accepts":
            len(
                null_false_accepts
            ),

        "null_reject_rate": (
            len(
                null_rejects
            )
            /
            len(
                null_examples
            )
            if null_examples
            else None
        ),

        "by_question_type":
            by_type,
    }


def summarize(
    records,
):

    output = {}


    output[
        "v2"
    ] = summarize_decisions(
        records=
            records,

        decision_getter=lambda record:
            record[
                "v2"
            ][
                "sufficient"
            ],
    )


    for policy_name in POLICIES:

        output[
            policy_name
        ] = summarize_decisions(
            records=
                records,

            decision_getter=lambda record, name=policy_name:
                record[
                    "policies"
                ][
                    name
                ][
                    "sufficient"
                ],
        )


    return output


# ============================================================
# Console comparison
# ============================================================

def print_case(
    record,
):

    print(
        "\n"
        +
        "=" * 100
    )


    print(
        record[
            "id"
        ],
        "|",
        record[
            "question_type"
        ],
    )


    print(
        record[
            "question"
        ]
    )


    print(
        "\nContext gold recall:",
        record[
            "context_gold_recall"
        ],
    )


    print(
        "\nV2:"
    )


    print(
        (
            "  sufficient="
            f"{record['v2']['sufficient']} "
            "score="
            f"{record['v2']['score']:.4f}"
        )
    )


    print(
        "\nRequirements:"
    )


    for (
        index,
        requirement,
    ) in enumerate(
        record[
            "requirements"
        ],
        start=1,
    ):

        print(
            (
                f"  R{index}: "
                f"{requirement['requirement']}"
            )
        )


        print(
            (
                "      best_doc="
                f"{requirement['best_document_id']} "
                "coverage="
                f"{requirement['best_coverage']:.4f} "
                "critical="
                f"{requirement['critical_coverage']:.4f}/"
                f"{requirement['required_critical_coverage']:.4f} "
                "anchor_ok="
                f"{requirement['anchor_ok']}"
            )
        )


    print(
        "\nPolicies:"
    )


    for policy_name in POLICIES:

        result = (
            record[
                "policies"
            ][
                policy_name
            ]
        )


        print(
            (
                f"  {policy_name:8s} "
                f"sufficient="
                f"{result['sufficient']} | "
                f"requirements="
                f"{result['supported_requirements']}/"
                f"{result['requirement_count']} | "
                f"mean="
                f"{result['mean_coverage']:.4f} | "
                f"min="
                f"{result['minimum_coverage']:.4f}"
            )
        )


# ============================================================
# Main
# ============================================================

def main():

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        examples = json.load(
            file
        )


    nodes = (
        RAGNodes()
    )


    analyzer = (
        RequirementAnalyzer()
    )


    records = []


    try:

        for (
            index,
            example,
        ) in enumerate(
            examples,
            start=1,
        ):

            print(
                (
                    f"\nRunning "
                    f"{index}/"
                    f"{len(examples)}..."
                )
            )


            record = (
                run_case(
                    nodes=
                        nodes,
                    analyzer=
                        analyzer,
                    example=
                        example,
                )
            )


            records.append(
                record
            )


            print_case(
                record
            )


    finally:

        nodes.close()


    summary = (
        summarize(
            records
        )
    )


    print(
        "\n\n"
        +
        "=" * 100
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 100
    )


    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )


    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "policies":
                    POLICIES,

                "summary":
                    summary,

                "records":
                    records,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )


    print(
        "\nSaved:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":

    main()