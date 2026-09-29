from dataclasses import dataclass

from adaptive_agentic_rag.retrieval.adaptive_retriever import (
    AdaptiveRetriever
)

from adaptive_agentic_rag.generation.context_builder import (
    ContextBuilder,
    BuiltContext
)

from adaptive_agentic_rag.agents.evidence_grader import (
    EvidenceGrader,
    EvidenceGrade
)

from adaptive_agentic_rag.agents.query_rewriter import (
    QueryRewriter
)


@dataclass
class RetrievalRound:

    round_number: int

    query: str

    decision: dict

    grade: EvidenceGrade


@dataclass
class SelfCorrectionResult:

    original_query: str

    final_query: str

    retrieval_rounds: int

    rewritten: bool

    evidence_sufficient: bool

    context: BuiltContext

    grade: EvidenceGrade

    decision: dict

    history: list[RetrievalRound]


class SelfCorrectionController:


    def __init__(
        self,
        retriever=None,
        context_builder=None,
        evidence_grader=None,
        query_rewriter=None,
        max_retries: int = 1
    ):

        self.retriever = (
            retriever
            if retriever is not None
            else AdaptiveRetriever()
        )


        self.context_builder = (
            context_builder
            if context_builder is not None
            else ContextBuilder()
        )


        self.evidence_grader = (
            evidence_grader
            if evidence_grader is not None
            else EvidenceGrader()
        )


        self.query_rewriter = (
            query_rewriter
            if query_rewriter is not None
            else QueryRewriter()
        )


        self.max_retries = max_retries


    def run(
        self,
        query: str,
        top_k: int = 10
    ) -> SelfCorrectionResult:


        original_query = query

        current_query = query


        original_query_type = None

        history = []


        total_rounds = (
            self.max_retries
            + 1
        )


        for round_index in range(
            total_rounds
        ):

            #
            # Retrieval
            #

            retrieval_output = (
                self.retriever.search(
                    current_query,
                    top_k=top_k
                )
            )


            decision = (
                retrieval_output[
                    "decision"
                ]
            )


            #
            # Keep original intent type
            #

            if original_query_type is None:

                original_query_type = (
                    decision[
                        "query_type"
                    ]
                )


            #
            # Build context
            #

            context = (
                self.context_builder.build(
                    retrieval_output[
                        "results"
                    ]
                )
            )


            #
            # IMPORTANT:
            #
            # Grade evidence against
            # ORIGINAL user query,
            # not rewritten query.
            #

            grade = (
                self.evidence_grader.grade(

                    query=original_query,

                    context=context,

                    query_type=(
                        original_query_type
                    )

                )
            )


            history.append(

                RetrievalRound(

                    round_number=(
                        round_index
                        + 1
                    ),

                    query=current_query,

                    decision=decision,

                    grade=grade

                )

            )


            #
            # Evidence is enough
            #

            if grade.sufficient:

                return SelfCorrectionResult(

                    original_query=(
                        original_query
                    ),

                    final_query=(
                        current_query
                    ),

                    retrieval_rounds=(
                        round_index
                        + 1
                    ),

                    rewritten=(
                        round_index > 0
                    ),

                    evidence_sufficient=True,

                    context=context,

                    grade=grade,

                    decision=decision,

                    history=history

                )


            #
            # No retries left
            #

            if (
                round_index
                >=
                self.max_retries
            ):

                break


            #
            # Rewrite query
            #

            rewritten_query = (
                self.query_rewriter.rewrite(

                    query=original_query,

                    query_type=(
                        original_query_type
                    ),

                    attempt=(
                        round_index
                        + 1
                    )

                )
            )


            #
            # Protection against
            # useless retry
            #

            if (
                rewritten_query.lower()
                ==
                current_query.lower()
            ):

                break


            current_query = (
                rewritten_query
            )


        #
        # Failed even after retry
        #

        return SelfCorrectionResult(

            original_query=(
                original_query
            ),

            final_query=(
                current_query
            ),

            retrieval_rounds=len(
                history
            ),

            rewritten=(
                current_query
                !=
                original_query
            ),

            evidence_sufficient=False,

            context=context,

            grade=grade,

            decision=decision,

            history=history

        )


    def close(self):

        self.retriever.close()