import re

from dataclasses import dataclass

import torch


YES_NO_STARTERS = {
    "am",
    "are",
    "can",
    "could",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "is",
    "should",
    "was",
    "were",
    "will",
    "would",
}


MULTI_EVIDENCE_PATTERNS = (
    r"\bwhile\b",
    r"\bwhereas\b",
    r"\bcompared\s+to\b",
    r"\bcompared\s+with\b",
    r"\bin\s+contrast\s+to\b",
    r"\bbetween\b",
    r"\bboth\s+articles\b",
    r"\bboth\s+reports\b",
    r"\bearlier\s+report\b",
    r"\blater\s+report\b",
    r"\bsubsequent\s+report\b",
    r"\bsubsequent\s+article\b",
    r"\bconsistent\b",
    r"\binconsistent\b",
    r"\bcontradict\b",
)


ENTITY_QUESTION_PATTERN = re.compile(
    r"""
    ^\s*
    (?:
        who
        |
        which
        (?:\s+[A-Za-z0-9'-]+){0,6}
        \s+
        (?:
            individual
            |person
            |company
            |organization
            |organisation
            |country
            |city
            |university
            |team
            |player
            |source
            |publication
        )
        |
        what
        (?:\s+[A-Za-z0-9'-]+){0,5}
        \s+
        (?:
            company
            |organization
            |organisation
            |country
            |city
            |university
            |source
            |publication
        )
    )
    \b
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


SOURCE_TARGET_PATTERNS = (
    r"\bwhich\s+news\s+source\b",
    r"\bwhich\s+source\b",
    r"\bwhich\s+publication\b",
    r"\bwhat\s+news\s+source\b",
    r"\bwhat\s+source\b",
    r"\bwhat\s+publication\b",
)


@dataclass
class AnswerSynthesisResult:

    accepted: bool

    final_answer: str | None

    mode: str

    reasons: list[str]

    yes_score: float | None = None

    no_score: float | None = None

    entity_candidate: str | None = None

    unique_citation_count: int = 0


class ConstrainedAnswerSynthesizer:
    """
    Post-grounding answer synthesis.

    It does NOT generate new evidence.

    Modes:

        yes_no
            Compare sequence log-probability of exactly:
                Yes
                No

            conditioned only on the question and verified
            relevant facts.

        entity
            Validate the draft entity answer against verified
            relevant claims.

        other
            Preserve the original draft direct answer.

    No second free-form generation pass is used.
    """

    # ========================================================
    # Text normalization
    # ========================================================

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:

        return " ".join(
            re.findall(
                r"[a-z0-9]+",
                (text or "").lower(),
            )
        )


    # ========================================================
    # Answer type
    # ========================================================

    @classmethod
    def detect_answer_type(
        cls,
        query: str,
    ) -> str:

        tokens = re.findall(
            r"[a-z]+",
            (query or "").lower(),
        )


        if (
            tokens
            and
            tokens[0]
            in
            YES_NO_STARTERS
        ):

            return "yes_no"


        if ENTITY_QUESTION_PATTERN.search(
            query or ""
        ):

            return "entity"


        return "other"


    # ========================================================
    # Structural evidence requirements
    # ========================================================

    @staticmethod
    def _requires_multiple_evidence(
        query: str,
    ) -> bool:

        return any(
            re.search(
                pattern,
                query or "",
                flags=re.IGNORECASE,
            )

            for pattern
            in MULTI_EVIDENCE_PATTERNS
        )


    @staticmethod
    def _unique_citations(
        relevant_claims,
    ) -> set[int]:

        return {
            claim.citation_id

            for claim
            in relevant_claims

            if (
                getattr(
                    claim,
                    "citation_id",
                    None,
                )
                is not None
            )
        }


    # ========================================================
    # Context-source helpers
    # ========================================================

    @classmethod
    def _source_aliases(
        cls,
        source: str,
    ) -> set[str]:

        source = (
            source
            or ""
        ).strip()


        if not source:

            return set()


        primary = (
            source
            .split(
                "|",
                1,
            )[
                0
            ]
            .strip()
        )


        aliases = {
            cls._normalize(
                source
            ),

            cls._normalize(
                primary
            ),
        }


        expanded = set(
            aliases
        )


        for alias in aliases:

            if alias.startswith(
                "the "
            ):

                without_the = (
                    alias[
                        4:
                    ]
                    .strip()
                )


                # Avoid collapsing "The Age" into generic "age".
                if len(
                    without_the.split()
                ) >= 2:

                    expanded.add(
                        without_the
                    )


        return {
            alias

            for alias
            in expanded

            if alias
        }


    @classmethod
    def _canonical_source(
        cls,
        source: str,
    ) -> str:

        primary = (
            (source or "")
            .split(
                "|",
                1,
            )[
                0
            ]
            .strip()
        )


        return cls._normalize(
            primary
        )


    @classmethod
    def _mentioned_context_sources(
        cls,
        query: str,
        context,
    ) -> set[str]:

        query_normalized = (
            cls._normalize(
                query
            )
        )


        output = set()


        for item in context.items:

            source = (
                getattr(
                    item,
                    "source",
                    "",
                )
                or ""
            )


            if not source:

                continue


            aliases = (
                cls._source_aliases(
                    source
                )
            )


            mentioned = any(
                re.search(
                    (
                        r"(?:^|\s)"
                        +
                        re.escape(
                            alias
                        )
                        +
                        r"(?:$|\s)"
                    ),
                    query_normalized,
                )

                for alias
                in aliases

                if alias
            )


            if mentioned:

                output.add(
                    cls._canonical_source(
                        source
                    )
                )


        return output


    @classmethod
    def _covered_sources(
        cls,
        relevant_claims,
        context,
    ) -> set[str]:

        citation_ids = (
            cls._unique_citations(
                relevant_claims
            )
        )


        covered = set()


        for item in context.items:

            citation_id = getattr(
                item,
                "citation_id",
                None,
            )


            if (
                citation_id
                not in
                citation_ids
            ):

                continue


            source = (
                getattr(
                    item,
                    "source",
                    "",
                )
                or ""
            )


            if source:

                covered.add(
                    cls._canonical_source(
                        source
                    )
                )


        return covered


    # ========================================================
    # Verified-facts prompt
    # ========================================================

    @staticmethod
    def _build_yes_no_messages(
        query: str,
        relevant_claims,
    ) -> list[dict]:

        facts = "\n".join(
            (
                f"- {claim.claim}"
            )

            for claim
            in relevant_claims
        )


        system = """
You are a binary evidence decision component.

Use ONLY the VERIFIED FACTS.

Decide whether the QUESTION should be answered Yes or No.

Do not use outside knowledge.

Do not explain.

The only valid answer is exactly one word:

Yes

or

No
""".strip()


        user = f"""
QUESTION:

{query}


VERIFIED FACTS:

{facts}


Choose the better supported answer:

Yes

or

No
""".strip()


        return [
            {
                "role":
                    "system",

                "content":
                    system,
            },
            {
                "role":
                    "user",

                "content":
                    user,
            },
        ]


    # ========================================================
    # Generic continuation score
    #
    # This works even if Yes or No consists of more than one
    # tokenizer token.
    #
    # We use mean token log-probability for length neutrality.
    # ========================================================

    @staticmethod
    def _score_candidate(
        *,
        prompt: str,
        candidate: str,
        model,
        tokenizer,
        device: str,
    ) -> float:

        prompt_ids = (
            tokenizer(
                prompt,
                add_special_tokens=False,
            )[
                "input_ids"
            ]
        )


        candidate_ids = (
            tokenizer(
                candidate,
                add_special_tokens=False,
            )[
                "input_ids"
            ]
        )


        if not candidate_ids:

            raise ValueError(
                (
                    "Candidate tokenization produced "
                    "no tokens."
                )
            )


        combined_ids = (
            prompt_ids
            +
            candidate_ids
        )


        input_ids = torch.tensor(
            [
                combined_ids
            ],
            dtype=torch.long,
            device=device,
        )


        attention_mask = torch.ones_like(
            input_ids
        )


        with torch.inference_mode():

            outputs = model(
                input_ids=
                    input_ids,

                attention_mask=
                    attention_mask,
            )


        logits = (
            outputs.logits[
                0
            ]
        )


        log_probs = torch.log_softmax(
            logits,
            dim=-1,
        )


        prompt_length = len(
            prompt_ids
        )


        token_scores = []


        for offset, token_id in enumerate(
            candidate_ids
        ):

            target_position = (
                prompt_length
                +
                offset
            )


            predictor_position = (
                target_position
                -
                1
            )


            token_log_prob = float(
                log_probs[
                    predictor_position,
                    token_id,
                ]
                .item()
            )


            token_scores.append(
                token_log_prob
            )


        return (
            sum(
                token_scores
            )
            /
            len(
                token_scores
            )
        )


    def _score_yes_no_candidates(
        self,
        *,
        query: str,
        relevant_claims,
        model,
        tokenizer,
        device: str,
    ) -> dict[str, float]:

        messages = (
            self._build_yes_no_messages(
                query=
                    query,

                relevant_claims=
                    relevant_claims,
            )
        )


        prompt = (
            tokenizer
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )


        yes_score = (
            self._score_candidate(
                prompt=
                    prompt,

                candidate=
                    "Yes",

                model=
                    model,

                tokenizer=
                    tokenizer,

                device=
                    device,
            )
        )


        no_score = (
            self._score_candidate(
                prompt=
                    prompt,

                candidate=
                    "No",

                model=
                    model,

                tokenizer=
                    tokenizer,

                device=
                    device,
            )
        )


        return {
            "yes":
                yes_score,

            "no":
                no_score,
        }


    # ========================================================
    # Entity helpers
    # ========================================================

    @staticmethod
    def _question_allows_source_answer(
        query: str,
    ) -> bool:

        return any(
            re.search(
                pattern,
                query or "",
                flags=re.IGNORECASE,
            )

            for pattern
            in SOURCE_TARGET_PATTERNS
        )


    @classmethod
    def _extract_entity_candidate(
        cls,
        direct_answer: str,
    ) -> str:

        text = (
            direct_answer
            or ""
        ).strip()


        if not text:

            return ""


        # Keep only the short entity head when the model adds
        # an unnecessary explanation.
        parts = re.split(
            (
                r","
                r"|\b(?:"
                r"is"
                r"|are"
                r"|was"
                r"|were"
                r"|has"
                r"|have"
                r"|had"
                r"|will"
                r"|would"
                r"|can"
                r"|could"
                r"|should"
                r"|reported"
                r"|reports"
                r"|suggested"
                r"|suggests"
                r")\b"
            ),
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )


        candidate = (
            parts[
                0
            ]
            .strip(
                " .,:;!?\"'"
            )
        )


        return candidate


    @classmethod
    def _entity_variants(
        cls,
        candidate: str,
    ) -> set[str]:

        normalized = (
            cls._normalize(
                candidate
            )
        )


        variants = {
            normalized
        }


        without_parenthetical = re.sub(
            r"\([^)]*\)",
            "",
            candidate,
        )


        without_parenthetical = (
            cls._normalize(
                without_parenthetical
            )
        )


        if without_parenthetical:

            variants.add(
                without_parenthetical
            )


        output = set()


        for variant in variants:

            if not variant:

                continue


            output.add(
                variant
            )


            if variant.startswith(
                "the "
            ):

                output.add(
                    variant[
                        4:
                    ]
                )


        return output


    @classmethod
    def _entity_appears_in_verified_claims(
        cls,
        candidate: str,
        relevant_claims,
    ) -> bool:

        variants = (
            cls._entity_variants(
                candidate
            )
        )


        if not variants:

            return False


        for claim in relevant_claims:

            claim_text = (
                cls._normalize(
                    getattr(
                        claim,
                        "claim",
                        "",
                    )
                )
            )


            for variant in variants:

                if (
                    re.search(
                        (
                            r"(?:^|\s)"
                            +
                            re.escape(
                                variant
                            )
                            +
                            r"(?:$|\s)"
                        ),
                        claim_text,
                    )
                ):

                    return True


        return False


    @classmethod
    def _entity_is_context_source(
        cls,
        candidate: str,
        context,
    ) -> bool:

        candidate_variants = (
            cls._entity_variants(
                candidate
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


            source_aliases = (
                cls._source_aliases(
                    source
                )
            )


            if (
                candidate_variants
                &
                source_aliases
            ):

                return True


        return False


    # ========================================================
    # Main
    # ========================================================

    def synthesize(
        self,
        *,
        query: str,
        draft_direct_answer: str,
        relevant_claims,
        context,
        model,
        tokenizer,
        device: str,
    ) -> AnswerSynthesisResult:

        answer_type = (
            self.detect_answer_type(
                query
            )
        )


        unique_citations = (
            self._unique_citations(
                relevant_claims
            )
        )


        # ====================================================
        # Yes / No
        # ====================================================

        if (
            answer_type
            ==
            "yes_no"
        ):

            reasons = []


            if (
                self._requires_multiple_evidence(
                    query
                )
                and
                len(
                    unique_citations
                )
                <
                2
            ):

                reasons.append(
                    (
                        "Multi-evidence yes/no question "
                        "has fewer than two independently "
                        "grounded citations."
                    )
                )


            required_sources = (
                self._mentioned_context_sources(
                    query=
                        query,

                    context=
                        context,
                )
            )


            covered_sources = (
                self._covered_sources(
                    relevant_claims=
                        relevant_claims,

                    context=
                        context,
                )
            )


            missing_sources = (
                required_sources
                -
                covered_sources
            )


            # If multiple explicitly named sources exist in
            # the retrieved context, the final verified facts
            # must cover all of them before binary synthesis.
            if (
                len(
                    required_sources
                )
                >=
                2
                and
                missing_sources
            ):

                reasons.append(
                    (
                        "Verified facts do not cover all "
                        "explicitly referenced sources: "
                        +
                        ", ".join(
                            sorted(
                                missing_sources
                            )
                        )
                    )
                )


            if reasons:

                return AnswerSynthesisResult(
                    accepted=
                        False,

                    final_answer=
                        None,

                    mode=
                        "yes_no",

                    reasons=
                        reasons,

                    unique_citation_count=
                        len(
                            unique_citations
                        ),
                )


            scores = (
                self._score_yes_no_candidates(
                    query=
                        query,

                    relevant_claims=
                        relevant_claims,

                    model=
                        model,

                    tokenizer=
                        tokenizer,

                    device=
                        device,
                )
            )


            final_answer = (
                "Yes"

                if (
                    scores[
                        "yes"
                    ]
                    >
                    scores[
                        "no"
                    ]
                )

                else
                "No"
            )


            return AnswerSynthesisResult(
                accepted=
                    True,

                final_answer=
                    final_answer,

                mode=
                    "yes_no",

                reasons=[
                    (
                        "Final binary answer selected by "
                        "constrained Yes/No continuation "
                        "scoring over verified facts."
                    )
                ],

                yes_score=
                    scores[
                        "yes"
                    ],

                no_score=
                    scores[
                        "no"
                    ],

                unique_citation_count=
                    len(
                        unique_citations
                    ),
            )


        # ====================================================
        # Entity
        # ====================================================

        if (
            answer_type
            ==
            "entity"
        ):

            candidate = (
                self._extract_entity_candidate(
                    draft_direct_answer
                )
            )


            reasons = []


            if not candidate:

                reasons.append(
                    "No usable entity candidate was produced."
                )


            elif not (
                self._entity_appears_in_verified_claims(
                    candidate=
                        candidate,

                    relevant_claims=
                        relevant_claims,
                )
            ):

                reasons.append(
                    (
                        "Draft entity does not appear in "
                        "any verified relevant claim."
                    )
                )


            if (
                candidate
                and
                not self._question_allows_source_answer(
                    query
                )
                and
                self._entity_is_context_source(
                    candidate=
                        candidate,

                    context=
                        context,
                )
            ):

                reasons.append(
                    (
                        "Draft entity matches evidence "
                        "provenance/source although the "
                        "question does not ask for a source."
                    )
                )


            return AnswerSynthesisResult(
                accepted=(
                    len(
                        reasons
                    )
                    ==
                    0
                ),

                final_answer=(
                    candidate
                    if not reasons
                    else None
                ),

                mode=
                    "entity",

                reasons=
                    reasons,

                entity_candidate=
                    candidate or None,

                unique_citation_count=
                    len(
                        unique_citations
                    ),
            )


        # ====================================================
        # Other
        # ====================================================

        return AnswerSynthesisResult(
            accepted=
                True,

            final_answer=
                draft_direct_answer,

            mode=
                "other",

            reasons=[
                (
                    "No constrained synthesis rule "
                    "applies to this answer type."
                )
            ],

            unique_citation_count=
                len(
                    unique_citations
                ),
        )