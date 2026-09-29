import re

from dataclasses import dataclass

import numpy as np
import torch

from sentence_transformers import CrossEncoder

from adaptive_agentic_rag.generation.context_builder import (
    BuiltContext,
)

from adaptive_agentic_rag.generation.atomic_claim_extractor import (
    AtomicClaimExtractor,
)

from adaptive_agentic_rag.generation.sentence_splitter import (
    split_sentences,
)


NLI_MODEL_NAME = (
    "cross-encoder/nli-deberta-v3-small"
)


LABELS = [
    "contradiction",
    "entailment",
    "neutral",
]


@dataclass
class ClaimSupport:

    claim: str

    supported: bool

    citation_id: int | None

    label: str

    entailment_score: float

    supporting_text: str | None

    evidence_relevance_score: float | None = None

    premise_mode: str | None = None


@dataclass
class GroundedClaims:

    claims: list[ClaimSupport]

    supported_count: int

    unsupported_count: int


class ClaimGrounder:

    def __init__(
        self,
        reranker,
        model_name: str = NLI_MODEL_NAME,
        device: str | None = None,
        max_candidate_units: int = 6,
    ):

        if device is None:

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )


        self.device = (
            device
        )


        self.reranker = (
            reranker
        )


        self.max_candidate_units = (
            max_candidate_units
        )


        print(
            f"Loading NLI model: "
            f"{model_name}"
        )


        self.model = (
            CrossEncoder(
                model_name,
                device=device,
            )
        )


        self.atomic_extractor = (
            AtomicClaimExtractor()
        )


    # ========================================================
    # Claim extraction
    # ========================================================

    def extract_claims(
        self,
        answer: str,
    ) -> list[str]:

        atomic_claims = []


        for line in answer.splitlines():

            line = (
                line.strip()
            )


            if not line:

                continue


            if (
                line.lower()
                in {
                    "answer:",
                    "response:",
                    "final answer:",
                }
            ):

                continue


            line = re.sub(
                r"^[-*•]\s*",
                "",
                line,
            )


            line = re.sub(
                r"\[\d+\]",
                "",
                line,
            )


            line = (
                line.strip()
            )


            if not line:

                continue


            extraction = (
                self.atomic_extractor
                .extract(
                    line
                )
            )


            atomic_claims.extend(
                extraction.claims
            )


        return list(
            dict.fromkeys(
                atomic_claims
            )
        )


    # ========================================================
    # Evidence sentence splitting
    # ========================================================

    def _split_sentences(
        self,
        text: str,
    ) -> list[str]:

        return (
            split_sentences(
                text,
                split_newlines=False,
                min_chars=20,
            )
        )


    # ========================================================
    # Evidence units
    # ========================================================

    def _build_evidence_units(
        self,
        context: BuiltContext,
    ) -> list[dict]:

        units = []


        for item in context.items:

            sentences = (
                self._split_sentences(
                    item.text
                )
            )


            provenance_parts = []


            if item.source:

                provenance_parts.append(
                    f"Source: {item.source}."
                )


            if item.title:

                provenance_parts.append(
                    f"Title: {item.title}."
                )


            provenance = " ".join(
                provenance_parts
            )


            for (
                index,
                sentence,
            ) in enumerate(
                sentences
            ):

                self._append_unit(
                    units=units,

                    citation_id=
                        item.citation_id,

                    source=
                        item.source,

                    title=
                        item.title,

                    text=
                        sentence,

                    provenance=
                        provenance,

                    unit_type=
                        "sentence",
                )


                if (
                    index + 1
                    <
                    len(
                        sentences
                    )
                ):

                    window = (
                        sentence
                        +
                        " "
                        +
                        sentences[
                            index + 1
                        ]
                    )


                    self._append_unit(
                        units=units,

                        citation_id=
                            item.citation_id,

                        source=
                            item.source,

                        title=
                            item.title,

                        text=
                            window,

                        provenance=
                            provenance,

                        unit_type=
                            "window",
                    )


        return units


    @staticmethod
    def _append_unit(
        *,
        units,
        citation_id,
        source,
        title,
        text,
        provenance,
        unit_type,
    ):

        text = (
            text.strip()
        )


        if not text:

            return


        provenance_text = (
            (
                f"{provenance} "
                f"Evidence: {text}"
            )
            .strip()
        )


        units.append(
            {
                "id":
                    (
                        f"evidence_"
                        f"{len(units)}"
                    ),

                "citation_id":
                    citation_id,

                "source":
                    source or "",

                "title":
                    title or "",

                "text":
                    text,

                "provenance_text":
                    provenance_text,

                "unit_type":
                    unit_type,
            }
        )


    # ========================================================
    # Source normalization
    # ========================================================

    @staticmethod
    def _normalize_source(
        text: str,
    ) -> str:

        return " ".join(
            re.findall(
                r"[a-z0-9]+",
                (text or "").lower(),
            )
        )


    # ========================================================
    # Context-source aliases
    #
    # Examples:
    #
    # CNBC | World Business News Leader
    #
    # →
    #
    # cnbc world business news leader
    # cnbc
    #
    #
    # The Sydney Morning Herald
    #
    # →
    #
    # the sydney morning herald
    # sydney morning herald
    #
    #
    # But:
    #
    # The Age
    #
    # does NOT become:
    #
    # age
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
            cls._normalize_source(
                source
            ),

            cls._normalize_source(
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


                if (
                    len(
                        without_the.split()
                    )
                    >=
                    2
                ):

                    expanded.add(
                        without_the
                    )


        return {
            alias

            for alias
            in expanded

            if alias
        }


    # ========================================================
    # Does claim explicitly mention a particular context
    # source?
    # ========================================================

    @classmethod
    def _claim_mentions_source(
        cls,
        *,
        claim: str,
        source: str,
    ) -> bool:

        claim_normalized = (
            cls._normalize_source(
                claim
            )
        )


        if not claim_normalized:

            return False


        for alias in (
            cls._source_aliases(
                source
            )
        ):

            if not alias:

                continue


            pattern = (
                r"(?:^|\s)"
                +
                re.escape(
                    alias
                )
                +
                r"(?:$|\s)"
            )


            if re.search(
                pattern,
                claim_normalized,
            ):

                return True


        return False


    # ========================================================
    # Explicit provenance detection
    #
    # This is DIFFERENT from matching a context source.
    #
    # We need to distinguish:
    #
    # A)
    #
    #   "Google faced an antitrust lawsuit."
    #
    #   → no source attribution
    #   → unrestricted grounding
    #
    #
    # B)
    #
    #   "The Age reported that Google..."
    #
    #   → explicit provenance attribution
    #
    # If The Age exists in context:
    #     bind to The Age
    #
    # If The Age does NOT exist in context:
    #     FAIL CLOSED
    #
    #
    # Without this distinction, missing source evidence can
    # silently fall back to another publisher.
    # ========================================================

    @staticmethod
    def _looks_like_source_phrase(
        phrase: str,
    ) -> bool:

        phrase = (
            phrase
            or ""
        ).strip()


        phrase = (
            phrase
            .strip(
                " \"'“”‘’.,:;"
            )
        )


        if not phrase:

            return False


        tokens = re.findall(
            r"[A-Za-z0-9&|'’.-]+",
            phrase,
        )


        if not tokens:

            return False


        # ----------------------------------------------------
        # Keep this intentionally conservative.
        #
        # A source attribution prefix should normally be short.
        #
        # This prevents things such as:
        #
        # "The live score update and highlight excerpt for ..."
        #
        # from being interpreted as a publisher.
        # ----------------------------------------------------

        if len(
            tokens
        ) > 8:

            return False


        connector_words = {
            "the",
            "of",
            "and",
            "for",
            "world",
            "business",
            "news",
            "leader",
        }


        meaningful = []


        for token in tokens:

            if (
                token.lower()
                in
                connector_words
            ):

                continue


            meaningful.append(
                token
            )


        if not meaningful:

            return False


        # ----------------------------------------------------
        # A likely source phrase normally contains named /
        # capitalized tokens:
        #
        # The Age
        # Fortune
        # TechCrunch
        # Reuters
        # CNBC
        # Sydney Morning Herald
        #
        # Internal capitals such as TechCrunch are accepted.
        # ----------------------------------------------------

        for token in meaningful:

            has_upper = any(
                character.isupper()

                for character
                in token
            )


            if not has_upper:

                return False


        return True


    # ========================================================
    # Extract explicit source-attribution prefix
    #
    # Supported structural forms include:
    #
    # The Fortune article ...
    # The Sydney Morning Herald report ...
    # TechCrunch reported ...
    # The Age reported ...
    # The Verge covered ...
    # According to Reuters, ...
    #
    #
    # This does NOT attempt full NER.
    #
    # It exists only to decide whether grounding should
    # fail closed when a source-attributed claim cannot be
    # bound to the current context.
    # ========================================================

    @classmethod
    def _extract_explicit_source_attribution(
        cls,
        claim: str,
    ) -> str | None:

        claim = (
            claim
            or ""
        ).strip()


        if not claim:

            return None


        patterns = [
            # ------------------------------------------------
            # "The Fortune article ..."
            #
            # "TechCrunch report ..."
            #
            # "The Verge coverage ..."
            # ------------------------------------------------
            (
                r"^\s*"
                r"(?P<source>.+?)"
                r"\s+"
                r"(?:article|report|coverage|reporting)"
                r"\b"
            ),

            # ------------------------------------------------
            # "The Age reported ..."
            #
            # "TechCrunch reports ..."
            #
            # "The Verge covered ..."
            # ------------------------------------------------
            (
                r"^\s*"
                r"(?P<source>.+?)"
                r"\s+"
                r"(?:reported|reports|covered|covers)"
                r"\b"
            ),

            # ------------------------------------------------
            # "According to Reuters, ..."
            # ------------------------------------------------
            (
                r"^\s*"
                r"according\s+to\s+"
                r"(?P<source>.+?)"
                r"(?:,|$)"
            ),
        ]


        for pattern in patterns:

            match = re.search(
                pattern,
                claim,
                flags=re.IGNORECASE,
            )


            if not match:

                continue


            source_phrase = (
                match.group(
                    "source"
                )
                .strip()
                .strip(
                    " \"'“”‘’.,:;"
                )
            )


            if not source_phrase:

                continue


            # ------------------------------------------------
            # Regex IGNORECASE is used for syntax detection,
            # but source-like validation uses original casing.
            # ------------------------------------------------

            if cls._looks_like_source_phrase(
                source_phrase
            ):

                return source_phrase


        return None


    # ========================================================
    # HARD PROVENANCE BINDING
    #
    # Three possible states:
    #
    # --------------------------------------------------------
    # 1. Claim explicitly names a source PRESENT in context
    #
    #    → keep units from that source only
    #
    #
    # 2. Claim explicitly attributes information to a source
    #    but that source is ABSENT from context
    #
    #    → return []
    #    → claim becomes unsupported
    #
    #
    # 3. Claim has no source attribution
    #
    #    → preserve Grounder V2 unrestricted behavior
    #
    #
    # This is deliberately FAIL-CLOSED for provenance.
    # ========================================================

    @classmethod
    def _apply_source_binding(
        cls,
        *,
        claim: str,
        units: list[dict],
    ) -> list[dict]:

        if not units:

            return []


        # ====================================================
        # Collect unique sources actually available in the
        # current grounding context.
        # ====================================================

        available_sources = []

        seen_sources = set()


        for unit in units:

            source = (
                unit.get(
                    "source",
                    "",
                )
                or ""
            ).strip()


            if not source:

                continue


            source_key = (
                cls._normalize_source(
                    source
                )
            )


            if not source_key:

                continue


            if (
                source_key
                in
                seen_sources
            ):

                continue


            seen_sources.add(
                source_key
            )


            available_sources.append(
                source
            )


        # ====================================================
        # Which AVAILABLE sources are explicitly named in
        # this claim?
        # ====================================================

        mentioned_sources = []


        for source in (
            available_sources
        ):

            if cls._claim_mentions_source(
                claim=
                    claim,

                source=
                    source,
            ):

                mentioned_sources.append(
                    source
                )


        # ====================================================
        # CASE 1:
        #
        # Claim names one or more sources that exist in the
        # current evidence context.
        #
        # Hard-bind to them.
        # ====================================================

        if mentioned_sources:

            allowed_sources = {
                cls._normalize_source(
                    source
                )

                for source
                in mentioned_sources
            }


            filtered_units = []


            for unit in units:

                unit_source = (
                    cls._normalize_source(
                        unit.get(
                            "source",
                            "",
                        )
                    )
                )


                if (
                    unit_source
                    in
                    allowed_sources
                ):

                    filtered_units.append(
                        unit
                    )


            return filtered_units


        # ====================================================
        # CASE 2:
        #
        # No current context source matched.
        #
        # Determine whether the claim nevertheless contains
        # an explicit provenance attribution.
        #
        # Example:
        #
        #   "The Age reported ..."
        #
        # but current context sources are:
        #
        #   TechCrunch
        #   The Verge
        #
        #
        # Old behavior:
        #
        #   return units
        #   → TechCrunch could support The Age claim ❌
        #
        # New behavior:
        #
        #   return []
        #   → unsupported ✅
        # ====================================================

        explicit_source = (
            cls._extract_explicit_source_attribution(
                claim
            )
        )


        if explicit_source is not None:

            return []


        # ====================================================
        # CASE 3:
        #
        # No explicit source attribution.
        #
        # Preserve original Grounder V2 behavior.
        # ====================================================

        return units


    # ========================================================
    # Candidate retrieval BEFORE NLI
    #
    # evidence units
    #       ↓
    # hard provenance binding
    #       ↓
    # BGE candidate ranking
    #       ↓
    # top-k
    #       ↓
    # NLI
    # ========================================================

    def _select_candidate_units(
        self,
        claim: str,
        units: list[dict],
    ) -> list[dict]:

        if not units:

            return []


        # ====================================================
        # Structural eligibility BEFORE semantic ranking.
        # ====================================================

        units = (
            self._apply_source_binding(
                claim=
                    claim,

                units=
                    units,
            )
        )


        if not units:

            return []


        documents = []


        for index, unit in enumerate(
            units
        ):

            documents.append(
                {
                    "id":
                        f"grounding_unit_{index}",

                    "text":
                        unit[
                            "provenance_text"
                        ],

                    "unit":
                        unit,
                }
            )


        top_k = min(
            self.max_candidate_units,
            len(
                documents
            ),
        )


        ranked = (
            self.reranker.rerank(
                query=
                    claim,

                documents=
                    documents,

                top_k=
                    top_k,
            )
        )


        candidates = []


        for item in ranked:

            unit = (
                item[
                    "unit"
                ].copy()
            )


            unit[
                "relevance_score"
            ] = round(
                float(
                    item[
                        "rerank_score"
                    ]
                ),
                4,
            )


            candidates.append(
                unit
            )


        return candidates


    # ========================================================
    # Score one premise
    # ========================================================

    def _predict_nli(
        self,
        premise: str,
        claim: str,
    ) -> dict:

        probabilities = (
            self.model.predict(
                [
                    (
                        premise,
                        claim,
                    )
                ],
                apply_softmax=True,
            )
        )


        probabilities = (
            np.asarray(
                probabilities
            )
        )


        if (
            probabilities.ndim
            ==
            2
        ):

            probabilities = (
                probabilities[
                    0
                ]
            )


        label_index = int(
            np.argmax(
                probabilities
            )
        )


        return {
            "label":
                LABELS[
                    label_index
                ],

            "contradiction":
                float(
                    probabilities[
                        0
                    ]
                ),

            "entailment":
                float(
                    probabilities[
                        1
                    ]
                ),

            "neutral":
                float(
                    probabilities[
                        2
                    ]
                ),
        }


    # ========================================================
    # Evaluate candidate evidence
    #
    # Test both:
    #
    # plain text
    #
    # and
    #
    # source + title + evidence
    #
    # Source eligibility has already been enforced BEFORE
    # this stage.
    # ========================================================

    def _evaluate_candidate(
        self,
        claim: str,
        unit: dict,
    ) -> list[dict]:

        variants = [
            (
                "plain",
                unit[
                    "text"
                ],
            ),

            (
                "provenance",
                unit[
                    "provenance_text"
                ],
            ),
        ]


        results = []


        for (
            mode,
            premise,
        ) in variants:

            prediction = (
                self._predict_nli(
                    premise=
                        premise,

                    claim=
                        claim,
                )
            )


            results.append(
                {
                    "mode":
                        mode,

                    "premise":
                        premise,

                    "citation_id":
                        unit[
                            "citation_id"
                        ],

                    "relevance_score":
                        unit[
                            "relevance_score"
                        ],

                    **prediction,
                }
            )


        return results


    # ========================================================
    # Check one claim
    # ========================================================

    def _check_claim(
        self,
        claim: str,
        context: BuiltContext,
    ) -> ClaimSupport:

        units = (
            self._build_evidence_units(
                context
            )
        )


        if not units:

            return ClaimSupport(
                claim=
                    claim,

                supported=
                    False,

                citation_id=
                    None,

                label=
                    "neutral",

                entailment_score=
                    0.0,

                supporting_text=
                    None,

                evidence_relevance_score=
                    None,

                premise_mode=
                    None,
            )


        candidates = (
            self._select_candidate_units(
                claim=
                    claim,

                units=
                    units,
            )
        )


        # ====================================================
        # This now includes provenance fail-closed.
        #
        # If a claim explicitly names a source that is absent
        # from the current context:
        #
        # candidates == []
        #
        # and the claim becomes unsupported.
        # ====================================================

        if not candidates:

            return ClaimSupport(
                claim=
                    claim,

                supported=
                    False,

                citation_id=
                    None,

                label=
                    "neutral",

                entailment_score=
                    0.0,

                supporting_text=
                    None,

                evidence_relevance_score=
                    None,

                premise_mode=
                    None,
            )


        evaluations = []


        for unit in candidates:

            evaluations.extend(
                self._evaluate_candidate(
                    claim=
                        claim,

                    unit=
                        unit,
                )
            )


        # ====================================================
        # Existing support rule remains unchanged.
        #
        # Entailment must be the winning NLI label.
        #
        # No new threshold.
        # ====================================================

        entailed = [

            evaluation

            for evaluation
            in evaluations

            if (
                evaluation[
                    "label"
                ]
                ==
                "entailment"
            )
        ]


        if entailed:

            best = max(

                entailed,

                key=lambda item: (
                    item[
                        "entailment"
                    ],

                    item[
                        "relevance_score"
                    ],
                ),
            )


            return ClaimSupport(
                claim=
                    claim,

                supported=
                    True,

                citation_id=
                    best[
                        "citation_id"
                    ],

                label=
                    "entailment",

                entailment_score=
                    round(
                        best[
                            "entailment"
                        ],
                        4,
                    ),

                supporting_text=
                    best[
                        "premise"
                    ],

                evidence_relevance_score=
                    best[
                        "relevance_score"
                    ],

                premise_mode=
                    best[
                        "mode"
                    ],
            )


        # ====================================================
        # Unsupported:
        #
        # preserve highest-entailment diagnostic candidate.
        # ====================================================

        best = max(

            evaluations,

            key=lambda item:
                item[
                    "entailment"
                ],
        )


        return ClaimSupport(
            claim=
                claim,

            supported=
                False,

            citation_id=
                None,

            label=
                best[
                    "label"
                ],

            entailment_score=
                round(
                    best[
                        "entailment"
                    ],
                    4,
                ),

            supporting_text=
                best[
                    "premise"
                ],

            evidence_relevance_score=
                best[
                    "relevance_score"
                ],

            premise_mode=
                best[
                    "mode"
                ],
        )


    # ========================================================
    # Ground full generated answer
    # ========================================================

    def ground(
        self,
        answer: str,
        context: BuiltContext,
    ) -> GroundedClaims:

        claims = (
            self.extract_claims(
                answer
            )
        )


        results = []


        for claim in claims:

            result = (
                self._check_claim(
                    claim=
                        claim,

                    context=
                        context,
                )
            )


            results.append(
                result
            )


        supported_count = sum(

            1

            for result
            in results

            if result.supported
        )


        unsupported_count = (
            len(
                results
            )
            -
            supported_count
        )


        return GroundedClaims(
            claims=
                results,

            supported_count=
                supported_count,

            unsupported_count=
                unsupported_count,
        )