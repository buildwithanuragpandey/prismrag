import re

from dataclasses import dataclass


_SENTENCE_DOT_PLACEHOLDER = "\uE000"


@dataclass
class AtomicClaims:

    original_text: str

    claims: list[str]


class AtomicClaimExtractor:


    # ========================================================
    # Cleaning
    # ========================================================

    def _clean(
        self,
        text: str
    ) -> str:

        text = (
            text
            or ""
        ).strip()


        # ----------------------------------------------------
        # Remove bullet markers
        # ----------------------------------------------------

        text = re.sub(
            r"^[-*•]\s*",
            "",
            text
        )


        # ----------------------------------------------------
        # Remove existing citations
        # ----------------------------------------------------

        text = re.sub(
            r"\[\d+\]",
            "",
            text
        )


        # ----------------------------------------------------
        # Normalize whitespace
        # ----------------------------------------------------

        text = re.sub(
            r"\s+",
            " ",
            text
        )


        return text.strip()


    # ========================================================
    # Sentence-boundary protection
    #
    # Naive splitting on:
    #
    #     (?<=[.!?])\s+
    #
    # incorrectly breaks:
    #
    # Epic v. Google
    # Chiefs vs. Chargers
    # U.S. antitrust law
    # Dr. Smith
    #
    # Protect these periods before sentence splitting.
    # ========================================================

    def _protect_sentence_periods(
        self,
        text: str
    ) -> str:

        patterns = [

            # Legal / comparison abbreviations
            r"\bv\.",
            r"\bvs\.",

            # Titles
            r"\bMr\.",
            r"\bMrs\.",
            r"\bMs\.",
            r"\bDr\.",
            r"\bProf\.",
            r"\bJr\.",
            r"\bSr\.",

            # Company / general abbreviations
            r"\bInc\.",
            r"\bLtd\.",
            r"\bCorp\.",
            r"\bCo\.",
            r"\bNo\.",

            # Common Latin abbreviations
            r"\be\.g\.",
            r"\bi\.e\.",

            # Country abbreviations
            r"\bU\.S\.",
            r"\bU\.K\.",
        ]


        protected = text


        for pattern in patterns:

            protected = re.sub(

                pattern,

                lambda match: (
                    match.group(0).replace(
                        ".",
                        _SENTENCE_DOT_PLACEHOLDER
                    )
                ),

                protected,

                flags=re.IGNORECASE
            )


        return protected


    @staticmethod
    def _restore_sentence_periods(
        text: str
    ) -> str:

        return text.replace(
            _SENTENCE_DOT_PLACEHOLDER,
            "."
        )


    # ========================================================
    # Sentence splitting
    # ========================================================

    def _split_sentences(
        self,
        text: str
    ) -> list[str]:

        protected = (
            self._protect_sentence_periods(
                text
            )
        )


        sentences = re.split(
            r"(?<=[.!?])\s+",
            protected
        )


        output = []


        for sentence in sentences:

            sentence = (
                self._restore_sentence_periods(
                    sentence
                )
                .strip()
            )


            if sentence:

                output.append(
                    sentence
                )


        return output


    # ========================================================
    # Semicolon splitting
    # ========================================================

    def _split_semicolon(
        self,
        sentence: str
    ) -> list[str]:

        parts = re.split(
            r"\s*;\s*",
            sentence
        )


        parts = [

            part.strip()

            for part
            in parts

            if part.strip()
        ]


        if len(parts) <= 1:

            return parts


        cleaned_parts = []


        # ----------------------------------------------------
        # Context-dependent continuation:
        #
        # however, it ...
        # but they ...
        #
        # is unsafe as a standalone claim.
        # ----------------------------------------------------

        dependent_pattern = re.compile(

            r"^(?:"
            r"however|"
            r"but|"
            r"yet|"
            r"nevertheless|"
            r"nonetheless"
            r"),?\s+"

            r"(?:"
            r"it|"
            r"they|"
            r"this|"
            r"these|"
            r"those|"
            r"he|"
            r"she"
            r")\b",

            flags=re.IGNORECASE
        )


        connector_pattern = re.compile(

            r"^(?:"
            r"however|"
            r"but|"
            r"yet|"
            r"nevertheless|"
            r"nonetheless"
            r"),?\s+",

            flags=re.IGNORECASE
        )


        for index, part in enumerate(
            parts
        ):

            if index == 0:

                cleaned_parts.append(
                    part
                )

                continue


            if dependent_pattern.match(
                part
            ):

                continue


            part = connector_pattern.sub(
                "",
                part
            ).strip()


            if not part:

                continue


            part = (
                part[0].upper()
                +
                part[1:]
            )


            cleaned_parts.append(
                part
            )


        return cleaned_parts


    # ========================================================
    # Dependent explanatory clauses
    # ========================================================

    def _split_explanatory_clause(
        self,
        sentence: str
    ) -> list[str]:

        """
        Remove dependent generated interpretations.

        Example:

        Walmart does not price-match competitors,
        limiting its ability to compete.

        ->

        Walmart does not price-match competitors.
        """

        pattern = (

            r",\s*"

            r"(?:"
            r"allowing|"
            r"providing|"
            r"limiting|"
            r"enhancing|"
            r"making|"
            r"causing|"
            r"creating|"
            r"giving|"
            r"offering|"
            r"bringing"
            r")"

            r"\s+"
        )


        match = re.search(
            pattern,
            sentence,
            flags=re.IGNORECASE
        )


        if not match:

            return [
                sentence.strip()
            ]


        main_claim = (
            sentence[
                :match.start()
            ]
            .strip()
        )


        if not main_claim:

            return []


        return [
            main_claim
        ]


    # ========================================================
    # Finite-verb heuristic
    # ========================================================

    @staticmethod
    def _has_likely_finite_verb(
        words: list[str]
    ) -> bool:

        auxiliary_verbs = {

            "am",
            "is",
            "are",
            "was",
            "were",

            "has",
            "have",
            "had",

            "does",
            "do",
            "did",

            "can",
            "could",

            "will",
            "would",

            "shall",
            "should",

            "may",
            "might",
            "must",
        }


        for word in words:

            clean_word = re.sub(
                r"[^A-Za-z']",
                "",
                word
            ).lower()


            if not clean_word:

                continue


            if clean_word in auxiliary_verbs:

                return True


            # ------------------------------------------------
            # Conservative finite-verb heuristics:
            #
            # starts
            # hosts
            # offers
            # reported
            # changed
            # ------------------------------------------------

            if clean_word.endswith(
                "ed"
            ):

                return True


            if (
                clean_word.endswith(
                    "s"
                )
                and
                len(
                    clean_word
                ) > 3
            ):

                return True


        return False


    # ========================================================
    # Detect unsafe "and" boundary
    # ========================================================

    @staticmethod
    def _looks_like_compound_subject_boundary(
        left: str,
        right: str
    ) -> bool:

        """
        Prevent:

        Taylor Swift and Travis Kelce have been dating.

        from becoming:

        Taylor Swift.
        Travis Kelce have been dating.

        Also prevent:

        reports indicate that Swift and Kelce have been dating

        from becoming:

        reports indicate that Swift.
        Kelce have been dating.
        """

        left_words = (
            left.split()
        )


        right_words = (
            right.split()
        )


        if (
            not left_words
            or
            not right_words
        ):

            return False


        right_first = re.sub(
            r"[^A-Za-z0-9'-]",
            "",
            right_words[
                0
            ]
        )


        if not right_first:

            return False


        right_starts_proper = (
            right_first[
                0
            ].isupper()
        )


        if not right_starts_proper:

            return False


        # ----------------------------------------------------
        # Case 1:
        #
        # Taylor Swift
        # +
        # Travis Kelce have ...
        #
        # Left side has no finite verb, therefore it cannot
        # be an independent clause.
        # ----------------------------------------------------

        if not AtomicClaimExtractor._has_likely_finite_verb(
            left_words
        ):

            return True


        # ----------------------------------------------------
        # Case 2:
        #
        # reports indicate that Swift
        # +
        # Kelce have ...
        #
        # "that Swift" is an unfinished subordinate clause.
        # ----------------------------------------------------

        if re.search(

            (
                r"\b"
                r"(?:that|whether|if)"
                r"\s+"
                r"[A-Z][A-Za-z0-9'-]*"
                r"$"
            ),

            left
        ):

            return True


        return False


    # ========================================================
    # Conservative "and" splitting
    # ========================================================

    def _split_and_clause(
        self,
        sentence: str
    ) -> list[str]:

        """
        Split only when BOTH sides look like independent
        clauses.

        Split:

        Amazon hosts Black Friday sales and
        Walmart starts its promotion earlier.

        Do NOT split:

        Taylor Swift and Travis Kelce have been dating.

        Do NOT split:

        Reports indicate that Swift and Kelce have been dating.

        Do NOT split:

        Amazon Prime members receive free games and DLC.
        """

        parts = re.split(
            r"\s+\band\b\s+",
            sentence,
            maxsplit=1,
            flags=re.IGNORECASE
        )


        if len(parts) != 2:

            return [
                sentence
            ]


        left = (
            parts[
                0
            ].strip()
        )

        right = (
            parts[
                1
            ].strip()
        )


        left_words = (
            left.split()
        )

        right_words = (
            right.split()
        )


        if (
            len(
                left_words
            ) < 2
            or
            len(
                right_words
            ) < 2
        ):

            return [
                sentence
            ]


        # ----------------------------------------------------
        # Prevent compound-subject destruction.
        # ----------------------------------------------------

        if (
            self
            ._looks_like_compound_subject_boundary(
                left,
                right
            )
        ):

            return [
                sentence
            ]


        first_word = re.sub(
            r"[^A-Za-z']",
            "",
            right_words[
                0
            ]
        )


        pronouns = {
            "he",
            "she",
            "it",
            "they",
            "we",
            "you",
        }


        subject_like = (

            (
                first_word
                and
                first_word[
                    0
                ].isupper()
            )

            or

            first_word.lower()
            in pronouns
        )


        if not subject_like:

            return [
                sentence
            ]


        # ----------------------------------------------------
        # BOTH sides must contain a likely finite verb.
        #
        # This is intentionally conservative.
        # Keeping a compound claim is safer than producing
        # two malformed pseudo-claims.
        # ----------------------------------------------------

        left_has_verb = (
            self._has_likely_finite_verb(
                left_words
            )
        )


        right_has_verb = (
            self._has_likely_finite_verb(
                right_words[
                    1:7
                ]
            )
        )


        if not (
            left_has_verb
            and
            right_has_verb
        ):

            return [
                sentence
            ]


        return [
            left,
            right
        ]


    # ========================================================
    # Final claim validity
    # ========================================================

    @staticmethod
    def _is_obvious_fragment(
        claim: str
    ) -> bool:

        text = (
            claim
            or ""
        ).strip()


        if not text:

            return True


        words = re.findall(
            r"[A-Za-z0-9'-]+",
            text
        )


        if len(words) < 2:

            return True


        lower = text.lower()


        # ----------------------------------------------------
        # These endings strongly indicate an accidental
        # sentence-boundary split.
        # ----------------------------------------------------

        bad_endings = (
            " v.",
            " vs.",
            " and",
            " or",
            " but",
        )


        if lower.endswith(
            bad_endings
        ):

            return True


        return False


    # ========================================================
    # Main extraction
    # ========================================================

    def extract(
        self,
        text: str
    ) -> AtomicClaims:

        original_text = (
            text
        )


        text = (
            self._clean(
                text
            )
        )


        if not text:

            return AtomicClaims(
                original_text=
                    original_text,
                claims=[],
            )


        atomic_claims = []


        # ====================================================
        # 1. Sentences
        # ====================================================

        sentences = (
            self._split_sentences(
                text
            )
        )


        for sentence in sentences:


            # =================================================
            # 2. Semicolon clauses
            # =================================================

            semicolon_parts = (
                self._split_semicolon(
                    sentence
                )
            )


            for part in semicolon_parts:


                # =============================================
                # 3. Dependent explanatory clause cleanup
                # =============================================

                explanatory_parts = (
                    self
                    ._split_explanatory_clause(
                        part
                    )
                )


                for explanatory_part in (
                    explanatory_parts
                ):


                    # =========================================
                    # 4. Conservative independent "and" split
                    # =========================================

                    final_parts = (
                        self._split_and_clause(
                            explanatory_part
                        )
                    )


                    for final_part in final_parts:

                        claim = (
                            final_part
                            .strip(
                                " ,"
                            )
                        )


                        if (
                            self
                            ._is_obvious_fragment(
                                claim
                            )
                        ):

                            continue


                        if claim[
                            -1
                        ] not in ".!?":

                            claim += "."


                        atomic_claims.append(
                            claim
                        )


        # ====================================================
        # Deduplicate preserving generation order
        # ====================================================

        unique_claims = list(
            dict.fromkeys(
                atomic_claims
            )
        )


        return AtomicClaims(
            original_text=
                original_text,
            claims=
                unique_claims,
        )