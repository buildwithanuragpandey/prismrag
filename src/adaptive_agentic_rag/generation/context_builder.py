import re

from dataclasses import dataclass

from adaptive_agentic_rag.generation.sentence_splitter import (
    split_sentences,
)


# ============================================================
# Lightweight lexical filtering
# ============================================================

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "why",
    "will",
    "with",
    "would",
}


# ============================================================
# Data structures
# ============================================================

@dataclass
class ContextItem:

    citation_id: int

    chunk_id: str

    document_id: str

    title: str

    source: str

    url: str | None

    text: str

    score: float


@dataclass
class BuiltContext:

    text: str

    items: list[ContextItem]

    total_words: int


# ============================================================
# Context builder
# ============================================================

class ContextBuilder:

    def __init__(
        self,
        max_words: int = 1800,
        max_chunks: int = 10,
        max_chunks_per_document: int = 2,
        max_words_per_chunk: int = 240,
        min_words_per_chunk: int = 80,
    ):

        self.max_words = (
            max_words
        )

        self.max_chunks = (
            max_chunks
        )

        self.max_chunks_per_document = (
            max_chunks_per_document
        )

        self.max_words_per_chunk = (
            max_words_per_chunk
        )

        self.min_words_per_chunk = (
            min_words_per_chunk
        )


    # ========================================================
    # Tokenization
    # ========================================================

    def _tokenize(
        self,
        text: str,
    ) -> set[str]:

        tokens = re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower(),
        )


        return {

            token

            for token
            in tokens

            if (
                len(token) > 1
                and
                token not in STOPWORDS
            )
        }


    # ========================================================
    # Sentence splitting
    # ========================================================

    def _split_sentences(
        self,
        text: str,
    ) -> list[str]:

        sentences = (
            split_sentences(
                text,
                split_newlines=True,
            )
        )


        cleaned = []


        for sentence in sentences:

            sentence = (
                sentence
                .strip()
                .lstrip(
                    "-•* "
                )
                .strip()
            )


            if sentence:

                cleaned.append(
                    sentence
                )


        return cleaned


    # ========================================================
    # Sentence relevance
    # ========================================================

    def _sentence_score(
        self,
        sentence: str,
        query_terms: set[str],
        position: int,
    ) -> float:

        sentence_terms = (
            self._tokenize(
                sentence
            )
        )


        if not sentence_terms:

            return 0.0


        overlap = (
            query_terms
            &
            sentence_terms
        )


        overlap_count = (
            len(
                overlap
            )
        )


        if not query_terms:

            query_coverage = 0.0

        else:

            query_coverage = (
                overlap_count
                /
                len(
                    query_terms
                )
            )


        sentence_density = (
            overlap_count
            /
            len(
                sentence_terms
            )
        )


        position_bonus = (
            0.05
            /
            (
                position
                +
                1
            )
        )


        return (
            2.0
            *
            query_coverage

            +

            sentence_density

            +

            position_bonus
        )


    # ========================================================
    # Compact one retrieved chunk
    # ========================================================

    def _compact_text(
        self,
        text: str,
        query: str,
        word_budget: int,
    ) -> str:

        text = (
            text.strip()
        )


        if not text:

            return ""


        words = (
            text.split()
        )


        if (
            len(words)
            <=
            word_budget
        ):

            return text


        sentences = (
            self._split_sentences(
                text
            )
        )


        if not sentences:

            return " ".join(
                words[
                    :word_budget
                ]
            )


        query_terms = (
            self._tokenize(
                query
            )
        )


        if not query_terms:

            return " ".join(
                words[
                    :word_budget
                ]
            )


        scored_sentences = []


        for (
            position,
            sentence,
        ) in enumerate(
            sentences
        ):

            scored_sentences.append(
                (
                    self._sentence_score(
                        sentence=
                            sentence,
                        query_terms=
                            query_terms,
                        position=
                            position,
                    ),

                    position,

                    sentence,
                )
            )


        ranked_sentences = sorted(

            scored_sentences,

            key=lambda item: (
                -item[0],
                item[1],
            ),
        )


        selected = []

        used_words = 0


        for (
            score,
            position,
            sentence,
        ) in ranked_sentences:

            sentence_words = (
                sentence.split()
            )


            if not sentence_words:

                continue


            remaining = (
                word_budget
                -
                used_words
            )


            if remaining <= 0:

                break


            if (
                len(sentence_words)
                <=
                remaining
            ):

                selected.append(
                    (
                        position,
                        sentence,
                    )
                )


                used_words += (
                    len(
                        sentence_words
                    )
                )


            elif not selected:

                selected.append(
                    (
                        position,

                        " ".join(
                            sentence_words[
                                :remaining
                            ]
                        ),
                    )
                )


                used_words += (
                    remaining
                )


        if not selected:

            return " ".join(
                words[
                    :word_budget
                ]
            )


        selected.sort(
            key=lambda item:
                item[0]
        )


        compacted = " ".join(

            sentence

            for _,
            sentence
            in selected
        )


        compacted_words = (
            compacted.split()
        )


        if (
            len(compacted_words)
            >
            word_budget
        ):

            compacted = " ".join(
                compacted_words[
                    :word_budget
                ]
            )


        return (
            compacted.strip()
        )


    # ========================================================
    # Candidate selection
    # ========================================================

    def _select_candidates(
        self,
        results: list[dict],
    ) -> list[dict]:

        if not results:

            return []


        selected = []

        selected_ids = set()

        document_counts = {}


        # ====================================================
        # Pass 1:
        # maximize document diversity
        # ====================================================

        seen_documents = set()


        for result in results:

            if (
                len(selected)
                >=
                self.max_chunks
            ):

                break


            chunk_id = (
                result.get(
                    "id"
                )
            )

            document_id = (
                result.get(
                    "document_id"
                )
            )


            if (
                not chunk_id
                or
                not document_id
            ):

                continue


            if (
                chunk_id
                in
                selected_ids
            ):

                continue


            if (
                document_id
                in
                seen_documents
            ):

                continue


            text = (
                result
                .get(
                    "text",
                    "",
                )
                .strip()
            )


            if not text:

                continue


            selected.append(
                result
            )


            selected_ids.add(
                chunk_id
            )


            seen_documents.add(
                document_id
            )


            document_counts[
                document_id
            ] = 1


        # ====================================================
        # Pass 2:
        # allow more chunks from already selected documents
        # ====================================================

        if (
            len(selected)
            <
            self.max_chunks
        ):

            for result in results:

                if (
                    len(selected)
                    >=
                    self.max_chunks
                ):

                    break


                chunk_id = (
                    result.get(
                        "id"
                    )
                )

                document_id = (
                    result.get(
                        "document_id"
                    )
                )


                if (
                    not chunk_id
                    or
                    not document_id
                ):

                    continue


                if (
                    chunk_id
                    in
                    selected_ids
                ):

                    continue


                current_count = (
                    document_counts.get(
                        document_id,
                        0,
                    )
                )


                if (
                    current_count
                    >=
                    self.max_chunks_per_document
                ):

                    continue


                text = (
                    result
                    .get(
                        "text",
                        "",
                    )
                    .strip()
                )


                if not text:

                    continue


                selected.append(
                    result
                )


                selected_ids.add(
                    chunk_id
                )


                document_counts[
                    document_id
                ] = (
                    current_count
                    +
                    1
                )


        return selected


    # ========================================================
    # Build final context
    # ========================================================

    def build(
        self,
        results: list[dict],
        query: str = "",
    ) -> BuiltContext:

        if not results:

            return BuiltContext(
                text="",
                items=[],
                total_words=0,
            )


        candidates = (
            self._select_candidates(
                results
            )
        )


        if not candidates:

            return BuiltContext(
                text="",
                items=[],
                total_words=0,
            )


        fair_share = max(

            self.min_words_per_chunk,

            self.max_words
            //
            len(
                candidates
            ),
        )


        per_chunk_budget = min(

            fair_share,

            self.max_words_per_chunk,
        )


        selected_items = []

        total_words = 0


        for result in candidates:

            if (
                len(selected_items)
                >=
                self.max_chunks
            ):

                break


            remaining_words = (
                self.max_words
                -
                total_words
            )


            if remaining_words <= 0:

                break


            item_budget = min(
                per_chunk_budget,
                remaining_words,
            )


            if item_budget <= 0:

                break


            original_text = (
                result
                .get(
                    "text",
                    "",
                )
                .strip()
            )


            compacted_text = (
                self._compact_text(
                    text=
                        original_text,
                    query=
                        query,
                    word_budget=
                        item_budget,
                )
            )


            if not compacted_text:

                continue


            metadata = (
                result.get(
                    "metadata",
                    {},
                )
                or {}
            )


            item = (
                ContextItem(

                    citation_id=(
                        len(
                            selected_items
                        )
                        +
                        1
                    ),

                    chunk_id=(
                        result[
                            "id"
                        ]
                    ),

                    document_id=(
                        result[
                            "document_id"
                        ]
                    ),

                    title=(
                        metadata.get(
                            "title",
                            "",
                        )
                    ),

                    source=(
                        metadata.get(
                            "source",
                            "",
                        )
                    ),

                    url=(
                        metadata.get(
                            "url"
                        )
                    ),

                    text=(
                        compacted_text
                    ),

                    score=float(
                        result.get(
                            "score",
                            0.0,
                        )
                    ),
                )
            )


            selected_items.append(
                item
            )


            total_words += (
                len(
                    compacted_text
                    .split()
                )
            )


        # ====================================================
        # Render citation-aware context
        # ====================================================

        context_parts = []


        for item in selected_items:

            header = (
                f"[{item.citation_id}] "
                f"{item.title}"
            )


            if item.source:

                header += (
                    f" - "
                    f"{item.source}"
                )


            context_parts.append(
                (
                    f"{header}\n"
                    f"{item.text}"
                )
            )


        context_text = (
            "\n\n".join(
                context_parts
            )
        )


        return BuiltContext(
            text=
                context_text,
            items=
                selected_items,
            total_words=
                total_words,
        )