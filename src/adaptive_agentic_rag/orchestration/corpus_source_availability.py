import json

from dataclasses import dataclass
from pathlib import Path

from adaptive_agentic_rag.orchestration.explicit_source_coverage import (
    ExplicitSourceCoverageGuard,
)


DEFAULT_CORPUS_PATH = Path(
    "data/processed/"
    "processed_corpus_v2.json"
)


@dataclass
class CorpusSourceAvailabilityResult:
    requested_sources: list[str]
    available_sources: list[str]
    unavailable_sources: list[str]
    matched_catalog_sources: dict[str, list[str]]

    @property
    def all_available(
        self,
    ) -> bool:

        return (
            bool(
                self.requested_sources
            )
            and
            not self.unavailable_sources
        )


class CorpusSourceAvailability:
    """
    Checks whether explicitly required publishers actually
    exist anywhere in the frozen retrieval corpus.

    Purpose
    -------
    Prevent structurally impossible retries.

    Example:

        Query requires:
            CNBC + Forbes

        Current context contains:
            CNBC

        But corpus contains:
            no Forbes documents

        Retrying retrieval cannot possibly recover Forbes.

        -> ABSTAIN

    This component:

    - does not use gold labels
    - does not use question_type
    - does not use embeddings
    - does not use an LLM
    - does not inspect answer correctness

    It only answers:

        "Can this source exist in retrieval results at all?"

    Loading
    -------
    The processed corpus is loaded lazily once per component
    instance. Only the source catalog is retained.
    """

    def __init__(
        self,
        *,
        corpus_path: str | Path = DEFAULT_CORPUS_PATH,
    ):

        self.corpus_path = Path(
            corpus_path
        )

        self._catalog_sources = None

        self._alias_lookup = None


    # ========================================================
    # Corpus structure
    # ========================================================

    @staticmethod
    def _corpus_items(
        payload,
    ) -> list[dict]:

        if isinstance(
            payload,
            list,
        ):

            return payload


        if isinstance(
            payload,
            dict,
        ):

            for key in (
                "chunks",
                "documents",
                "records",
                "items",
                "data",
            ):

                value = (
                    payload.get(
                        key
                    )
                )


                if isinstance(
                    value,
                    list,
                ):

                    return value


        raise ValueError(
            (
                "Unsupported processed corpus structure: "
                f"{type(payload).__name__}"
            )
        )


    @staticmethod
    def _source_from_item(
        item: dict,
    ) -> str | None:

        if not isinstance(
            item,
            dict,
        ):

            return None


        # ====================================================
        # Top-level source
        # ====================================================

        source = (
            item.get(
                "source"
            )
        )


        if isinstance(
            source,
            str,
        ):

            source = (
                source.strip()
            )


            if source:

                return source


        # ====================================================
        # Chunk metadata source
        # ====================================================

        metadata = (
            item.get(
                "metadata"
            )
        )


        if isinstance(
            metadata,
            dict,
        ):

            source = (
                metadata.get(
                    "source"
                )
            )


            if isinstance(
                source,
                str,
            ):

                source = (
                    source.strip()
                )


                if source:

                    return source


        return None


    # ========================================================
    # Source identity
    # ========================================================

    @staticmethod
    def _normalize(
        source: str,
    ) -> str:

        # Keep identity semantics exactly aligned with the
        # ExplicitSourceCoverageGuard.

        return (
            ExplicitSourceCoverageGuard
            ._normalize(
                source
            )
        )


    @classmethod
    def _aliases(
        cls,
        source: str,
    ) -> set[str]:

        # Reuse the source identity semantics already frozen
        # in ExplicitSourceCoverageGuard.
        #
        # Example:
        #
        # The New York Times
        # -> the new york times
        # -> new york times
        #
        # Cnbc | World Business News Leader
        # -> full alias
        # -> cnbc

        return (
            ExplicitSourceCoverageGuard
            ._source_aliases(
                source
            )
        )


    # ========================================================
    # Lazy catalog loading
    # ========================================================

    def _ensure_loaded(
        self,
    ) -> None:

        if (
            self._catalog_sources
            is not None
            and
            self._alias_lookup
            is not None
        ):

            return


        if not self.corpus_path.exists():

            raise FileNotFoundError(
                (
                    "Cannot build corpus source catalog. "
                    "Processed corpus does not exist: "
                    f"{self.corpus_path}"
                )
            )


        with open(
            self.corpus_path,
            "r",
            encoding="utf-8",
        ) as file:

            payload = (
                json.load(
                    file
                )
            )


        items = (
            self._corpus_items(
                payload
            )
        )


        catalog_sources = []

        seen_sources = set()


        for item in items:

            source = (
                self._source_from_item(
                    item
                )
            )


            if not source:

                continue


            normalized = (
                self._normalize(
                    source
                )
            )


            if not normalized:

                continue


            if normalized in seen_sources:

                continue


            seen_sources.add(
                normalized
            )


            catalog_sources.append(
                source
            )


        alias_lookup = {}


        for source in (
            catalog_sources
        ):

            for alias in (
                self._aliases(
                    source
                )
            ):

                if not alias:

                    continue


                alias_lookup.setdefault(
                    alias,
                    []
                )


                if (
                    source
                    not in
                    alias_lookup[
                        alias
                    ]
                ):

                    alias_lookup[
                        alias
                    ].append(
                        source
                    )


        self._catalog_sources = (
            catalog_sources
        )

        self._alias_lookup = (
            alias_lookup
        )


    # ========================================================
    # Matching
    # ========================================================

    def match(
        self,
        source: str,
    ) -> list[str]:

        self._ensure_loaded()


        normalized = (
            self._normalize(
                source
            )
        )


        if not normalized:

            return []


        # Exact canonical/alias identity only.
        #
        # We deliberately do NOT use fuzzy lexical matching
        # here. The Source Guard has already extracted the
        # publisher identity.

        return list(
            self._alias_lookup.get(
                normalized,
                [],
            )
        )


    # ========================================================
    # Public availability check
    # ========================================================

    def check(
        self,
        sources: list[str],
    ) -> CorpusSourceAvailabilityResult:

        self._ensure_loaded()


        requested_sources = []

        seen = set()


        for source in (
            sources
            or []
        ):

            source = str(
                source
            ).strip()


            normalized = (
                self._normalize(
                    source
                )
            )


            if not normalized:

                continue


            if normalized in seen:

                continue


            seen.add(
                normalized
            )


            requested_sources.append(
                source
            )


        available_sources = []

        unavailable_sources = []

        matched_catalog_sources = {}


        for source in (
            requested_sources
        ):

            matches = (
                self.match(
                    source
                )
            )


            matched_catalog_sources[
                source
            ] = (
                matches
            )


            if matches:

                available_sources.append(
                    source
                )


            else:

                unavailable_sources.append(
                    source
                )


        return CorpusSourceAvailabilityResult(
            requested_sources=
                requested_sources,

            available_sources=
                available_sources,

            unavailable_sources=
                unavailable_sources,

            matched_catalog_sources=
                matched_catalog_sources,
        )