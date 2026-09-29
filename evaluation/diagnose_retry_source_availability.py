import json
import re

from collections import Counter
from pathlib import Path

from adaptive_agentic_rag.orchestration.explicit_source_coverage import (
    ExplicitSourceCoverageGuard,
)


DIAGNOSTIC_PATH = Path(
    "evaluation/results/"
    "adaptive_retry_frozen500_diagnostic.json"
)


CORPUS_PATH = Path(
    "data/processed/"
    "processed_corpus_v2.json"
)


OUTPUT_PATH = Path(
    "evaluation/results/"
    "retry_source_availability_diagnostic.json"
)


# ============================================================
# Generic JSON loading
# ============================================================

def load_json(
    path: Path,
):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


# ============================================================
# Corpus extraction
# ============================================================

def corpus_items(
    payload,
):

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


def source_from_item(
    item,
) -> str | None:

    if not isinstance(
        item,
        dict,
    ):

        return None


    # ========================================================
    # Top-level source
    # ========================================================

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


    # ========================================================
    # Metadata source
    # ========================================================

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


def build_corpus_source_catalog(
    payload,
):

    items = (
        corpus_items(
            payload
        )
    )


    counts = (
        Counter()
    )


    for item in items:

        source = (
            source_from_item(
                item
            )
        )


        if not source:

            continue


        counts[
            source
        ] += 1


    return counts


# ============================================================
# Alias catalog
# ============================================================

def normalize(
    value: str,
) -> str:

    return (
        ExplicitSourceCoverageGuard
        ._normalize(
            value
        )
    )


def build_alias_lookup(
    source_counts,
):

    alias_lookup = {}


    for source in (
        source_counts
    ):

        aliases = (
            ExplicitSourceCoverageGuard
            ._source_aliases(
                source
            )
        )


        for alias in aliases:

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


    return alias_lookup


# ============================================================
# Source matching
# ============================================================

def match_corpus_source(
    missing_source,
    *,
    alias_lookup,
):

    normalized = (
        normalize(
            missing_source
        )
    )


    if not normalized:

        return []


    # ========================================================
    # Exact alias match first
    # ========================================================

    exact = (
        alias_lookup.get(
            normalized
        )
    )


    if exact:

        return list(
            exact
        )


    # ========================================================
    # Conservative fallback.
    #
    # Only normalized whole-word alias containment.
    # ========================================================

    matches = []


    for (
        alias,
        sources,
    ) in alias_lookup.items():

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
            normalized,
        ):

            for source in sources:

                if source not in matches:

                    matches.append(
                        source
                    )


    return matches


# ============================================================
# Retry candidates
# ============================================================

def retry_candidates(
    payload,
):

    candidates = (
        payload.get(
            "retry_candidates"
        )
    )


    if isinstance(
        candidates,
        list,
    ):

        return candidates


    records = (
        payload.get(
            "records",
            [],
        )
    )


    return [
        record

        for record
        in records

        if (
            record.get(
                "rewrite_attempted"
            )
            or
            record.get(
                "initial_route"
            )
            ==
            "rewrite"
        )
    ]


# ============================================================
# One retry
# ============================================================

def inspect_retry(
    record,
    *,
    alias_lookup,
    source_counts,
):

    missing_sources = list(
        record.get(
            "initial_missing_sources",
            [],
        )
        or []
    )


    source_checks = []


    for missing_source in (
        missing_sources
    ):

        matches = (
            match_corpus_source(
                missing_source,
                alias_lookup=
                    alias_lookup,
            )
        )


        available = bool(
            matches
        )


        matched_chunk_count = sum(
            source_counts.get(
                source,
                0,
            )

            for source
            in matches
        )


        source_checks.append(
            {
                "required_source":
                    missing_source,

                "available_in_corpus":
                    available,

                "matched_corpus_sources":
                    matches,

                "matched_chunk_count":
                    matched_chunk_count,
            }
        )


    all_missing_sources_available = (
        bool(
            source_checks
        )
        and
        all(
            check[
                "available_in_corpus"
            ]

            for check
            in source_checks
        )
    )


    any_missing_source_available = (
        any(
            check[
                "available_in_corpus"
            ]

            for check
            in source_checks
        )
    )


    return {
        "id":
            record.get(
                "id"
            ),

        "question_type":
            record.get(
                "question_type"
            ),

        "question":
            record.get(
                "question"
            ),

        "initial_required_sources":
            record.get(
                "initial_required_sources",
                [],
            ),

        "initial_covered_sources":
            record.get(
                "initial_covered_sources",
                [],
            ),

        "initial_missing_sources":
            missing_sources,

        "rewrite_rescued":
            bool(
                record.get(
                    "rewrite_rescued",
                    False,
                )
            ),

        "source_checks":
            source_checks,

        "all_missing_sources_available_in_corpus":
            all_missing_sources_available,

        "any_missing_source_available_in_corpus":
            any_missing_source_available,
    }


# ============================================================
# Summary
# ============================================================

def summarize(
    records,
):

    total = (
        len(
            records
        )
    )


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


    null_records = [
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


    corpus_available = [
        record

        for record
        in records

        if record[
            "all_missing_sources_available_in_corpus"
        ]
    ]


    corpus_unavailable = [
        record

        for record
        in records

        if not record[
            "all_missing_sources_available_in_corpus"
        ]
    ]


    answerable_available = [
        record

        for record
        in answerable

        if record[
            "all_missing_sources_available_in_corpus"
        ]
    ]


    answerable_unavailable = [
        record

        for record
        in answerable

        if not record[
            "all_missing_sources_available_in_corpus"
        ]
    ]


    null_available = [
        record

        for record
        in null_records

        if record[
            "all_missing_sources_available_in_corpus"
        ]
    ]


    null_unavailable = [
        record

        for record
        in null_records

        if not record[
            "all_missing_sources_available_in_corpus"
        ]
    ]


    rescued = [
        record

        for record
        in records

        if record[
            "rewrite_rescued"
        ]
    ]


    rescued_available = [
        record

        for record
        in rescued

        if record[
            "all_missing_sources_available_in_corpus"
        ]
    ]


    missing_source_frequency = (
        Counter()
    )


    unavailable_source_frequency = (
        Counter()
    )


    available_source_frequency = (
        Counter()
    )


    for record in records:

        for check in (
            record[
                "source_checks"
            ]
        ):

            source = (
                check[
                    "required_source"
                ]
            )


            missing_source_frequency[
                source
            ] += 1


            if (
                check[
                    "available_in_corpus"
                ]
            ):

                available_source_frequency[
                    source
                ] += 1


            else:

                unavailable_source_frequency[
                    source
                ] += 1


    return {
        "total_retry_candidates":
            total,

        "answerable_retry_candidates":
            len(
                answerable
            ),

        "null_retry_candidates":
            len(
                null_records
            ),

        # ----------------------------------------------------
        # Corpus availability
        # ----------------------------------------------------

        "retry_all_missing_sources_available":
            len(
                corpus_available
            ),

        "retry_missing_source_unavailable":
            len(
                corpus_unavailable
            ),

        "answerable_source_available":
            len(
                answerable_available
            ),

        "answerable_source_unavailable":
            len(
                answerable_unavailable
            ),

        "null_source_available":
            len(
                null_available
            ),

        "null_source_unavailable":
            len(
                null_unavailable
            ),

        # ----------------------------------------------------
        # Existing rewrite outcome
        # ----------------------------------------------------

        "rewrite_rescues":
            len(
                rescued
            ),

        "rescues_where_source_available":
            len(
                rescued_available
            ),

        # ----------------------------------------------------
        # Source frequencies
        # ----------------------------------------------------

        "missing_source_frequency":
            dict(
                missing_source_frequency
                .most_common()
            ),

        "available_missing_source_frequency":
            dict(
                available_source_frequency
                .most_common()
            ),

        "unavailable_missing_source_frequency":
            dict(
                unavailable_source_frequency
                .most_common()
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    diagnostic_payload = (
        load_json(
            DIAGNOSTIC_PATH
        )
    )


    corpus_payload = (
        load_json(
            CORPUS_PATH
        )
    )


    source_counts = (
        build_corpus_source_catalog(
            corpus_payload
        )
    )


    alias_lookup = (
        build_alias_lookup(
            source_counts
        )
    )


    candidates = (
        retry_candidates(
            diagnostic_payload
        )
    )


    records = [
        inspect_retry(
            candidate,
            alias_lookup=
                alias_lookup,
            source_counts=
                source_counts,
        )

        for candidate
        in candidates
    ]


    summary = (
        summarize(
            records
        )
    )


    print(
        "\n"
        +
        "=" * 100
    )


    print(
        "RETRY SOURCE AVAILABILITY DIAGNOSTIC"
    )


    print(
        "=" * 100
    )


    print(
        "Corpus sources:",
        len(
            source_counts
        )
    )


    print(
        "Retry candidates:",
        len(
            records
        )
    )


    for (
        index,
        record,
    ) in enumerate(
        records,
        start=1,
    ):

        print(
            "\n"
            +
            "-" * 100
        )


        print(
            (
                f"{index}/"
                f"{len(records)} | "
                f"{record['question_type']} | "
                f"{record['id']}"
            )
        )


        print(
            "Missing:",
            record[
                "initial_missing_sources"
            ],
        )


        for check in (
            record[
                "source_checks"
            ]
        ):

            print(
                (
                    f"  {check['required_source']}: "
                    f"available="
                    f"{check['available_in_corpus']} "
                    f"matches="
                    f"{check['matched_corpus_sources']} "
                    f"chunks="
                    f"{check['matched_chunk_count']}"
                )
            )


        print(
            "Rewrite rescued:",
            record[
                "rewrite_rescued"
            ],
        )


    print(
        "\n"
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
                "diagnostic_path":
                    str(
                        DIAGNOSTIC_PATH
                    ),

                "corpus_path":
                    str(
                        CORPUS_PATH
                    ),

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