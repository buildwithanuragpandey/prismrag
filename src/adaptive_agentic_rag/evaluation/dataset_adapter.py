import re
from urllib.parse import urlsplit, urlunsplit


def normalize_text_key(
    value: str
) -> str:

    value = value.strip().casefold()

    return re.sub(
        r"\s+",
        " ",
        value
    )


def normalize_url(
    url: str
) -> str:

    url = url.strip()

    parts = urlsplit(url)

    path = parts.path.rstrip("/")

    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            path,
            parts.query,
            ""
        )
    )


def make_document_key(
    source: str,
    title: str,
    url: str | None = None
) -> str:

    if url and url.strip():

        return (
            "url::"
            + normalize_url(url)
        )

    source = normalize_text_key(
        source
    )

    title = normalize_text_key(
        title
    )

    return (
        f"title::{source}::{title}"
    )


def get_relevant_document_keys(
    evidence_list: list[dict]
) -> set[str]:

    return {
        make_document_key(
            source=evidence["source"],
            title=evidence["title"],
            url=evidence.get("url")
        )
        for evidence in evidence_list
    }


def get_retrieved_document_keys(
    results: list[dict]
) -> list[str]:

    keys = []

    for result in results:

        metadata = result[
            "metadata"
        ]

        key = make_document_key(
            source=metadata["source"],
            title=metadata["title"],
            url=metadata.get("url")
        )

        keys.append(key)

    return keys