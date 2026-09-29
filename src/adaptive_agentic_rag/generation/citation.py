import re

from dataclasses import dataclass

from adaptive_agentic_rag.generation.context_builder import (
    BuiltContext
)


@dataclass
class CitationValidation:

    cited_ids: list[int]

    invalid_ids: list[int]

    has_citations: bool

    valid: bool


def extract_citation_ids(
    answer: str
) -> list[int]:

    matches = re.findall(
        r"\[(\d+)\]",
        answer
    )


    citation_ids = []

    seen = set()


    for match in matches:

        citation_id = int(
            match
        )


        if citation_id in seen:
            continue


        seen.add(
            citation_id
        )

        citation_ids.append(
            citation_id
        )


    return citation_ids


def validate_citations(
    answer: str,
    context: BuiltContext
) -> CitationValidation:

    cited_ids = extract_citation_ids(
        answer
    )


    valid_context_ids = {

        item.citation_id

        for item in context.items

    }


    invalid_ids = [

        citation_id

        for citation_id in cited_ids

        if citation_id
        not in valid_context_ids

    ]


    has_citations = (
        len(cited_ids) > 0
    )


    valid = (

        has_citations

        and

        len(invalid_ids) == 0

    )


    return CitationValidation(

        cited_ids=cited_ids,

        invalid_ids=invalid_ids,

        has_citations=has_citations,

        valid=valid

    )