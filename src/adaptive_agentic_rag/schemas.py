from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Document:
    """
    A searchable document in the knowledge base.
    """

    id: str
    text: str
    metadata: Dict

@dataclass
class Chunk:
    """
    A retrievable unit created from a document.
    """

    id: str
    document_id: str
    text: str
    metadata: Dict
@dataclass
class Evidence:
    """
    Ground truth evidence supporting an answer.
    """

    document_id: str
    text: str


@dataclass
class EvaluationExample:
    """
    One benchmark example.
    """

    id: str
    question: str
    answer: str
    evidence_ids: List[str]
    question_type: str
    