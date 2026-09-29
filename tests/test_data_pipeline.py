from datasets import load_dataset

from adaptive_agentic_rag.data.multihop_adapter import (
    convert_corpus_to_documents
)


def test_document_conversion():

    corpus = load_dataset(
        "yixuantt/MultiHopRAG",
        "corpus"
    )["train"]


    documents = convert_corpus_to_documents(
        corpus
    )


    assert len(documents) == 609

    assert documents[0].id == "doc_0000"

    assert len(documents[0].text) > 0