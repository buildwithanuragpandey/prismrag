from adaptive_agentic_rag.schemas import Document
from adaptive_agentic_rag.processing.chunker import chunk_document



def test_small_chunks_are_removed():

    doc = Document(
        id="doc_test",

        text="""
        Mac

        This is a meaningful paragraph with enough words
        to stay inside the chunking pipeline and provide
        useful retrieval context.
        """,

        metadata={}
    )


    chunks = chunk_document(
        doc
    )


    for chunk in chunks:

        assert len(
            chunk.text.split()
        ) >= 20