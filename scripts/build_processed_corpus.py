from datasets import load_dataset
import json

from adaptive_agentic_rag.data.multihop_adapter import (
    convert_corpus_to_documents
)

from adaptive_agentic_rag.processing.chunker import (
    chunk_document
)
from adaptive_agentic_rag.processing.cleaner import (
    clean_document
)

def main():

    print("Loading corpus...")

    corpus = load_dataset(
        "yixuantt/MultiHopRAG",
        "corpus"
    )["train"]


    print("Converting documents...")

    documents = convert_corpus_to_documents(
        corpus
    )


    print(
        f"Documents loaded: {len(documents)}"
    )


    all_chunks = []


    print("Chunking documents...")


    for document in documents:


        cleaned_document = clean_document(
            document
        )


        chunks = chunk_document(
            cleaned_document
        )

        all_chunks.extend(chunks)



    print(
        f"Total chunks: {len(all_chunks)}"
    )


    output = []


    for chunk in all_chunks:

        output.append(
            {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "metadata": chunk.metadata
            }
        )


    with open(
        "data/processed/processed_corpus.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )


    print(
        "Saved processed corpus"
    )


if __name__ == "__main__":
    main()