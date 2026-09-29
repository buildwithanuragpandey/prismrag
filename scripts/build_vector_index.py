import json

from adaptive_agentic_rag.embeddings.model import (
    EmbeddingModel
)

from adaptive_agentic_rag.vectorstore.qdrant_store import (
    QdrantVectorStore
)



def main():


    print("Loading chunks...")


    with open(
        "data/processed/processed_corpus.json",
        "r",
        encoding="utf-8"
    ) as f:

        chunks = json.load(f)



    print(
        f"Chunks loaded: {len(chunks)}"
    )



    texts = [
        chunk["text"]
        for chunk in chunks
    ]


    print(
        "Loading embedding model..."
    )


    embedder = EmbeddingModel()



    print(
        "Creating embeddings..."
    )


    embeddings = embedder.encode_documents(
        texts,
        batch_size=32,
        show_progress_bar=True
    )



    vector_size = len(
        embeddings[0]
    )


    print(
        f"Vector dimension: {vector_size}"
    )



    store = QdrantVectorStore(

        collection_name=
            "multihop_chunks",

        vector_size=
            vector_size
    )



    print(
        "Adding vectors..."
    )


    store.add_documents(

        embeddings,

        chunks
    )



    print(
        "Index created successfully"
    )



if __name__ == "__main__":
    main()