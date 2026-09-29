import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from adaptive_agentic_rag.embeddings.model import (
    EmbeddingModel,
)


CORPUS_PATH = Path(
    "data/processed/processed_corpus_v2.json"
)

QDRANT_PATH = "data/qdrant"

COLLECTION_NAME = "multihop_chunks_v2b"

EMBEDDING_BATCH_SIZE = 32
UPSERT_BATCH_SIZE = 256


def build_embedding_text(
    chunk: dict,
) -> str:

    metadata = (
        chunk.get("metadata", {})
        or {}
    )

    title = (
        metadata.get("title", "")
        or ""
    ).strip()

    source = (
        metadata.get("source", "")
        or ""
    ).strip()

    text = (
        chunk.get("text", "")
        or ""
    ).strip()

    parts = [
        part
        for part in (
            title,
            source,
            text,
        )
        if part
    ]

    return "\n".join(parts)


def main():

    print("Loading V2 corpus...")

    with open(
        CORPUS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        chunks = json.load(file)


    print("Chunks:", len(chunks))


    texts = [
        build_embedding_text(chunk)
        for chunk in chunks
    ]


    print("\nExample representation:\n")

    print(
        texts[0][:1000]
    )


    print("\nLoading embedding model...")

    embedder = EmbeddingModel()


    print(
        "\nEncoding title + source + chunk..."
    )

    embeddings = (
        embedder.encode_documents(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
        )
    )


    if len(embeddings) != len(chunks):

        raise RuntimeError(
            "Embedding count mismatch."
        )


    vector_size = len(
        embeddings[0]
    )


    print(
        "\nVector dimension:",
        vector_size,
    )


    client = QdrantClient(
        path=QDRANT_PATH
    )


    try:

        if client.collection_exists(
            COLLECTION_NAME
        ):

            print(
                "Deleting existing:",
                COLLECTION_NAME,
            )

            client.delete_collection(
                collection_name=COLLECTION_NAME
            )


        print(
            "Creating:",
            COLLECTION_NAME,
        )


        client.create_collection(

            collection_name=
                COLLECTION_NAME,

            vectors_config=
                VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
        )


        total = len(chunks)


        print("\nUploading vectors...")


        for start in range(
            0,
            total,
            UPSERT_BATCH_SIZE,
        ):

            end = min(
                start + UPSERT_BATCH_SIZE,
                total,
            )


            points = []


            for index in range(
                start,
                end,
            ):

                points.append(

                    PointStruct(

                        id=index,

                        vector=
                            embeddings[
                                index
                            ].tolist(),

                        # Important:
                        # payload remains the original chunk.
                        # Metadata is used for embedding only.
                        payload=
                            chunks[
                                index
                            ],
                    )
                )


            client.upsert(

                collection_name=
                    COLLECTION_NAME,

                points=points,

                wait=True,
            )


            print(
                f"Uploaded {end}/{total}"
            )


        info = client.get_collection(
            COLLECTION_NAME
        )


        print(
            "\n"
            +
            "=" * 80
        )

        print(
            "V2-B VECTOR INDEX"
        )

        print(
            "=" * 80
        )

        print(
            "Collection:",
            COLLECTION_NAME,
        )

        print(
            "Expected points:",
            total,
        )

        print(
            "Qdrant points:",
            info.points_count,
        )

        print(
            "Vector dimension:",
            vector_size,
        )


        if info.points_count != total:

            raise RuntimeError(
                "Qdrant point count mismatch."
            )


        print(
            "\nV2-B index created successfully."
        )


    finally:

        client.close()


if __name__ == "__main__":
    main()