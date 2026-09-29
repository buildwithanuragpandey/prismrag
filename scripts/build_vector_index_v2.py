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


# ============================================================
# Configuration
# ============================================================

CORPUS_PATH = Path(
    "data/processed/processed_corpus_v2.json"
)

QDRANT_PATH = "data/qdrant"

COLLECTION_NAME = (
    "multihop_chunks_v2"
)

EMBEDDING_BATCH_SIZE = 32

UPSERT_BATCH_SIZE = 256


# ============================================================
# Main
# ============================================================

def main():

    print(
        "Loading V2-A corpus..."
    )


    with open(
        CORPUS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        chunks = json.load(
            file
        )


    print(
        "Chunks:",
        len(
            chunks
        )
    )


    # ========================================================
    # IMPORTANT:
    #
    # V2-A changes ONLY chunking.
    #
    # Do NOT add title/source to embedding representation here.
    # That will be a separate experiment (V2-B).
    # ========================================================

    texts = [
        chunk["text"]
        for chunk
        in chunks
    ]


    print(
        "\nLoading embedding model..."
    )


    embedder = (
        EmbeddingModel()
    )


    print(
        "\nEncoding V2-A chunks..."
    )


    embeddings = (
        embedder.encode_documents(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
        )
    )


    if len(
        embeddings
    ) != len(
        chunks
    ):

        raise RuntimeError(
            "Embedding count does not match chunk count."
        )


    vector_size = len(
        embeddings[0]
    )


    print(
        "\nVector dimension:",
        vector_size
    )


    # ========================================================
    # Qdrant
    #
    # Only V2 collection is recreated.
    #
    # Existing V1 collection:
    #     multihop_chunks
    #
    # remains untouched.
    # ========================================================

    print(
        "\nOpening Qdrant..."
    )


    client = QdrantClient(
        path=QDRANT_PATH
    )


    try:

        if client.collection_exists(
            COLLECTION_NAME
        ):

            print(
                "Removing existing V2 collection:",
                COLLECTION_NAME,
            )

            client.delete_collection(
                collection_name=
                    COLLECTION_NAME
            )


        print(
            "Creating collection:",
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


        # ====================================================
        # Upsert in batches
        # ====================================================

        print(
            "\nUploading vectors..."
        )


        total = len(
            chunks
        )


        for start in range(
            0,
            total,
            UPSERT_BATCH_SIZE,
        ):

            end = min(
                start
                +
                UPSERT_BATCH_SIZE,

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
                f"Uploaded "
                f"{end}/{total}"
            )


        # ====================================================
        # Verify count
        # ====================================================

        collection_info = (
            client.get_collection(
                COLLECTION_NAME
            )
        )


        points_count = (
            collection_info.points_count
        )


        print(
            "\n"
            +
            "=" * 80
        )

        print(
            "V2-A VECTOR INDEX"
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
            points_count,
        )

        print(
            "Vector dimension:",
            vector_size,
        )


        if (
            points_count
            !=
            total
        ):

            raise RuntimeError(
                "Qdrant point count mismatch."
            )


        print(
            "\nV2-A index created successfully."
        )


    finally:

        client.close()


if __name__ == "__main__":

    main()