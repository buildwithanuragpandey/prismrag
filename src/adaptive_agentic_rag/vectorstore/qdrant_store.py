from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)



class QdrantVectorStore:


    def __init__(
        self,
        collection_name: str,
        vector_size: int | None = None,
        create: bool = False
    ):


        self.client = QdrantClient(
            path="data/qdrant"
        )


        self.collection_name = collection_name



        if create:

            if vector_size is None:
                raise ValueError(
                    "vector_size required when creating collection"
                )


            if not self.client.collection_exists(
                collection_name
            ):

                self.client.create_collection(

                    collection_name=

                        collection_name,


                    vectors_config=

                        VectorParams(

                            size=vector_size,

                            distance=Distance.COSINE

                        )
                )



    def add_documents(
        self,
        embeddings,
        documents
    ):


        points = []


        for idx, (
            embedding,
            document
        ) in enumerate(
            zip(
                embeddings,
                documents
            )
        ):


            points.append(

                PointStruct(

                    id=idx,

                    vector=
                        embedding.tolist(),

                    payload=document

                )
            )


        self.client.upsert(

            collection_name=
                self.collection_name,

            points=points
        )
    def get_vectors(
        self,
        ids: list[str]
    ):

        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=ids,
            with_payload=True,
            with_vectors=True
        )


        return points