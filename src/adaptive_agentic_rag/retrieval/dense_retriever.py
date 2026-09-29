from adaptive_agentic_rag.embeddings.model import (
    EmbeddingModel
)

from adaptive_agentic_rag.vectorstore.qdrant_store import (
    QdrantVectorStore
)


class DenseRetriever:


    def __init__(
        self,
        collection_name: str = "multihop_chunks_v2"
    ):

        self.embedder = EmbeddingModel()

        self.store = QdrantVectorStore(
            collection_name=collection_name
        )

        # Query embedding cache
        self.query_cache = {}



    def embed_queries(
        self,
        queries: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False
    ):

        return self.embedder.encode_queries(
            queries,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar
        )



    def search_by_vector(
        self,
        query_vector,
        top_k: int = 5
    ):

        response = self.store.client.query_points(

            collection_name=self.store.collection_name,

            query=query_vector.tolist(),

            limit=top_k,

            with_payload=True,
            with_vectors=True


        )


        return [

            {
                "score": point.score,

                **point.payload,

                "vector": point.vector

            }

            for point in response.points

        ]

    def search(
        self,
        query: str,
        top_k: int = 5
    ):


        if query in self.query_cache:

            query_vector = self.query_cache[query]


        else:

            query_vector = (
                self.embedder
                .encode_queries(
                    [query]
                )[0]
            )


            self.query_cache[query] = query_vector



        return self.search_by_vector(

            query_vector=query_vector,

            top_k=top_k

        )



    def close(self):

        self.store.client.close()