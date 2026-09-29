from sentence_transformers import SentenceTransformer


MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


class EmbeddingModel:

    def __init__(
        self,
        device: str | None = None
    ):

        self.model = SentenceTransformer(
            MODEL_NAME,
            device=device
        )

    def encode_documents(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False
    ):

        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar
        )

    def encode_queries(
        self,
        queries: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False
    ):

        return self.model.encode(
            queries,
            batch_size=batch_size,
            normalize_embeddings=True,
            prompt_name="query",
            show_progress_bar=show_progress_bar
        )