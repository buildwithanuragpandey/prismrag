from adaptive_agentic_rag.embeddings.model import (
    EmbeddingModel
)



model = EmbeddingModel()


vector = model.encode(
    [
        "hello world asdf"
    ]
)


print(
    len(vector[0])
)