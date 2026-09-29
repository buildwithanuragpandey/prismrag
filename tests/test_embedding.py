import numpy as np

import adaptive_agentic_rag.embeddings.model as embedding_module

from adaptive_agentic_rag.embeddings.model import (
    EmbeddingModel,
)


class FakeSentenceTransformer:

    def __init__(
        self,
        model_name,
        device=None,
    ):

        self.model_name = (
            model_name
        )

        self.device = (
            device
        )

        self.calls = []


    def encode(
        self,
        texts,
        batch_size=32,
        normalize_embeddings=False,
        show_progress_bar=False,
        prompt_name=None,
    ):

        self.calls.append(
            {
                "texts":
                    list(
                        texts
                    ),

                "batch_size":
                    batch_size,

                "normalize_embeddings":
                    normalize_embeddings,

                "show_progress_bar":
                    show_progress_bar,

                "prompt_name":
                    prompt_name,
            }
        )


        return np.asarray(
            [
                [
                    float(
                        index + 1
                    ),
                    0.0,
                    0.0,
                ]

                for index, _
                in enumerate(
                    texts
                )
            ],
            dtype=float,
        )


def test_document_embeddings_use_document_mode(
    monkeypatch,
):

    monkeypatch.setattr(
        embedding_module,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )


    model = (
        EmbeddingModel()
    )


    vectors = (
        model.encode_documents(
            [
                "Sam Bankman-Fried founded FTX",
                "Amazon Cyber Monday deals",
            ],
            batch_size=16,
        )
    )


    assert vectors.shape == (
        2,
        3,
    )


    assert len(
        model.model.calls
    ) == 1


    call = (
        model.model.calls[
            0
        ]
    )


    assert call[
        "batch_size"
    ] == 16


    assert (
        call[
            "normalize_embeddings"
        ]
        is True
    )


    assert (
        call[
            "prompt_name"
        ]
        is None
    )


def test_query_embeddings_use_query_prompt(
    monkeypatch,
):

    monkeypatch.setattr(
        embedding_module,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )


    model = (
        EmbeddingModel()
    )


    vectors = (
        model.encode_queries(
            [
                "Who founded FTX?",
                "What are Amazon Cyber Monday deals?",
            ],
            batch_size=8,
        )
    )


    assert vectors.shape == (
        2,
        3,
    )


    assert len(
        model.model.calls
    ) == 1


    call = (
        model.model.calls[
            0
        ]
    )


    assert call[
        "batch_size"
    ] == 8


    assert (
        call[
            "normalize_embeddings"
        ]
        is True
    )


    assert (
        call[
            "prompt_name"
        ]
        ==
        "query"
    )


def test_embedding_model_passes_device(
    monkeypatch,
):

    monkeypatch.setattr(
        embedding_module,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )


    model = (
        EmbeddingModel(
            device="cpu"
        )
    )


    assert (
        model.model.device
        ==
        "cpu"
    )