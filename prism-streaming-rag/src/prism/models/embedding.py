"""
PRISM Embedding Model Adapter

Provides a clean interface for embedding text.
Backed by sentence-transformers (configurable model name).
Caches models at module level (loaded once).

Interface:
    EmbeddingModel
        .encode(text: str) -> List[float]
        .encode_batch(texts: List[str]) -> List[List[float]]
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

_model_cache: dict[str, any] = {}


import hashlib
import numpy as np

class MockEmbeddingModel:
    """Deterministic fallback embedding generator when SentenceTransformer is unavailable."""
    def __init__(self, dimension: int = 384):
        self.dim = dimension

    def encode(self, text, *args, **kwargs):
        if isinstance(text, list):
            return np.array([self.encode(t) for t in text])
        # Generate deterministic vector from md5 hash
        h = hashlib.md5(text.encode("utf-8")).digest()
        seed = int.from_bytes(h[:4], "big")
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dim)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim


class EmbeddingModel:
    """
    Adapter for sentence-transformers embedding model.

    Args:
        model_name: HuggingFace model name
        device: "cpu" | "cuda" | "mps"
        batch_size: Batch size for encode_batch
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        """Lazy load the model with graceful fallback."""
        if self._model is not None:
            return

        global _model_cache
        cache_key = f"{self.model_name}:{self.device}"

        if cache_key in _model_cache:
            self._model = _model_cache[cache_key]
            return

        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s on %s", self.model_name, self.device)
            model = SentenceTransformer(self.model_name, device=self.device)
            _model_cache[cache_key] = model
            self._model = model
            logger.info("Embedding model loaded.")
        except Exception as exc:
            logger.warning("Failed to load SentenceTransformer (%s). Falling back to MockEmbeddingModel.", exc)
            self._model = MockEmbeddingModel(dimension=384)
            _model_cache[cache_key] = self._model

    def encode(self, text: str) -> List[float]:
        """Encode a single text string into a float vector."""
        self._load()
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode a batch of texts."""
        self._load()
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
        )
        return [e.tolist() for e in embeddings]

    @property
    def dimension(self) -> int:
        """Return embedding dimensionality."""
        self._load()
        return self._model.get_sentence_embedding_dimension()

    def __call__(self, text: str) -> List[float]:
        """Allow using model as a callable (for SemanticNoveltyDetector)."""
        return self.encode(text)
