"""
PRISM Configuration Loader

Loads YAML config + .env overrides.
All components receive the PRISMConfig instance via dependency injection.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# ──────────────────────────────────────────────────────────────
# Sub-config models
# ──────────────────────────────────────────────────────────────

class EmbeddingModelConfig(BaseModel):
    name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32


class RerankerConfig(BaseModel):
    name: str = "BAAI/bge-reranker-base"
    device: str = "cpu"
    enabled: bool = True


class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "llama3.2:3b"
    temperature: float = 0.1
    max_tokens: int = 1024


class ModelsConfig(BaseModel):
    embedding: EmbeddingModelConfig = Field(default_factory=EmbeddingModelConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    intent_classifier: Dict[str, Any] = Field(default_factory=lambda: {"provider": "llm"})


class RetrievalConfig(BaseModel):
    top_k_dense: int = 20
    top_k_bm25: int = 20
    top_k_rrf: int = 20
    top_k_reranked: int = 5
    rrf_k: int = 60
    mmr_lambda: float = 0.7
    mmr_top_k: int = 5
    parallel_subqueries: bool = True


class FAISSConfig(BaseModel):
    index_path: str = "data/faiss_index/index.faiss"
    metadata_path: str = "data/faiss_index/metadata.json"


class QdrantConfig(BaseModel):
    host: str = "localhost"
    port: int = 6333
    collection: str = "prism_chunks_v1"


class VectorstoreConfig(BaseModel):
    backend: str = "faiss"
    faiss: FAISSConfig = Field(default_factory=FAISSConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)


class CorpusConfig(BaseModel):
    dataset_name: str = "yixuantt/MultiHopRAG"
    dataset_revision: str = "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82"
    processed_corpus_path: str = "data/processed/corpus.json"
    bm25_index_path: str = "data/processed/bm25_index.pkl"
    chunk_size: int = 1000
    chunk_overlap: int = 100
    min_chunk_words: int = 20


class UtilityWeightsConfig(BaseModel):
    novelty: float = 0.30
    intent_stability: float = 0.25
    completeness: float = 0.20
    evidence_gap: float = 0.20
    retrieval_cost: float = 0.05


class NoveltyConfig(BaseModel):
    high_novelty_threshold: float = 0.30
    low_novelty_threshold: float = 0.05


class IntentStabilityConfig(BaseModel):
    stable_window: int = 2
    max_intents: int = 5


class EvidenceCoverageConfig(BaseModel):
    sufficient_coverage_threshold: float = 0.70
    max_evidence_age_chunks: int = 10


class ControllerConfig(BaseModel):
    retrieve_threshold: float = 0.55
    suppress_threshold: float = 0.20
    min_completeness_to_retrieve: float = 0.40
    min_intent_stability_to_retrieve: float = 0.50
    utility_weights: UtilityWeightsConfig = Field(default_factory=UtilityWeightsConfig)
    novelty: NoveltyConfig = Field(default_factory=NoveltyConfig)
    intent_stability: IntentStabilityConfig = Field(default_factory=IntentStabilityConfig)
    evidence: EvidenceCoverageConfig = Field(default_factory=EvidenceCoverageConfig)
    min_retrieval_interval_s: float = 2.0


class StreamingConfig(BaseModel):
    chunk_interval_s: float = 1.0
    websocket_host: str = "0.0.0.0"
    websocket_port: int = 8001


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:5173"])


class DatabaseConfig(BaseModel):
    url: str = "sqlite:///data/prism.db"


class TelemetryConfig(BaseModel):
    log_dir: str = "results/telemetry"
    log_to_file: bool = True
    log_to_db: bool = True
    log_level: str = "INFO"


class BaselinesConfig(BaseModel):
    fixed_threshold: Dict[str, Any] = Field(
        default_factory=lambda: {"completeness_threshold": 0.60}
    )


class EvaluationConfig(BaseModel):
    test_dataset: str = "evaluation/datasets/frozen_e2e_smoke_20.json"
    full_test_dataset: str = "evaluation/datasets/final_untouched_test.json"
    results_dir: str = "results/benchmark"
    recall_k: List[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    ndcg_k: int = 10


# ──────────────────────────────────────────────────────────────
# Root Config
# ──────────────────────────────────────────────────────────────

class PRISMConfig(BaseModel):
    """Root configuration object — injected into all components."""

    project: Dict[str, str] = Field(
        default_factory=lambda: {
            "name": "prism-streaming-rag",
            "version": "0.1.0",
        }
    )
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    vectorstore: VectorstoreConfig = Field(default_factory=VectorstoreConfig)
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    controller: ControllerConfig = Field(default_factory=ControllerConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    baselines: BaselinesConfig = Field(default_factory=BaselinesConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


# ──────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────

def load_config(config_path: Optional[str] = None) -> PRISMConfig:
    """
    Load PRISMConfig from YAML file.
    Falls back to environment variable PRISM_CONFIG_PATH, then defaults.
    """
    if config_path is None:
        config_path = os.environ.get("PRISM_CONFIG_PATH", "configs/config.yaml")

    path = Path(config_path)
    if not path.exists():
        # Return defaults if no config file found
        return PRISMConfig()

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return PRISMConfig()

    # Apply DATABASE_URL env override
    db_url = os.environ.get("DATABASE_URL")
    if db_url and "database" in raw:
        raw["database"]["url"] = db_url

    return PRISMConfig.model_validate(raw)


# Module-level singleton (lazy — avoids import-time file access)
_config: Optional[PRISMConfig] = None


def get_config() -> PRISMConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config
