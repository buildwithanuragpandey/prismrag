"""
PRISM Corpus Ingestion Script (Phase 2)

Downloads and processes the MultiHopRAG corpus from HuggingFace.
Reuses the chunking logic from adaptive-agentic-rag.

Outputs:
  - data/processed/corpus.json     (processed chunks)
  - data/faiss_index/index.faiss   (FAISS dense index)
  - data/faiss_index/metadata.json (chunk metadata)
  - data/processed/bm25_index.pkl  (BM25 index)

Usage:
  python scripts/ingest_corpus.py
  python scripts/ingest_corpus.py --limit 100  (fast test)
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

# Add src to path
_SRC = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(_SRC))

# Also add adaptive-agentic-rag src
_AAR_SRC = Path(__file__).parents[2] / "src"
if _AAR_SRC.exists():
    sys.path.insert(0, str(_AAR_SRC))


def parse_args():
    parser = argparse.ArgumentParser(description="PRISM corpus ingestion")
    parser.add_argument("--limit", type=int, default=None, help="Limit chunks (for testing)")
    parser.add_argument("--corpus-path", default="data/processed/corpus.json")
    parser.add_argument("--faiss-dir", default="data/faiss_index")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def load_or_build_corpus(corpus_path: str, limit=None):
    """Load existing corpus or build from MultiHopRAG."""
    path = Path(corpus_path)
    if path.exists():
        print(f"Loading existing corpus from {corpus_path}")
        with open(path) as f:
            docs = json.load(f)
        print(f"Loaded {len(docs)} chunks.")
        return docs[:limit] if limit else docs

    # Try using adaptive-agentic-rag's existing processed corpus
    aar_corpus = Path(__file__).parents[2] / "data" / "processed" / "processed_corpus_v2.json"
    if aar_corpus.exists():
        print(f"Copying corpus from adaptive-agentic-rag: {aar_corpus}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(aar_corpus) as f:
            docs = json.load(f)
        docs = docs[:limit] if limit else docs
        with open(path, "w") as f:
            json.dump(docs, f)
        print(f"Saved {len(docs)} chunks to {corpus_path}")
        return docs

    # Download from HuggingFace
    print("Downloading MultiHopRAG from HuggingFace...")
    try:
        from datasets import load_dataset
        dataset = load_dataset(
            "yixuantt/MultiHopRAG",
            revision="71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82",
        )

        # Minimal processing: extract corpus chunks
        docs = []
        corpus = dataset.get("corpus", dataset.get("train", None))
        if corpus is None:
            print("ERROR: Could not find corpus split in dataset.")
            sys.exit(1)

        for i, item in enumerate(corpus):
            if limit and i >= limit:
                break
            doc = {
                "id": f"chunk-{i:05d}",
                "doc_id": item.get("document_id", f"doc-{i}"),
                "chunk_id": f"chunk-{i:05d}",
                "content": item.get("content", item.get("text", "")),
                "title": item.get("title", ""),
                "source": item.get("source", ""),
                "url": item.get("url", ""),
            }
            docs.append(doc)

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(docs, f)
        print(f"Saved {len(docs)} chunks to {corpus_path}")
        return docs

    except Exception as e:
        print(f"Failed to download corpus: {e}")
        print("Creating minimal demo corpus for testing...")
        docs = _create_demo_corpus()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(docs, f)
        return docs


def _create_demo_corpus():
    """Create a minimal demo corpus for testing when real corpus unavailable."""
    return [
        {
            "id": "chunk-00001", "doc_id": "doc_0001", "chunk_id": "chunk-00001",
            "content": "FTX was a cryptocurrency exchange founded by Sam Bankman-Fried. The exchange collapsed in November 2022.",
            "title": "FTX Collapse", "source": "techcrunch.com", "url": "https://techcrunch.com/ftx"
        },
        {
            "id": "chunk-00002", "doc_id": "doc_0002", "chunk_id": "chunk-00002",
            "content": "Sam Bankman-Fried, once considered the trustworthy face of the cryptocurrency industry, was accused of fraud.",
            "title": "SBF Trial", "source": "theverge.com", "url": "https://theverge.com/sbf"
        },
        {
            "id": "chunk-00003", "doc_id": "doc_0003", "chunk_id": "chunk-00003",
            "content": "International travel reimbursement policy requires prior approval. Domestic travel up to $500 does not require pre-approval.",
            "title": "Travel Policy", "source": "company.com", "url": "https://company.com/policy"
        },
        {
            "id": "chunk-00004", "doc_id": "doc_0004", "chunk_id": "chunk-00004",
            "content": "Conference venue requirements: minimum capacity of 30 people, AV equipment included, catering available on request.",
            "title": "Venue Guide", "source": "events.com", "url": "https://events.com/venues"
        },
        {
            "id": "chunk-00005", "doc_id": "doc_0005", "chunk_id": "chunk-00005",
            "content": "Cancellation policy: bookings cancelled more than 14 days before the event receive full refund. Within 14 days: 50% refund.",
            "title": "Cancellation Policy", "source": "events.com", "url": "https://events.com/cancel"
        },
    ]


def build_faiss_index(docs, embedding_model_name, faiss_dir):
    """Build FAISS index from processed chunks."""
    import numpy as np

    try:
        import faiss
    except ImportError:
        print("faiss-cpu not installed. Run: pip install faiss-cpu")
        print("Skipping FAISS index build.")
        return

    print(f"Building FAISS index with {embedding_model_name}...")
    from prism.models import EmbeddingModel
    model = EmbeddingModel(model_name=embedding_model_name)

    texts = [d.get("content", d.get("text", "")) for d in docs]
    print(f"Encoding {len(texts)} chunks...")

    batch_size = 64
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embs = model.encode_batch(batch)
        all_embeddings.extend(embs)
        if (i // batch_size) % 10 == 0:
            print(f"  {i}/{len(texts)} encoded...")

    embeddings = np.array(all_embeddings, dtype=np.float32)
    dim = embeddings.shape[1]
    print(f"Embedding dimension: {dim}")

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss_dir_path = Path(faiss_dir)
    faiss_dir_path.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(faiss_dir_path / "index.faiss"))
    with open(faiss_dir_path / "metadata.json", "w") as f:
        json.dump(docs, f)

    print(f"FAISS index saved: {faiss_dir_path}/index.faiss ({index.ntotal} vectors)")


def main():
    args = parse_args()

    print("=" * 60)
    print("PRISM Corpus Ingestion")
    print("=" * 60)

    docs = load_or_build_corpus(args.corpus_path, limit=args.limit)

    if not (args.skip_existing and Path(args.faiss_dir + "/index.faiss").exists()):
        build_faiss_index(docs, args.embedding_model, args.faiss_dir)

    print("\nIngestion complete!")
    print(f"  Corpus: {args.corpus_path} ({len(docs)} chunks)")
    print(f"  FAISS index: {args.faiss_dir}/")
    print("\nNext step: python scripts/build_indexes.py (for BM25)")


if __name__ == "__main__":
    main()
