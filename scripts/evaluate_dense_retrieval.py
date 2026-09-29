import argparse
import json
from pathlib import Path

from datasets import load_dataset

from adaptive_agentic_rag.retrieval.dense_retriever import (
    DenseRetriever
)

from adaptive_agentic_rag.evaluation.retrieval_eval import (
    evaluate_dense_retriever
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None
    )

    args = parser.parse_args()

    print("Loading QA dataset...")

    dataset = load_dataset(
        "yixuantt/MultiHopRAG",
        "MultiHopRAG"
    )["train"]

    retriever = DenseRetriever()

    try:

        result = evaluate_dense_retriever(
            dataset=dataset,
            retriever=retriever,
            limit=args.limit
        )

    finally:

        retriever.close()

    print("\n===== DENSE RETRIEVAL =====")

    print(
        f"Evaluated: "
        f"{result['num_evaluated']}"
    )

    print(
        f"Skipped no evidence: "
        f"{result['skipped_no_evidence']}"
    )

    print(
        f"Query embedding time: "
        f"{result['embedding_seconds']:.2f}s"
    )

    print(
        f"Average Qdrant retrieval: "
        f"{result['avg_retrieval_ms']:.2f} ms"
    )

    print("\n===== METRICS =====")

    for name, value in (
        result["metrics"].items()
    ):

        print(
            f"{name:<22}: "
            f"{value:.4f}"
        )

    output_dir = Path(
        "evaluation/results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if args.limit is None:

        filename = (
            "dense_baseline_full.json"
        )

    else:

        filename = (
            f"dense_smoke_{args.limit}.json"
        )


    output_path = (
        output_dir
        / filename
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nSaved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()