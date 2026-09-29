import json
from pathlib import Path

from datasets import load_dataset

from adaptive_agentic_rag.retrieval.reranked_retriever import (
    RerankedRetriever
)

from adaptive_agentic_rag.evaluation.retrieval_eval import (
    evaluate_dense_retriever
)


LAMBDAS = [
    1.0,
    0.95,
    0.9,
    0.85,
    0.8,
    0.75,
    0.7
]


def main():

    print("Loading QA dataset...")

    dataset = load_dataset(
        "yixuantt/MultiHopRAG",
        "MultiHopRAG"
    )["train"]


    results = []


    for lam in LAMBDAS:

        print("\n======================")
        print(
            f"Testing lambda={lam}"
        )
        print("======================")


        retriever = RerankedRetriever(
            mmr_lambda=lam
        )


        try:

            result = evaluate_dense_retriever(
                dataset=dataset,
                retriever=retriever,
                limit=200
            )


        finally:

            retriever.close()



        results.append(
            {
                "lambda": lam,

                "recall@5":
                    result["metrics"]["recall@5"],

                "recall@10":
                    result["metrics"]["recall@10"],

                "recall@20":
                    result["metrics"]["recall@20"],

                "ndcg@10":
                    result["metrics"]["ndcg@10"],

                "duplicate_rate@10":
                    result["metrics"]["duplicate_rate@10"]
            }
        )


    print("\n\n===== FINAL RESULTS =====")


    for r in results:

        print(r)



    output = Path(
        "evaluation/results/mmr_lambda_sweep.json"
    )


    with open(
        output,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )


    print(
        f"\nSaved: {output}"
    )


if __name__ == "__main__":
    main()