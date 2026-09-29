from datasets import load_dataset

from adaptive_agentic_rag.evaluation.dataset_adapter import (
    make_document_key
)


def main():

    print("Loading corpus...")

    corpus = load_dataset(
        "yixuantt/MultiHopRAG",
        "corpus"
    )["train"]

    print("Loading QA dataset...")

    qa_dataset = load_dataset(
        "yixuantt/MultiHopRAG",
        "MultiHopRAG"
    )["train"]

    corpus_keys = {
        make_document_key(
            source=item["source"],
            title=item["title"],
            url=item["url"]
        )
        for item in corpus
    }

    total_evidence = 0
    mapped_evidence = 0

    total_answerable_queries = 0
    fully_mapped_queries = 0

    unmapped = []

    for example in qa_dataset:

        evidence_list = example[
            "evidence_list"
        ]

        if not evidence_list:
            continue

        total_answerable_queries += 1

        query_fully_mapped = True

        for evidence in evidence_list:

            total_evidence += 1

            key = make_document_key(
                source=evidence["source"],
                title=evidence["title"],
                url=evidence.get("url")
            )

            if key in corpus_keys:

                mapped_evidence += 1

            else:

                query_fully_mapped = False

                unmapped.append(
                    {
                        "query":
                            example["query"],

                        "source":
                            evidence["source"],

                        "title":
                            evidence["title"],

                        "url":
                            evidence.get("url")
                    }
                )

        if query_fully_mapped:
            fully_mapped_queries += 1

    evidence_coverage = (
        mapped_evidence
        / total_evidence
        if total_evidence
        else 0
    )

    query_coverage = (
        fully_mapped_queries
        / total_answerable_queries
        if total_answerable_queries
        else 0
    )

    print()
    print("===== EVIDENCE MAPPING =====")

    print(
        f"Corpus documents: "
        f"{len(corpus)}"
    )

    print(
        f"Answerable queries: "
        f"{total_answerable_queries}"
    )

    print(
        f"Total evidence items: "
        f"{total_evidence}"
    )

    print(
        f"Mapped evidence items: "
        f"{mapped_evidence}"
    )

    print(
        f"Evidence coverage: "
        f"{evidence_coverage:.4%}"
    )

    print(
        f"Fully mapped queries: "
        f"{fully_mapped_queries}"
    )

    print(
        f"Query coverage: "
        f"{query_coverage:.4%}"
    )

    print(
        f"Unmapped evidence: "
        f"{len(unmapped)}"
    )

    if unmapped:

        print(
            "\nFirst unmapped examples:"
        )

        for item in unmapped[:10]:

            print()
            print(
                item["source"]
            )

            print(
                item["title"]
            )

            print(
                item["url"]
            )


if __name__ == "__main__":
    main()