import json
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi


CORPUS_PATH = Path(
    "data/processed/processed_corpus.json"
)

OUTPUT_PATH = Path(
    "data/bm25/bm25_index.pkl"
)


def tokenize(text):
    """
    Simple tokenizer
    """
    return text.lower().split()



def main():

    print("Loading corpus...")

    with open(
        CORPUS_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        corpus = json.load(f)


    print(
        f"Chunks loaded: {len(corpus)}"
    )


    texts = []
    ids = []


    for item in corpus:

        ids.append(
            item["id"]
        )

        texts.append(
            item["text"]
        )


    print("Tokenizing...")

    tokenized_docs = [
        tokenize(text)
        for text in texts
    ]


    print("Building BM25...")

    bm25 = BM25Okapi(
        tokenized_docs
    )


    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        OUTPUT_PATH,
        "wb"
    ) as f:

        pickle.dump(
            {
                "bm25": bm25,
                "ids": ids,
                "texts": texts
            },
            f
        )


    print(
        "BM25 index saved:"
    )

    print(
        OUTPUT_PATH
    )



if __name__ == "__main__":
    main()