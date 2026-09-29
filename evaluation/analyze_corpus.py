import json
from statistics import mean


CORPUS_PATH = (
    "data/processed/processed_corpus.json"
)


def load_chunks():

    with open(
        CORPUS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)



def main():

    chunks = load_chunks()


    print(
        f"Total chunks: {len(chunks)}"
    )


    lengths = [
        len(chunk["text"].split())
        for chunk in chunks
    ]


    print(
        f"Average words per chunk: {mean(lengths):.2f}"
    )


    print(
        f"Min chunk size: {min(lengths)}"
    )


    print(
        f"Max chunk size: {max(lengths)}"
    )


    print("\n===== SAMPLE CHUNKS =====")


    for chunk in chunks[:3]:

        print("\n----------------")
        
        print(
            "ID:",
            chunk["id"]
        )

        print(
            "Source:",
            chunk["metadata"]["source"]
        )

        print(
            "Text preview:"
        )

        print(
            chunk["text"][:500]
        )



if __name__ == "__main__":
    main()