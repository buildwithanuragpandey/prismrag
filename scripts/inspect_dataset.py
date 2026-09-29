from datasets import load_dataset


print("Loading QA dataset...")

qa_dataset = load_dataset(
    "yixuantt/MultiHopRAG",
    "MultiHopRAG"
)


print("Loading corpus dataset...")

corpus_dataset = load_dataset(
    "yixuantt/MultiHopRAG",
    "corpus"
)


print("\n===== QA DATASET =====")
print(qa_dataset)


print("\n===== CORPUS DATASET =====")
print(corpus_dataset)
print("\n===== FIRST QA EXAMPLE =====")

print(
    qa_dataset["train"][0]
)


print("\n===== FIRST CORPUS DOCUMENT =====")

print(
    corpus_dataset["train"][0]
)