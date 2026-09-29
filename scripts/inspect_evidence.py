from datasets import load_dataset


dataset = load_dataset(
    "yixuantt/MultiHopRAG",
    "MultiHopRAG"
)["train"]


for item in dataset:

    print(item["query"])

    print(item["evidence_list"])

    break