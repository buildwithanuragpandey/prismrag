from datasets import Dataset

from adaptive_agentic_rag.schemas import Document


def convert_corpus_to_documents(
    corpus_dataset: Dataset
) -> list[Document]:

    documents = []

    for idx, item in enumerate(corpus_dataset):

        document = Document(
            id=f"doc_{idx:04d}",

            text=item["body"],

            metadata={
                "title": item["title"],
                "author": item["author"],
                "category": item["category"],
                "published_at": item["published_at"],
                "source": item["source"],
                "url": item["url"],
            }
        )

        documents.append(document)

    return documents