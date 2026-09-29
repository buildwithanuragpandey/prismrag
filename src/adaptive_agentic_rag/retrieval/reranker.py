from typing import List
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)



class BGEReranker:


    def __init__(
        self,
        model_name="BAAI/bge-reranker-base",
        device=None
    ):


        print(
            f"Loading reranker: {model_name}"
        )


        self.device = (
            device
            if device
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )


        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )


        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_name
            )
        )


        self.model.to(
            self.device
        )


        self.model.eval()



    def rerank(
        self,
        query: str,
        documents: List[dict],
        top_k=5,
        batch_size=8
    ):


        if len(documents) == 0:

            return []



        pairs = []


        for doc in documents:


            pairs.append(
                (
                    query,
                    doc["text"]
                )
            )



        scores = []



        with torch.no_grad():


            for i in range(
                0,
                len(pairs),
                batch_size
            ):


                batch = pairs[
                    i:i+batch_size
                ]


                queries = [
                    x[0]
                    for x in batch
                ]


                texts = [
                    x[1]
                    for x in batch
                ]


                encoded = (
                    self.tokenizer(
                        queries,
                        texts,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=512
                    )
                )


                encoded = {
                    k:v.to(self.device)
                    for k,v in encoded.items()
                }


                outputs = (
                    self.model(
                        **encoded
                    )
                )


                logits = (
                    outputs.logits
                    .view(-1)
                    .float()
                    .cpu()
                    .tolist()
                )


                scores.extend(
                    logits
                )



        ranked = []



        for doc, score in zip(
            documents,
            scores
        ):


            item = doc.copy()

            item["rerank_score"] = (
                float(score)
            )

            ranked.append(
                item
            )



        ranked.sort(
            key=lambda x:
                x["rerank_score"],
            reverse=True
        )


        return ranked[:top_k]