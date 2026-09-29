import numpy as np



def cosine_similarity(a,b):

    return np.dot(a,b)



def mmr_select(
    query_embedding,
    document_embeddings,
    documents,
    top_k=5,
    lambda_param=0.7,
    max_per_document=1
):


    selected = []

    candidates = list(
        range(len(documents))
    )


    document_count = {}



    while len(selected) < top_k and candidates:


        best_score = -float("inf")

        best_candidate = None



        for idx in candidates:


            doc_id = documents[idx].get(
                "document_id"
            )


            if doc_id is not None:

                if document_count.get(
                    doc_id,
                    0
                ) >= max_per_document:

                    continue



            relevance = cosine_similarity(
                query_embedding,
                document_embeddings[idx]
            )



            if not selected:

                diversity = 0


            else:

                diversity = max(

                    cosine_similarity(
                        document_embeddings[idx],
                        document_embeddings[selected_idx]
                    )

                    for selected_idx in selected

                )



            mmr_score = (

                lambda_param * relevance

                -

                (1-lambda_param)
                * diversity

            )



            if mmr_score > best_score:

                best_score = mmr_score

                best_candidate = idx



        if best_candidate is None:

            break



        selected.append(
            best_candidate
        )


        doc_id = documents[
            best_candidate
        ].get(
            "document_id"
        )


        if doc_id:

            document_count[doc_id] = (
                document_count.get(
                    doc_id,
                    0
                )
                + 1
            )


        candidates.remove(
            best_candidate
        )



    return [

        documents[i]

        for i in selected

    ]