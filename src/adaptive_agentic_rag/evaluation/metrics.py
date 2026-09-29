import math


def recall_at_k(
    retrieved_keys: list[str],
    relevant_keys: set[str],
    k: int
) -> float:

    if not relevant_keys:
        return 0.0

    retrieved = set(
        retrieved_keys[:k]
    )

    found = retrieved.intersection(
        relevant_keys
    )

    return len(found) / len(relevant_keys)


def hit_at_k(
    retrieved_keys: list[str],
    relevant_keys: set[str],
    k: int
) -> float:

    retrieved = set(
        retrieved_keys[:k]
    )

    return float(
        bool(
            retrieved.intersection(
                relevant_keys
            )
        )
    )


def complete_evidence_recall_at_k(
    retrieved_keys: list[str],
    relevant_keys: set[str],
    k: int
) -> float:

    if not relevant_keys:
        return 0.0

    retrieved = set(
        retrieved_keys[:k]
    )

    return float(
        relevant_keys.issubset(
            retrieved
        )
    )


def reciprocal_rank(
    retrieved_keys: list[str],
    relevant_keys: set[str],
    k: int
) -> float:

    for rank, key in enumerate(
        retrieved_keys[:k],
        start=1
    ):

        if key in relevant_keys:

            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved_keys: list[str],
    relevant_keys: set[str],
    k: int
) -> float:

    if not relevant_keys:
        return 0.0

    seen_relevant = set()

    dcg = 0.0

    for rank, key in enumerate(
        retrieved_keys[:k],
        start=1
    ):

        # یک document مرتبط فقط یک بار gain می‌گیرد.
        # اگر چند chunk از همان document آمده باشد،
        # duplicate را دوباره relevant حساب نمی‌کنیم.
        if (
            key in relevant_keys
            and key not in seen_relevant
        ):

            gain = 1.0
            seen_relevant.add(key)

        else:

            gain = 0.0

        dcg += gain / math.log2(
            rank + 1
        )

    ideal_relevant_count = min(
        len(relevant_keys),
        k
    )

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(
            1,
            ideal_relevant_count + 1
        )
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg
def unique_documents_at_k(
    retrieved_keys: list[str],
    k: int
) -> int:

    return len(
        set(
            retrieved_keys[:k]
        )
    )


def duplicate_rate_at_k(
    retrieved_keys: list[str],
    k: int
) -> float:

    retrieved = retrieved_keys[:k]

    if not retrieved:
        return 0.0

    unique_count = len(
        set(retrieved)
    )

    return 1.0 - (
        unique_count
        / len(retrieved)
    )
def first_complete_evidence_rank(
    retrieved_keys: list[str],
    relevant_keys: set[str]
) -> int | None:

    if not relevant_keys:
        return None

    found = set()

    for rank, key in enumerate(
        retrieved_keys,
        start=1
    ):

        if key in relevant_keys:
            found.add(key)

        if relevant_keys.issubset(found):
            return rank

    return None