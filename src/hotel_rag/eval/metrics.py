"""Classic information-retrieval metrics.

All functions deduplicate retrieved document ids, keeping the first occurrence,
and return 0.0 when there are no relevant documents (an undefined score).
"""

import math


def _dedupe(doc_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for doc_id in doc_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            unique.append(doc_id)
    return unique


def recall_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    if not relevant_doc_ids:
        return 0.0
    retrieved = _dedupe(retrieved_doc_ids)[:k]
    return len(set(retrieved) & relevant_doc_ids) / len(relevant_doc_ids)


def reciprocal_rank(retrieved_doc_ids: list[str], relevant_doc_ids: set[str]) -> float:
    if not relevant_doc_ids:
        return 0.0
    for rank, doc_id in enumerate(_dedupe(retrieved_doc_ids), start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    if not relevant_doc_ids:
        return 0.0

    retrieved = _dedupe(retrieved_doc_ids)[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved, start=1)
        if doc_id in relevant_doc_ids
    )
    ideal_hits = min(len(relevant_doc_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0
