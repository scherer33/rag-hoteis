import math

from hotel_rag.eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 2) == 0.5
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 3) == 1.0
    assert recall_at_k([], {"a"}, 5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "a", "b"], {"a"}) == 0.5
    assert reciprocal_rank(["a"], {"a"}) == 1.0
    assert reciprocal_rank(["x"], {"a"}) == 0.0


def test_ndcg_at_k():
    # DCG = 1/log2(2) + 1/log2(4); IDCG com 2 relevantes = 1/log2(2) + 1/log2(3)
    expected = (1.0 + 1 / math.log2(4)) / (1.0 + 1 / math.log2(3))
    assert ndcg_at_k(["a", "x", "b"], {"a", "b"}, 3) == expected


def test_empty_relevant_set_returns_zero():
    assert recall_at_k(["a"], set(), 3) == 0.0
    assert reciprocal_rank(["a"], set()) == 0.0
    assert ndcg_at_k(["a"], set(), 3) == 0.0


def test_duplicate_doc_ids_do_not_inflate_scores():
    assert ndcg_at_k(["a", "a", "b"], {"a", "b"}, 3) == ndcg_at_k(["a", "b"], {"a", "b"}, 3)
