from hotel_rag.eval.golden import GoldenItem
from hotel_rag.eval.run import evaluate, to_markdown
from hotel_rag.models import Dept


class FixedRetriever:
    def __init__(self, doc_ids: list[str]) -> None:
        self.doc_ids = doc_ids

    def retrieve(self, query, top_k=5, dept=None, rewrite=None):
        from tests.fakes import make_retrieved

        return [make_retrieved(doc_id, 1.0) for doc_id in self.doc_ids[:top_k]]


ITEMS = [
    GoldenItem(
        question="q",
        answer="a",
        relevant_doc_ids=["a"],
        language="pt",
        dept=Dept.COPA,
    )
]


def test_evaluate_perfect_retrieval_scores_one():
    scores = evaluate(FixedRetriever(["a"]), ITEMS, k=5)

    assert scores["recall@k"] == 1.0
    assert scores["mrr"] == 1.0
    assert scores["ndcg@k"] == 1.0


def test_evaluate_miss_scores_zero():
    scores = evaluate(FixedRetriever(["z"]), ITEMS, k=5)

    assert scores["recall@k"] == 0.0
    assert scores["mrr"] == 0.0


def test_evaluate_averages_over_items():
    items = [
        ITEMS[0],
        GoldenItem(
            question="q2",
            answer="b",
            relevant_doc_ids=["b"],
            language="pt",
            dept=Dept.COPA,
        ),
    ]
    scores = evaluate(FixedRetriever(["a"]), items, k=5)

    assert scores["recall@k"] == 0.5


def test_to_markdown_contains_columns_and_rows():
    markdown = to_markdown({"vector": {"recall@k": 0.5, "mrr": 0.4, "ndcg@k": 0.45}})

    assert "| config | recall@k | mrr | ndcg@k |" in markdown
    assert "vector" in markdown
