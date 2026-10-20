import json
from pathlib import Path

from hotel_rag.eval.golden import load_golden


def test_load_golden_parses_jsonl(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "q",
                "answer": "a",
                "relevant_doc_ids": ["checkin"],
                "language": "pt",
                "dept": "rececao",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    items = load_golden(path)

    assert items[0].relevant_doc_ids == ["checkin"]


def test_golden_dataset_is_complete_and_multilingual():
    items = load_golden(Path(__file__).parents[2] / "data" / "golden" / "golden_set.jsonl")

    assert len(items) >= 30
    assert sum(item.language == "en" for item in items) >= 5
    assert all(item.relevant_doc_ids for item in items)


def test_golden_dataset_covers_every_department():
    items = load_golden(Path(__file__).parents[2] / "data" / "golden" / "golden_set.jsonl")
    counts: dict[str, int] = {}
    for item in items:
        counts[item.dept.value] = counts.get(item.dept.value, 0) + 1

    assert len(counts) == 5
    assert min(counts.values()) >= 4


def test_golden_dataset_references_real_documents():
    from hotel_rag.corpus import load_corpus

    root = Path(__file__).parents[2]
    known = {document.metadata.doc_id for document in load_corpus(root / "data" / "corpus")}
    items = load_golden(root / "data" / "golden" / "golden_set.jsonl")

    missing = {
        doc_id
        for item in items
        for doc_id in item.relevant_doc_ids
        if doc_id not in known
    }
    assert missing == set()
