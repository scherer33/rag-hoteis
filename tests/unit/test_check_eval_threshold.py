import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

from check_eval_threshold import check_threshold  # noqa: E402


def test_passes_when_above():
    assert check_threshold({"hybrid_rerank": {"recall@k": 0.8}}, {"recall@k": 0.7}) == []


def test_fails_when_below():
    failures = check_threshold({"hybrid_rerank": {"recall@k": 0.5}}, {"recall@k": 0.7})

    assert len(failures) == 1
    assert "recall@k" in failures[0]


def test_fails_when_metric_is_missing():
    failures = check_threshold({"vector": {"mrr": 0.9}}, {"recall@k": 0.7})

    assert len(failures) == 1
    assert "missing" in failures[0]
