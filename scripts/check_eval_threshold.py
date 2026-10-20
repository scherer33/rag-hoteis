"""Fail CI when retrieval quality drops below the configured thresholds.

Usage:
    python scripts/check_eval_threshold.py results/latest.json config/eval_thresholds.json
"""

import json
import sys
from pathlib import Path

DEFAULT_RESULTS = Path("results/latest.json")
DEFAULT_THRESHOLDS = Path("config/eval_thresholds.json")


def check_threshold(results: dict, thresholds: dict[str, float]) -> list[str]:
    """Return a human-readable list of failures (empty means the gate passes)."""
    failures: list[str] = []
    for config, scores in results.items():
        for metric, minimum in thresholds.items():
            value = scores.get(metric)
            if value is None:
                failures.append(f"{config}: {metric} missing")
            elif value < minimum:
                failures.append(f"{config}: {metric}={value:.3f} < {minimum:.3f}")
    return failures


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    results_path = Path(args[0]) if args else DEFAULT_RESULTS
    thresholds_path = Path(args[1]) if len(args) > 1 else DEFAULT_THRESHOLDS

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    results = payload.get("results", payload)
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))

    failures = check_threshold(results, thresholds)
    if failures:
        print("eval gate failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("eval gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
