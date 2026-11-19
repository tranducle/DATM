"""Deterministic Phase C regex regression checks."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from common import find_best_threshold, metrics_at_threshold, split_indices
from phase_c_exfiltration import regex_score


def load(name: str) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    with (ROOT / "data" / name).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            groups.setdefault(row["label"], []).append(row["text"])
    return groups


def run(name: str, n_train: int, seed: int) -> dict[str, float]:
    groups = load(name)
    benign = groups["benign"]
    attack = groups["exfiltration"]
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(200):
        b_train, b_test = split_indices(len(benign), n_train, rng)
        a_train, a_test = split_indices(len(attack), n_train, rng)
        train_labels = np.asarray([0] * len(b_train) + [1] * len(a_train))
        test_labels = np.asarray([0] * len(b_test) + [1] * len(a_test))
        train_scores = np.asarray(
            [regex_score(benign[i]) for i in b_train] + [regex_score(attack[i]) for i in a_train],
            dtype=float,
        )
        test_scores = np.asarray(
            [regex_score(benign[i]) for i in b_test] + [regex_score(attack[i]) for i in a_test],
            dtype=float,
        )
        threshold, _ = find_best_threshold(train_labels, train_scores)
        rows.append({"threshold": threshold, **metrics_at_threshold(test_labels, test_scores, threshold)})
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in ["auc", "precision", "recall", "f1", "threshold"]
    }


def assert_close(actual: float, expected: float, label: str) -> None:
    if math.isinf(expected):
        if not math.isinf(actual):
            raise AssertionError(f"{label}: expected infinity, got {actual}")
        return
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def main() -> None:
    original = run("phase_c_original_templates.csv", 10, 8042)
    paraphrastic = run("phase_c_paraphrastic_templates.csv", 8, 3042)
    marker = run("phase_c_marker_scrubbed_templates.csv", 8, 9042)

    expected = {
        "original": {"auc": 0.97, "precision": 1.0, "recall": 0.94, "f1": 0.9666666666666669, "threshold": 1.0},
        "paraphrastic": {"auc": 0.73484375, "precision": 0.8733333333333334, "recall": 0.5525, "f1": 0.6551746031746032, "threshold": 1.0},
        "marker": {"auc": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0, "threshold": float("inf")},
    }
    actual = {"original": original, "paraphrastic": paraphrastic, "marker": marker}
    for condition, metrics in expected.items():
        for key, value in metrics.items():
            assert_close(actual[condition][key], value, f"{condition}.{key}")

    print("phase_c_regex_tests=PASS")


if __name__ == "__main__":
    main()
