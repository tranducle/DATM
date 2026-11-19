"""Verify Phase B split counts and held out reporting arithmetic."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from phase_b_response_integrity import SEED_BASE, load_truthfulqa


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def main() -> None:
    questions, correct, incorrect = load_truthfulqa()
    if len(questions) != 817:
        raise AssertionError(f"expected 817 questions, got {len(questions)}")

    rng = np.random.default_rng(SEED_BASE)
    idx = np.arange(len(questions))
    rng.shuffle(idx)
    train_idx = np.sort(idx[:408])
    test_idx = np.sort(idx[408:])

    train_incorrect = sum(min(10, len(incorrect[i])) for i in train_idx)
    test_incorrect = sum(min(10, len(incorrect[i])) for i in test_idx)
    if train_incorrect != 1628:
        raise AssertionError(f"expected 1628 training incorrect candidates, got {train_incorrect}")
    if test_incorrect != 1665:
        raise AssertionError(f"expected 1665 test incorrect candidates, got {test_incorrect}")

    tp, fn, fp, tn = 342, 67, 196, 213
    if tp + fn != 409 or fp + tn != 409:
        raise AssertionError("held out pooled class counts do not sum to 409 per class")

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    fpr = fp / (fp + tn)
    assert_close(precision, 0.6356877323420075, "precision")
    assert_close(recall, 0.8361858190709046, "recall")
    assert_close(f1, 0.7222808870116156, "f1")

    auc = 0.7140619675874726
    lower_bound = recall * (1.0 - fpr)
    if auc < lower_bound:
        raise AssertionError(f"AUC {auc} is below the operating point lower bound {lower_bound}")

    print("phase_b_reporting_tests=PASS")


if __name__ == "__main__":
    main()
