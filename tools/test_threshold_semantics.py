"""Regression checks for ROC threshold semantics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from common import find_best_threshold, metrics_at_threshold


def main() -> None:
    labels = np.asarray([0, 1], dtype=int)
    scores = np.asarray([0.0, 1.0], dtype=float)
    threshold, auc = find_best_threshold(labels, scores)
    if threshold != 1.0:
        raise AssertionError(f"expected threshold 1.0, got {threshold}")
    if auc != 1.0:
        raise AssertionError(f"expected AUC 1.0, got {auc}")

    fpr, tpr, thresholds = roc_curve(labels, scores)
    idx = int(np.where(thresholds == threshold)[0][0])
    expected_tpr = float(tpr[idx])
    expected_fpr = float(fpr[idx])

    metrics = metrics_at_threshold(labels, scores, threshold)
    if metrics["recall"] != expected_tpr:
        raise AssertionError(
            f"threshold recall {metrics['recall']} does not match roc_curve TPR {expected_tpr}"
        )
    expected_precision = 1.0 if expected_fpr == 0.0 and expected_tpr > 0.0 else None
    if expected_precision is not None and metrics["precision"] != expected_precision:
        raise AssertionError(
            f"expected precision {expected_precision}, got {metrics['precision']}"
        )

    tied_scores = np.asarray([0.0, 1.0, 1.0, 2.0], dtype=float)
    tied_labels = np.asarray([0, 1, 1, 1], dtype=int)
    tied_metrics = metrics_at_threshold(tied_labels, tied_scores, 1.0)
    if tied_metrics["recall"] != 1.0:
        raise AssertionError("scores equal to the selected threshold must be classified positive")

    print("threshold_semantics_tests=PASS")


if __name__ == "__main__":
    main()
