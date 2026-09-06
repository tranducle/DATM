"""Regression checks for Phase D classification metrics."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from phase_d_sensitivity import (
    ADVERSARIAL_SCALE,
    BENIGN_SCALE,
    RHO_MAX,
    RHO_MIN,
    RHO_STEP,
    classification_metrics,
    summarize_run_metrics,
)


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def test_perfect_classifier() -> None:
    metrics = classification_metrics(
        mean_tpr=1.0,
        mean_fpr=0.0,
        n_positive=200,
        n_negative=500,
    )
    assert_close(metrics["precision"], 1.0, "perfect precision")
    assert_close(metrics["recall"], 1.0, "perfect recall")
    assert_close(metrics["f1"], 1.0, "perfect f1")


def test_prevalence_aware_f1() -> None:
    metrics = classification_metrics(
        mean_tpr=0.5,
        mean_fpr=0.1,
        n_positive=200,
        n_negative=500,
    )
    assert_close(metrics["precision"], 2.0 / 3.0, "precision")
    assert_close(metrics["recall"], 0.5, "recall")
    assert_close(metrics["f1"], 4.0 / 7.0, "f1")


def test_zero_positive_predictions() -> None:
    metrics = classification_metrics(
        mean_tpr=0.0,
        mean_fpr=0.0,
        n_positive=200,
        n_negative=500,
    )
    assert_close(metrics["precision"], 0.0, "zero precision")
    assert_close(metrics["recall"], 0.0, "zero recall")
    assert_close(metrics["f1"], 0.0, "zero f1")


def test_mean_f1_is_average_of_run_level_f1() -> None:
    summary = summarize_run_metrics(
        tprs=[1.0, 0.0],
        fprs=[0.0, 0.0],
        n_positive=200,
        n_negative=500,
    )
    assert_close(summary["tpr"], 0.5, "mean tpr")
    assert_close(summary["fpr"], 0.0, "mean fpr")
    assert_close(summary["f1"], 0.5, "mean run-level f1")


def test_original_phase_d_geometry_and_refined_grid() -> None:
    assert_close(BENIGN_SCALE, 0.10, "benign scale")
    assert_close(ADVERSARIAL_SCALE, 0.15, "adversarial scale")
    assert_close(RHO_MIN, 0.05, "rho min")
    assert_close(RHO_MAX, 0.50, "rho max")
    assert_close(RHO_STEP, 0.01, "rho step")


if __name__ == "__main__":
    test_perfect_classifier()
    test_prevalence_aware_f1()
    test_zero_positive_predictions()
    test_mean_f1_is_average_of_run_level_f1()
    test_original_phase_d_geometry_and_refined_grid()
    print("phase_d_metric_tests=PASS")
