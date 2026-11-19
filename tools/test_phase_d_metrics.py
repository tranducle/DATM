"""Deterministic checks for the controlled Phase D sensitivity analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import phase_d_sensitivity as exp


def assert_close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{label}: expected {expected} within {tolerance}, got {actual}")


def test_fixed_geometry() -> None:
    assert exp.DIM == 384
    assert exp.N_BENIGN == 500
    assert exp.N_ADVERSARIAL == 200
    assert exp.OFFICIAL_RUNS == 300
    assert_close(exp.BENIGN_SCALE, 0.10, 1e-12, "benign scale")
    assert exp.CONDITIONS == [
        ("equal_variance", 0.10),
        ("unequal_variance_comparator", 0.15),
    ]
    assert exp.MEAN_SHIFTS == [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    assert exp.RHOS.size == 45
    assert_close(float(exp.RHOS[0]), 0.05, 1e-12, "rho start")
    assert_close(float(exp.RHOS[-1]), 0.49, 1e-12, "rho end")


def test_normalized_radius_matches_scale() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(12345)
    z = torch.randn((10000, exp.DIM), generator=generator, dtype=torch.float32)
    zero = torch.zeros(exp.DIM, dtype=torch.float32)
    r010 = float(np.mean(exp.normalized_distances(z * 0.10, zero)))
    r015 = float(np.mean(exp.normalized_distances(z * 0.15, zero)))
    assert_close(r010, 0.10, 0.002, "normalized radius at scale 0.10")
    assert_close(r015, 0.15, 0.003, "normalized radius at scale 0.15")
    assert r015 > r010


def test_common_random_numbers_are_reproducible() -> None:
    first_b, first_a = exp.standard_draws(exp.run_seed(2, 7))
    second_b, second_a = exp.standard_draws(exp.run_seed(2, 7))
    if not torch.equal(first_b, second_b):
        raise AssertionError("benign random draw is not reproducible")
    if not torch.equal(first_a, second_a):
        raise AssertionError("adversarial random draw is not reproducible")


def test_equal_variance_zero_shift_is_same_distribution_family() -> None:
    benign_z, adversarial_z = exp.standard_draws(777)
    benign_scores, adversarial_scores = exp.scores_for_draw(
        benign_z,
        adversarial_z,
        adversarial_scale=0.10,
        mean_shift=0.0,
    )
    if not np.isfinite(benign_scores).all() or not np.isfinite(adversarial_scores).all():
        raise AssertionError("non-finite scores")
    if abs(float(np.mean(benign_scores)) - float(np.mean(adversarial_scores))) > 0.01:
        raise AssertionError("equal-variance zero-shift score means differ unexpectedly")


def test_threshold_metrics_known_case() -> None:
    benign = np.array([0.05, 0.10, 0.20], dtype=float)
    adversarial = np.array([0.15, 0.25], dtype=float)
    metrics = exp.threshold_arrays(benign, adversarial)
    idx = int(np.argmin(np.abs(exp.RHOS - 0.14)))
    assert_close(float(metrics["tpr"][idx]), 1.0, 1e-12, "TPR")
    assert_close(float(metrics["fpr"][idx]), 1.0 / 3.0, 1e-12, "FPR")


if __name__ == "__main__":
    test_fixed_geometry()
    test_normalized_radius_matches_scale()
    test_common_random_numbers_are_reproducible()
    test_equal_variance_zero_shift_is_same_distribution_family()
    test_threshold_metrics_known_case()
    print("phase_d_metric_tests=PASS")
