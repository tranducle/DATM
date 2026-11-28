"""Verify Phase B split counts, held-out reporting arithmetic, and nine-variant ablation manifest parity."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from phase_b_response_integrity import ABLATION_SCHEMES, SEED_BASE, aggregate_scores, load_truthfulqa


def assert_close(actual: float, expected: float, label: str, tol: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def check_ablation_helpers_and_manifest() -> None:
    sample = np.asarray([0.1, 0.4, 0.9, 0.2], dtype=float)
    assert_close(aggregate_scores(sample, "max-pool"), 0.9, "max-pool")
    assert_close(aggregate_scores(sample, "mean-pool"), 0.4, "mean-pool")
    assert_close(aggregate_scores(sample, "top2-mean"), 0.65, "top2-mean")

    expected_order = [
        ("full_multi_signal_maxpool", "max-pool", 10),
        ("contradiction_only_maxpool", "max-pool", 10),
        ("drop_length_ratio", "max-pool", 10),
        ("drop_cosine_dist", "max-pool", 10),
        ("drop_entail_gap", "max-pool", 10),
        ("mean_pool_aggregation", "mean-pool", 10),
        ("top2_mean_aggregation", "top2-mean", 10),
        ("incorrect_cap_3", "max-pool", 3),
        ("incorrect_cap_1", "max-pool", 1),
    ]
    actual_order = [(cfg["name"], cfg["aggregation"], cfg["incorrect_cap"]) for cfg in ABLATION_SCHEMES]
    if actual_order != expected_order:
        raise AssertionError(f"ABLATION_SCHEMES mismatch: {actual_order}")

    manifest = json.loads((ROOT / "results" / "paper_results_manifest.json").read_text(encoding="utf-8"))
    audit = manifest["phase_b"]["bounded_repeated_audit"]
    if audit["subset_questions"] != 240 or audit["n_repetitions"] != 100:
        raise AssertionError("bounded_repeated_audit scope mismatch")
    if audit["default_max_incorrect_answers_per_question"] != 10:
        raise AssertionError("bounded_repeated_audit default incorrect cap must be 10")

    schemes = audit["schemes"]
    if list(schemes.keys()) != [name for name, _, _ in expected_order]:
        raise AssertionError(f"manifest scheme keys mismatch: {list(schemes.keys())}")

    assert_close(
        audit["full_multi_signal"]["auc"]["mean"],
        schemes["full_multi_signal_maxpool"]["summary"]["auc"]["mean"],
        "full_multi_signal alias auc",
    )
    assert_close(
        audit["contradiction_only"]["f1"]["mean"],
        schemes["contradiction_only_maxpool"]["summary"]["f1"]["mean"],
        "contradiction_only alias f1",
    )

    expected_rounded = {
        "full_multi_signal_maxpool": ("0.699", "0.695"),
        "contradiction_only_maxpool": ("0.736", "0.755"),
        "drop_length_ratio": ("0.715", "0.703"),
        "drop_cosine_dist": ("0.689", "0.721"),
        "drop_entail_gap": ("0.694", "0.694"),
        "mean_pool_aggregation": ("0.601", "0.697"),
        "top2_mean_aggregation": ("0.659", "0.690"),
        "incorrect_cap_3": ("0.679", "0.703"),
        "incorrect_cap_1": ("0.532", "0.645"),
    }
    full_f1 = schemes["full_multi_signal_maxpool"]["summary"]["f1"]["mean"]
    at_or_above = 0
    for name, (exp_auc, exp_f1) in expected_rounded.items():
        summary = schemes[name]["summary"]
        got_auc = f"{summary['auc']['mean']:.3f}"
        got_f1 = f"{summary['f1']['mean']:.3f}"
        if got_auc != exp_auc or got_f1 != exp_f1:
            raise AssertionError(f"{name}: expected ({exp_auc}, {exp_f1}), got ({got_auc}, {got_f1})")
        if name != "full_multi_signal_maxpool" and summary["f1"]["mean"] >= full_f1:
            at_or_above += 1
    if at_or_above != 5:
        raise AssertionError(f"expected 5 alternatives with mean F1 >= full fusion, got {at_or_above}")


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

    check_ablation_helpers_and_manifest()
    print("phase_b_reporting_tests=PASS")


if __name__ == "__main__":
    main()
