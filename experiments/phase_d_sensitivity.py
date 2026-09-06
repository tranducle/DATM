"""Phase D Monte Carlo sensitivity analysis for stylized adversarial drift."""

from __future__ import annotations

import csv

import numpy as np
import torch

from common import RESULTS_DIR, write_json

SEED = 42
DIM = 384
MC_RUNS = 300
N_BENIGN = 500
BENIGN_SCALE = 0.10
N_ADVERSARIAL = 200
ADVERSARIAL_SCALE = 0.15
MEAN_SHIFT_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
RHO_MIN = 0.05
RHO_MAX = 0.50
RHO_STEP = 0.01
NORMALIZE_BY_DIM = True


def distances(points: torch.Tensor, centroid: torch.Tensor) -> torch.Tensor:
    d = torch.sqrt(torch.sum((points - centroid) ** 2, dim=1))
    return d / np.sqrt(DIM) if NORMALIZE_BY_DIM else d


def classification_metrics(
    mean_tpr: float,
    mean_fpr: float,
    n_positive: int,
    n_negative: int,
) -> dict[str, float]:
    """Return prevalence-aware precision, recall, and standard F1."""
    tp = mean_tpr * n_positive
    fp = mean_fpr * n_negative
    fn = (1.0 - mean_tpr) * n_positive

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = mean_tpr
    denominator = 2.0 * tp + fp + fn
    f1 = (2.0 * tp / denominator) if denominator > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def summarize_run_metrics(
    tprs: list[float],
    fprs: list[float],
    n_positive: int,
    n_negative: int,
) -> dict[str, float]:
    """Average standard classification metrics across Monte Carlo runs."""
    if len(tprs) != len(fprs) or not tprs:
        raise ValueError("tprs and fprs must be non-empty lists of equal length")

    run_metrics = [
        classification_metrics(tpr, fpr, n_positive, n_negative)
        for tpr, fpr in zip(tprs, fprs)
    ]
    f1s = [m["f1"] for m in run_metrics]
    precisions = [m["precision"] for m in run_metrics]
    return {
        "tpr": float(np.mean(tprs)),
        "fpr": float(np.mean(fprs)),
        "precision": float(np.mean(precisions)),
        "f1": float(np.mean(f1s)),
        "f1_lo": float(np.percentile(f1s, 2.5)),
        "f1_hi": float(np.percentile(f1s, 97.5)),
    }


def write_reference_csv(rows: list[dict[str, float]]) -> None:
    """Write the canonical Phase D sweep table used by the manuscript."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mean_shift",
        "rho",
        "tpr_mean",
        "tpr_ci_lo",
        "tpr_ci_hi",
        "fpr_mean",
        "fpr_ci_lo",
        "fpr_ci_hi",
        "precision_mean",
        "f1_mean",
        "f1_ci_lo",
        "f1_ci_hi",
    ]
    with (RESULTS_DIR / "phase_d_reference_sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "mean_shift": row["sigma"],
                    "rho": row["rho"],
                    "tpr_mean": row["tpr"],
                    "tpr_ci_lo": row["tpr_lo"],
                    "tpr_ci_hi": row["tpr_hi"],
                    "fpr_mean": row["fpr"],
                    "fpr_ci_lo": row["fpr_lo"],
                    "fpr_ci_hi": row["fpr_hi"],
                    "precision_mean": row["precision"],
                    "f1_mean": row["f1"],
                    "f1_ci_lo": row["f1_lo"],
                    "f1_ci_hi": row["f1_hi"],
                }
            )


def run() -> dict[str, object]:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rhos = np.arange(RHO_MIN, RHO_MAX, RHO_STEP)
    rows = []

    for sigma in MEAN_SHIFT_LEVELS:
        for rho in rhos:
            tprs, fprs = [], []
            for _ in range(MC_RUNS):
                benign = torch.randn(N_BENIGN, DIM, device=device) * BENIGN_SCALE
                adversarial = torch.randn(N_ADVERSARIAL, DIM, device=device) * ADVERSARIAL_SCALE + sigma
                centroid = torch.mean(benign, dim=0)
                benign_dist = distances(benign, centroid)
                adversarial_dist = distances(adversarial, centroid)
                tprs.append(torch.mean((adversarial_dist > rho).float()).item())
                fprs.append(torch.mean((benign_dist > rho).float()).item())

            metrics = summarize_run_metrics(
                tprs=tprs,
                fprs=fprs,
                n_positive=N_ADVERSARIAL,
                n_negative=N_BENIGN,
            )
            rows.append({
                "sigma": sigma,
                "rho": float(rho),
                "tpr": metrics["tpr"],
                "fpr": metrics["fpr"],
                "tpr_lo": float(np.percentile(tprs, 2.5)),
                "tpr_hi": float(np.percentile(tprs, 97.5)),
                "fpr_lo": float(np.percentile(fprs, 2.5)),
                "fpr_hi": float(np.percentile(fprs, 97.5)),
                "precision": metrics["precision"],
                "recall": metrics["tpr"],
                "f1": metrics["f1"],
                "f1_lo": metrics["f1_lo"],
                "f1_hi": metrics["f1_hi"],
            })

    result = {
        "seed": SEED,
        "dimension": DIM,
        "mc_runs_per_cell": MC_RUNS,
        "n_benign": N_BENIGN,
        "n_adversarial": N_ADVERSARIAL,
        "benign_scale": BENIGN_SCALE,
        "adversarial_scale": ADVERSARIAL_SCALE,
        "adversarial_mean_shift_levels": MEAN_SHIFT_LEVELS,
        "rho_min": RHO_MIN,
        "rho_max_exclusive": RHO_MAX,
        "rho_step": RHO_STEP,
        "normalize_by_dimension": NORMALIZE_BY_DIM,
        "rows": rows,
    }
    write_json("phase_d_sensitivity.json", result)
    write_reference_csv(rows)
    return result


if __name__ == "__main__":
    payload = run()
    print(f"Phase D complete: {len(payload['rows'])} sigma/rho cells")
