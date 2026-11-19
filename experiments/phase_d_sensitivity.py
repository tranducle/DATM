"""Controlled Monte Carlo sensitivity analysis for Phase D.

The analysis separates adversarial mean displacement from a scale difference
by pairing an equal-variance condition with an unequal-variance comparator.
Both conditions reuse the same standard-normal draws within each run.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.metrics import roc_auc_score

BASE_SEED = 42
DIM = 384
N_BENIGN = 500
N_ADVERSARIAL = 200
BENIGN_SCALE = 0.10
PILOT_RUNS = 20
OFFICIAL_RUNS = 300
MEAN_SHIFTS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
CONDITIONS = [
    ("equal_variance", 0.10),
    ("unequal_variance_comparator", 0.15),
]
RHOS = np.arange(0.05, 0.50, 0.01, dtype=np.float64)
SQRT_DIM = math.sqrt(DIM)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "generated"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_seed(shift_index: int, run_index: int) -> int:
    return BASE_SEED + shift_index * 100000 + run_index


def standard_draws(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    benign_z = torch.randn((N_BENIGN, DIM), generator=generator, dtype=torch.float32)
    adversarial_z = torch.randn((N_ADVERSARIAL, DIM), generator=generator, dtype=torch.float32)
    return benign_z, adversarial_z


def normalized_distances(points: torch.Tensor, centroid: torch.Tensor) -> np.ndarray:
    values = torch.linalg.vector_norm(points - centroid, dim=1) / SQRT_DIM
    return values.numpy().astype(np.float64, copy=False)


def scores_for_draw(
    benign_z: torch.Tensor,
    adversarial_z: torch.Tensor,
    adversarial_scale: float,
    mean_shift: float,
) -> tuple[np.ndarray, np.ndarray]:
    benign = benign_z * BENIGN_SCALE
    centroid = benign.mean(dim=0)
    adversarial = adversarial_z * adversarial_scale + mean_shift
    benign_scores = normalized_distances(benign, centroid)
    adversarial_scores = normalized_distances(adversarial, centroid)
    return benign_scores, adversarial_scores


def auc_value(benign_scores: np.ndarray, adversarial_scores: np.ndarray) -> float:
    labels = np.concatenate(
        [
            np.zeros(benign_scores.size, dtype=np.int8),
            np.ones(adversarial_scores.size, dtype=np.int8),
        ]
    )
    scores = np.concatenate([benign_scores, adversarial_scores])
    return float(roc_auc_score(labels, scores))


def threshold_arrays(
    benign_scores: np.ndarray,
    adversarial_scores: np.ndarray,
) -> dict[str, np.ndarray]:
    benign_positive = benign_scores[:, None] > RHOS[None, :]
    adversarial_positive = adversarial_scores[:, None] > RHOS[None, :]

    fpr = benign_positive.mean(axis=0, dtype=np.float64)
    tpr = adversarial_positive.mean(axis=0, dtype=np.float64)
    tp = tpr * N_ADVERSARIAL
    fp = fpr * N_BENIGN
    fn = (1.0 - tpr) * N_ADVERSARIAL

    precision = np.divide(
        tp,
        tp + fp,
        out=np.zeros_like(tp),
        where=(tp + fp) > 0,
    )
    f1 = np.divide(
        2.0 * tp,
        2.0 * tp + fp + fn,
        out=np.zeros_like(tp),
        where=(2.0 * tp + fp + fn) > 0,
    )
    return {
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision,
        "recall": tpr.copy(),
        "f1": f1,
    }


def summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p025": float(np.percentile(values, 2.5)),
        "p975": float(np.percentile(values, 97.5)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def environment_record() -> dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "execution_device": "cpu",
    }


def run(mode: str) -> dict[str, object]:
    if mode not in {"pilot", "official"}:
        raise ValueError("mode must be pilot or official")

    runs = PILOT_RUNS if mode == "pilot" else OFFICIAL_RUNS
    started = time.time()

    raw_auc_rows: list[dict[str, object]] = []
    raw_threshold_rows: list[dict[str, object]] = []
    auc_groups: dict[tuple[str, float], list[float]] = defaultdict(list)
    threshold_groups: dict[tuple[str, float, str], list[np.ndarray]] = defaultdict(list)

    for shift_index, mean_shift in enumerate(MEAN_SHIFTS):
        for run_index in range(runs):
            seed = run_seed(shift_index, run_index)
            benign_z, adversarial_z = standard_draws(seed)

            for condition, adversarial_scale in CONDITIONS:
                benign_scores, adversarial_scores = scores_for_draw(
                    benign_z=benign_z,
                    adversarial_z=adversarial_z,
                    adversarial_scale=adversarial_scale,
                    mean_shift=mean_shift,
                )
                auc = auc_value(benign_scores, adversarial_scores)
                metrics = threshold_arrays(benign_scores, adversarial_scores)

                raw_auc_rows.append(
                    {
                        "condition": condition,
                        "adversarial_scale": adversarial_scale,
                        "mean_shift": mean_shift,
                        "run": run_index,
                        "seed": seed,
                        "auc": auc,
                        "benign_score_mean": float(np.mean(benign_scores)),
                        "adversarial_score_mean": float(np.mean(adversarial_scores)),
                    }
                )
                auc_groups[(condition, mean_shift)].append(auc)

                for metric_name, values in metrics.items():
                    threshold_groups[(condition, mean_shift, metric_name)].append(values)

                for rho_index, rho in enumerate(RHOS):
                    raw_threshold_rows.append(
                        {
                            "condition": condition,
                            "adversarial_scale": adversarial_scale,
                            "mean_shift": mean_shift,
                            "run": run_index,
                            "seed": seed,
                            "rho": float(rho),
                            "tpr": float(metrics["tpr"][rho_index]),
                            "fpr": float(metrics["fpr"][rho_index]),
                            "precision": float(metrics["precision"][rho_index]),
                            "recall": float(metrics["recall"][rho_index]),
                            "f1": float(metrics["f1"][rho_index]),
                        }
                    )

    auc_summary_rows: list[dict[str, object]] = []
    threshold_summary_rows: list[dict[str, object]] = []
    band_rows: list[dict[str, object]] = []

    for condition, adversarial_scale in CONDITIONS:
        for mean_shift in MEAN_SHIFTS:
            auc_values = np.asarray(auc_groups[(condition, mean_shift)], dtype=np.float64)
            auc_summary_rows.append(
                {
                    "condition": condition,
                    "adversarial_scale": adversarial_scale,
                    "mean_shift": mean_shift,
                    **{f"auc_{k}": v for k, v in summary(auc_values).items()},
                }
            )

            metric_matrices = {
                metric: np.vstack(threshold_groups[(condition, mean_shift, metric)])
                for metric in ["tpr", "fpr", "precision", "recall", "f1"]
            }
            mean_f1 = metric_matrices["f1"].mean(axis=0)
            qualifies = mean_f1 >= 0.95
            qualifying_rhos = RHOS[qualifies]
            if qualifying_rhos.size:
                band_lo = float(qualifying_rhos.min())
                band_hi = float(qualifying_rhos.max())
                band_count = int(qualifying_rhos.size)
                band_width = float(band_hi - band_lo)
            else:
                band_lo = None
                band_hi = None
                band_count = 0
                band_width = 0.0

            band_rows.append(
                {
                    "condition": condition,
                    "adversarial_scale": adversarial_scale,
                    "mean_shift": mean_shift,
                    "band_lo": band_lo,
                    "band_hi": band_hi,
                    "band_count": band_count,
                    "band_width": band_width,
                }
            )

            for rho_index, rho in enumerate(RHOS):
                row: dict[str, object] = {
                    "condition": condition,
                    "adversarial_scale": adversarial_scale,
                    "mean_shift": mean_shift,
                    "rho": float(rho),
                }
                for metric, matrix in metric_matrices.items():
                    values = matrix[:, rho_index]
                    row[f"{metric}_mean"] = float(np.mean(values))
                    row[f"{metric}_p025"] = float(np.percentile(values, 2.5))
                    row[f"{metric}_p975"] = float(np.percentile(values, 97.5))
                threshold_summary_rows.append(row)

    contrast_rows: list[dict[str, object]] = []
    for mean_shift in MEAN_SHIFTS:
        equal_auc = np.asarray(auc_groups[("equal_variance", mean_shift)], dtype=np.float64)
        unequal_auc = np.asarray(
            auc_groups[("unequal_variance_comparator", mean_shift)],
            dtype=np.float64,
        )
        delta = unequal_auc - equal_auc
        contrast_rows.append(
            {
                "mean_shift": mean_shift,
                **{f"delta_auc_{k}": v for k, v in summary(delta).items()},
                "equal_auc_mean": float(np.mean(equal_auc)),
                "unequal_auc_mean": float(np.mean(unequal_auc)),
            }
        )

    prefix = "phase_d" if mode == "official" else "phase_d_pilot"
    write_csv(
        RESULTS_DIR / f"{prefix}_raw_auc_runs.csv",
        [
            "condition",
            "adversarial_scale",
            "mean_shift",
            "run",
            "seed",
            "auc",
            "benign_score_mean",
            "adversarial_score_mean",
        ],
        raw_auc_rows,
    )
    write_csv(
        RESULTS_DIR / f"{prefix}_raw_threshold_runs.csv",
        [
            "condition",
            "adversarial_scale",
            "mean_shift",
            "run",
            "seed",
            "rho",
            "tpr",
            "fpr",
            "precision",
            "recall",
            "f1",
        ],
        raw_threshold_rows,
    )
    write_csv(
        RESULTS_DIR / f"{prefix}_auc_summary.csv",
        list(auc_summary_rows[0].keys()),
        auc_summary_rows,
    )
    write_csv(
        RESULTS_DIR / f"{prefix}_threshold_summary.csv",
        list(threshold_summary_rows[0].keys()),
        threshold_summary_rows,
    )
    write_csv(
        RESULTS_DIR / f"{prefix}_threshold_bands.csv",
        list(band_rows[0].keys()),
        band_rows,
    )
    write_csv(
        RESULTS_DIR / f"{prefix}_condition_contrasts.csv",
        list(contrast_rows[0].keys()),
        contrast_rows,
    )

    payload = {
        "mode": mode,
        "runs_per_cell": runs,
        "config": {
            "base_seed": BASE_SEED,
            "dimension": DIM,
            "n_benign": N_BENIGN,
            "n_adversarial": N_ADVERSARIAL,
            "benign_scale": BENIGN_SCALE,
            "conditions": [
                {"id": condition, "adversarial_scale": scale}
                for condition, scale in CONDITIONS
            ],
            "mean_shifts": MEAN_SHIFTS,
            "rho_min": float(RHOS[0]),
            "rho_max": float(RHOS[-1]),
            "rho_step": 0.01,
            "rho_count": int(RHOS.size),
            "common_random_numbers": True,
            "distance": "dimension-normalized Euclidean distance",
        },
        "environment": environment_record(),
        "auc_summary": auc_summary_rows,
        "band_summary": band_rows,
        "contrasts": contrast_rows,
        "raw_auc_rows": len(raw_auc_rows),
        "raw_threshold_rows": len(raw_threshold_rows),
        "elapsed_seconds": float(time.time() - started),
    }
    persisted = {
        key: value
        for key, value in payload.items()
        if key not in {"environment", "elapsed_seconds", "mode"}
    }
    (RESULTS_DIR / f"{prefix}_sensitivity.json").write_text(
        json.dumps(persisted, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    selected_mode = sys.argv[1] if len(sys.argv) > 1 else "official"
    result = run(selected_mode)
    print(json.dumps(
        {
            "mode": result["mode"],
            "runs_per_cell": result["runs_per_cell"],
            "raw_auc_rows": result["raw_auc_rows"],
            "raw_threshold_rows": result["raw_threshold_rows"],
            "elapsed_seconds": result["elapsed_seconds"],
        },
        indent=2,
    ))
