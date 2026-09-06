"""Phase D Monte Carlo sensitivity analysis for stylized adversarial drift."""

from __future__ import annotations

import numpy as np
import torch

from common import write_json

SEED = 42
DIM = 384
MC_RUNS = 300
N_BENIGN = 500
BENIGN_SCALE = 0.01
N_ADVERSARIAL = 200
ADVERSARIAL_SCALE = 0.65
SOPH_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
RHO_MIN = 0.05
RHO_MAX = 0.50
RHO_STEP = 0.01
NORMALIZE_BY_DIM = True


def distances(points: torch.Tensor, centroid: torch.Tensor) -> torch.Tensor:
    d = torch.sqrt(torch.sum((points - centroid) ** 2, dim=1))
    return d / np.sqrt(DIM) if NORMALIZE_BY_DIM else d


def run() -> dict[str, object]:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rhos = np.arange(RHO_MIN, RHO_MAX, RHO_STEP)
    rows = []

    for sigma in SOPH_LEVELS:
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

            mean_tpr = float(np.mean(tprs))
            mean_fpr = float(np.mean(fprs))
            precision = mean_tpr / (mean_tpr + mean_fpr + 1e-10)
            f1 = 2 * mean_tpr * (1 - mean_fpr) / (mean_tpr + 1 - mean_fpr + 1e-10)
            rows.append({
                "sigma": sigma,
                "rho": float(rho),
                "tpr": mean_tpr,
                "fpr": mean_fpr,
                "tpr_lo": float(np.percentile(tprs, 2.5)),
                "tpr_hi": float(np.percentile(tprs, 97.5)),
                "fpr_lo": float(np.percentile(fprs, 2.5)),
                "fpr_hi": float(np.percentile(fprs, 97.5)),
                "precision": float(precision),
                "f1": float(f1),
            })

    result = {
        "seed": SEED,
        "dimension": DIM,
        "mc_runs_per_cell": MC_RUNS,
        "n_benign": N_BENIGN,
        "n_adversarial": N_ADVERSARIAL,
        "benign_scale": BENIGN_SCALE,
        "adversarial_scale": ADVERSARIAL_SCALE,
        "sophistication_levels": SOPH_LEVELS,
        "rho_min": RHO_MIN,
        "rho_max_exclusive": RHO_MAX,
        "rho_step": RHO_STEP,
        "normalize_by_dimension": NORMALIZE_BY_DIM,
        "rows": rows,
    }
    write_json("phase_d_sensitivity.json", result)
    return result


if __name__ == "__main__":
    payload = run()
    print(f"Phase D complete: {len(payload['rows'])} sigma/rho cells")
