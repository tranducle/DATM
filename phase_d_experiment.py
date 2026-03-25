#!/usr/bin/env python3
"""
DATM Phase D — Monte Carlo Sensitivity Analysis
=================================================
Standalone experiment for autonomous optimization via auto_experiment_tools.py.

Target metric: phase_d_mean_f1 (maximize)

The Monte Carlo simulation tests framework robustness by varying:
  - σ (attack sophistication): how much adversarial embeddings deviate
  - ρ (detection threshold): the boundary enforcement parameter

The LLM can tune:
  - Embedding dimension and generation strategy
  - Number of benign/adversarial samples
  - Attack distribution parameters (mean shift, covariance)
  - Benign cluster tightness
  - Detection metric (Euclidean, cosine, Mahalanobis)
  - Adaptive threshold strategies
  - Number of MC runs for stability

Output format: phase_d_mean_f1: X.XXXX
"""

import os, json, time, warnings
import numpy as np
import torch
from pathlib import Path

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIGURABLE PARAMETERS (LLM can tune these)
# ============================================================

# Embedding dimension
DIM = 384

# Number of Monte Carlo runs per configuration
MC_RUNS = 300

# Benign cluster parameters
N_BENIGN = 500
BENIGN_SCALE = 0.01  # std of benign embeddings

# Adversarial parameters
N_ADVERSARIAL = 200
ADVERSARIAL_SCALE = 0.65  # std of adversarial embeddings

# Attack sophistication levels (σ)
SOPH_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

# Detection thresholds (ρ)
RHO_MIN = 0.05
RHO_MAX = 0.50
RHO_STEP = 0.01

# Distance metric: "euclidean" or "cosine" or "mahalanobis"
DISTANCE_METRIC = "euclidean"

# Whether to use dimensionality-normalized distances
NORMALIZE_BY_DIM = True

# Adaptive threshold: if True, use per-σ optimal ρ instead of fixed grid
ADAPTIVE_THRESHOLD = False

# ============================================================
# MAIN EXPERIMENT
# ============================================================

def compute_distances(points, centroid, metric="euclidean"):
    """Compute distances from points to centroid."""
    if metric == "euclidean":
        dists = torch.sqrt(torch.sum((points - centroid)**2, dim=1))
        if NORMALIZE_BY_DIM:
            dists /= np.sqrt(DIM)
        return dists
    elif metric == "cosine":
        # Cosine distance = 1 - cosine_similarity
        norms_p = torch.linalg.norm(points, dim=1, keepdim=True)
        norm_c = torch.linalg.norm(centroid)
        sims = (points @ centroid) / (norms_p.flatten() * norm_c + 1e-10)
        return 1 - sims
    elif metric == "mahalanobis":
        # Simplified: use diagonal covariance
        diff = points - centroid
        var = torch.var(points, dim=0, unbiased=False) + 1e-6
        return torch.sqrt(torch.sum(diff**2 / var, dim=1))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def run_experiment():
    """Run the Phase D Monte Carlo experiment and print metrics."""
    rhos = np.arange(RHO_MIN, RHO_MAX, RHO_STEP)
    
    # Print params for snapshot tracking
    import json as _json
    _params = {
        "DIM": DIM, "MC_RUNS": MC_RUNS,
        "N_BENIGN": N_BENIGN, "BENIGN_SCALE": BENIGN_SCALE,
        "N_ADVERSARIAL": N_ADVERSARIAL, "ADVERSARIAL_SCALE": ADVERSARIAL_SCALE,
        "DISTANCE_METRIC": DISTANCE_METRIC, "NORMALIZE_BY_DIM": NORMALIZE_BY_DIM,
        "RHO_STEP": RHO_STEP, "ADAPTIVE_THRESHOLD": ADAPTIVE_THRESHOLD,
    }
    print(f"PARAMS: {_json.dumps(_params)}")
    
    print(f"[Phase D] MC: {MC_RUNS} runs x {len(SOPH_LEVELS)} std_devs x {len(rhos)} rhos")
    print(f"[Phase D] Metric: {DISTANCE_METRIC}, Dim: {DIM}")
    print(f"[Phase D] Benign: N={N_BENIGN}, scale={BENIGN_SCALE}")
    print(f"[Phase D] Adversarial: N={N_ADVERSARIAL}, scale={ADVERSARIAL_SCALE}")
    
    all_f1s = []
    rows = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Phase D] device: {device}")
    
    for s in SOPH_LEVELS:
        for rho in rhos:
            tprs, fprs = [], []
            for _ in range(MC_RUNS):
                # Generate benign cluster
                b = torch.randn(N_BENIGN, DIM, device=device) * BENIGN_SCALE
                # Generate adversarial samples (shifted by σ)
                a = torch.randn(N_ADVERSARIAL, DIM, device=device) * ADVERSARIAL_SCALE + s
                
                # Compute centroid from benign samples
                c = torch.mean(b, dim=0)
                
                # Compute distances
                bd = compute_distances(b, c, DISTANCE_METRIC)
                ad = compute_distances(a, c, DISTANCE_METRIC)
                
                tprs.append(torch.mean((ad > rho).float()).item())
                fprs.append(torch.mean((bd > rho).float()).item())
            
            mean_tpr = np.mean(tprs)
            mean_fpr = np.mean(fprs)
            precision = mean_tpr / (mean_tpr + mean_fpr + 1e-10)
            f1 = 2 * mean_tpr * (1 - mean_fpr) / (mean_tpr + 1 - mean_fpr + 1e-10)
            
            rows.append({
                'soph': s, 'rho': rho,
                'tpr': mean_tpr, 'fpr': mean_fpr,
                'tpr_lo': np.percentile(tprs, 2.5),
                'tpr_hi': np.percentile(tprs, 97.5),
                'fpr_lo': np.percentile(fprs, 2.5),
                'fpr_hi': np.percentile(fprs, 97.5),
                'f1': f1, 'precision': precision,
            })
            all_f1s.append(f1)
    
    mean_f1 = np.mean(all_f1s)
    
    # Also compute AUC-like metric: mean of best F1 per σ level
    best_f1_per_soph = []
    for s in SOPH_LEVELS:
        soph_f1s = [r['f1'] for r in rows if r['soph'] == s]
        best_f1_per_soph.append(max(soph_f1s))
    
    mean_best_f1 = np.mean(best_f1_per_soph)
    min_best_f1 = min(best_f1_per_soph)
    
    # Composite metric: mean F1 across ALL σ×ρ cells (reflects overall robustness)
    # This has more room for improvement than per-σ best (which saturates at 1.0)
    composite = mean_f1
    
    # Print metrics
    print(f"phase_d_mean_f1: {composite:.6f}")
    print(f"mean_f1_all: {mean_f1:.6f}")
    print(f"mean_best_f1_per_soph: {mean_best_f1:.6f}")
    print(f"min_best_f1: {min_best_f1:.6f}")
    
    for i, s in enumerate(SOPH_LEVELS):
        print(f"best_f1_soph_{s:.2f}: {best_f1_per_soph[i]:.6f}")
    
    # Save results
    results = {
        "composite_f1": float(composite),
        "mean_f1": float(mean_f1),
        "mean_best_f1": float(mean_best_f1),
        "min_best_f1": float(min_best_f1),
        "best_f1_per_soph": {str(s): float(f) for s, f in zip(SOPH_LEVELS, best_f1_per_soph)},
        "config": {
            "dim": DIM, "mc_runs": MC_RUNS,
            "n_benign": N_BENIGN, "n_adversarial": N_ADVERSARIAL,
            "benign_scale": BENIGN_SCALE, "adversarial_scale": ADVERSARIAL_SCALE,
            "distance_metric": DISTANCE_METRIC,
            "normalize_by_dim": NORMALIZE_BY_DIM,
        }
    }
    
    with open(OUTPUT_DIR / "phase_d_experiment_result.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return composite


if __name__ == "__main__":
    score = run_experiment()
    print(f"\n[Phase D] Final composite F1: {score:.4f}")
