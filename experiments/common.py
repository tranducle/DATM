"""Shared utilities for DATM computational illustrations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import auc, precision_recall_fscore_support, roc_curve
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def load_texts(csv_name: str) -> dict[str, list[str]]:
    """Load a two-class prompt/template CSV grouped by its label column."""
    groups: dict[str, list[str]] = {}
    with (DATA_DIR / csv_name).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            groups.setdefault(row["label"], []).append(row["text"])
    return groups


def split_indices(n: int, n_train: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    rng.shuffle(idx)
    return np.sort(idx[:n_train]), np.sort(idx[n_train:])


def find_best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Select a training threshold by maximizing Youden J."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    best_idx = int(np.argmax(tpr - fpr))
    return float(thresholds[best_idx]), float(auc(fpr, tpr))


def metrics_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (scores > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred, average="binary", zero_division=0
    )
    fpr, tpr, _ = roc_curve(labels, scores)
    return {
        "auc": float(auc(fpr, tpr)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def summarize(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def score_centroid_detector(
    train_benign_emb: np.ndarray,
    train_attack_emb: np.ndarray,
    test_benign_emb: np.ndarray,
    test_attack_emb: np.ndarray,
) -> dict[str, float]:
    """Fit a benign centroid, choose a threshold on training data, and score held-out data."""
    centroid = np.mean(train_benign_emb, axis=0)
    train_benign_scores = 1 - cosine_similarity(train_benign_emb, centroid.reshape(1, -1)).ravel()
    train_attack_scores = 1 - cosine_similarity(train_attack_emb, centroid.reshape(1, -1)).ravel()
    test_benign_scores = 1 - cosine_similarity(test_benign_emb, centroid.reshape(1, -1)).ravel()
    test_attack_scores = 1 - cosine_similarity(test_attack_emb, centroid.reshape(1, -1)).ravel()

    train_labels = np.concatenate([np.zeros(len(train_benign_scores)), np.ones(len(train_attack_scores))])
    train_scores = np.concatenate([train_benign_scores, train_attack_scores])
    test_labels = np.concatenate([np.zeros(len(test_benign_scores)), np.ones(len(test_attack_scores))])
    test_scores = np.concatenate([test_benign_scores, test_attack_scores])

    threshold, train_auc = find_best_threshold(train_labels, train_scores)
    metrics = metrics_at_threshold(test_labels, test_scores, threshold)
    return {"train_auc": train_auc, "threshold": threshold, **metrics}


def write_json(name: str, payload: object) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
