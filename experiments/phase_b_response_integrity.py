"""Phase B response-integrity proxy on TruthfulQA.

TruthfulQA is loaded from its public Hugging Face dataset. The script reports
one 408/409 question-level hold-out using the fixed multi-signal score and a
bounded 240-question repeated audit comparing nine signal, aggregation, and
incorrect-answer-cap variants across 100 repeated 120/120 train/test re-splits.
Thresholds are selected on training questions and frozen on test data.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from common import find_best_threshold, metrics_at_threshold, summarize, write_json

SEED_BASE = 42
NLI_MODEL = "cross-encoder/nli-deberta-v3-large"
EMBEDDING_MODEL = "all-mpnet-base-v2"
MAX_INCORRECT = 10
FULL_WEIGHTS = {
    "contradiction": 1.0,
    "entail_gap": 0.45,
    "cosine_dist": 4.4,
    "length_ratio": 1.45,
}
ABLATION_SCHEMES = [
    {
        "name": "full_multi_signal_maxpool",
        "weights": dict(FULL_WEIGHTS),
        "aggregation": "max-pool",
        "incorrect_cap": 10,
    },
    {
        "name": "contradiction_only_maxpool",
        "weights": {"contradiction": 1.0},
        "aggregation": "max-pool",
        "incorrect_cap": 10,
    },
    {
        "name": "drop_length_ratio",
        "weights": {"contradiction": 1.0, "entail_gap": 0.45, "cosine_dist": 4.4},
        "aggregation": "max-pool",
        "incorrect_cap": 10,
    },
    {
        "name": "drop_cosine_dist",
        "weights": {"contradiction": 1.0, "entail_gap": 0.45, "length_ratio": 1.45},
        "aggregation": "max-pool",
        "incorrect_cap": 10,
    },
    {
        "name": "drop_entail_gap",
        "weights": {"contradiction": 1.0, "cosine_dist": 4.4, "length_ratio": 1.45},
        "aggregation": "max-pool",
        "incorrect_cap": 10,
    },
    {
        "name": "mean_pool_aggregation",
        "weights": dict(FULL_WEIGHTS),
        "aggregation": "mean-pool",
        "incorrect_cap": 10,
    },
    {
        "name": "top2_mean_aggregation",
        "weights": dict(FULL_WEIGHTS),
        "aggregation": "top2-mean",
        "incorrect_cap": 10,
    },
    {
        "name": "incorrect_cap_3",
        "weights": dict(FULL_WEIGHTS),
        "aggregation": "max-pool",
        "incorrect_cap": 3,
    },
    {
        "name": "incorrect_cap_1",
        "weights": dict(FULL_WEIGHTS),
        "aggregation": "max-pool",
        "incorrect_cap": 1,
    },
]


def load_truthfulqa() -> tuple[list[str], list[str], list[list[str]]]:
    from datasets import load_dataset

    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    questions, correct, incorrect = [], [], []
    for row in ds:
        q = str(row.get("question", "")).strip()
        best = str(row.get("best_answer", "")).strip()
        inc = [str(x).strip() for x in row.get("incorrect_answers", [])[:MAX_INCORRECT] if str(x).strip()]
        if q and best and inc:
            questions.append(q)
            correct.append(best)
            incorrect.append(inc)
    return questions, correct, incorrect


def nli_scores(questions: list[str], answers: list[str], model) -> np.ndarray:
    return np.asarray(model.predict([[q, a] for q, a in zip(questions, answers)], batch_size=64))


def cosine_distances(questions: list[str], answers: list[str], model) -> np.ndarray:
    q_emb = model.encode(questions, show_progress_bar=False, batch_size=256)
    a_emb = model.encode(answers, show_progress_bar=False, batch_size=256)
    return np.asarray([
        1 - cosine_similarity(q.reshape(1, -1), a.reshape(1, -1))[0, 0]
        for q, a in zip(q_emb, a_emb)
    ])


def length_ratios(questions: list[str], answers: list[str]) -> np.ndarray:
    return np.asarray([len(a.split()) / max(len(q.split()), 1) for q, a in zip(questions, answers)])


def aggregate_scores(per_answer: np.ndarray, aggregation: str) -> float:
    if aggregation == "max-pool":
        return float(np.max(per_answer))
    if aggregation == "mean-pool":
        return float(np.mean(per_answer))
    if aggregation == "top2-mean":
        top_k = min(2, len(per_answer))
        return float(np.mean(np.sort(per_answer)[-top_k:]))
    raise ValueError(f"Unsupported aggregation: {aggregation}")


def build_feature_store(questions, correct, incorrect, nli_model, emb_model):
    correct_nli = nli_scores(questions, correct, nli_model)
    correct_features = {
        "contradiction": correct_nli[:, 2],
        "entail_gap": correct_nli[:, 2] - correct_nli[:, 0],
        "cosine_dist": cosine_distances(questions, correct, emb_model),
        "length_ratio": length_ratios(questions, correct),
    }
    incorrect_features = []
    for q, answers in zip(questions, incorrect):
        scores = nli_scores([q] * len(answers), answers, nli_model)
        incorrect_features.append({
            "contradiction": scores[:, 2],
            "entail_gap": scores[:, 2] - scores[:, 0],
            "cosine_dist": cosine_distances([q] * len(answers), answers, emb_model),
            "length_ratio": length_ratios([q] * len(answers), answers),
        })
    return correct_features, incorrect_features


def fit_scalers(correct_features, incorrect_features, train_idx, keys, incorrect_cap):
    result = {}
    for key in keys:
        values = [correct_features[key][train_idx]]
        values.extend(incorrect_features[i][key][:incorrect_cap] for i in train_idx)
        flat = np.concatenate(values)
        mean = float(np.mean(flat))
        std = float(np.std(flat)) or 1.0
        result[key] = (mean, std)
    return result


def combine(
    correct_features,
    incorrect_features,
    idx,
    scalers,
    weights,
    incorrect_cap,
    aggregation: str = "max-pool",
):
    weight_sum = float(sum(weights.values()))
    correct_scores, incorrect_scores = [], []
    for i in idx:
        correct_total = 0.0
        for key, weight in weights.items():
            mean, std = scalers[key]
            correct_total += weight * ((correct_features[key][i] - mean) / std)
        correct_scores.append(correct_total / weight_sum)

        n_answers = min(incorrect_cap, len(incorrect_features[i][next(iter(weights))]))
        per_answer = np.zeros(n_answers, dtype=float)
        for key, weight in weights.items():
            mean, std = scalers[key]
            per_answer += weight * ((incorrect_features[i][key][:n_answers] - mean) / std)
        incorrect_scores.append(aggregate_scores(per_answer / weight_sum, aggregation))
    return np.asarray(correct_scores), np.asarray(incorrect_scores)


def evaluate_split(
    correct_features,
    incorrect_features,
    train_idx,
    test_idx,
    weights,
    incorrect_cap,
    aggregation: str = "max-pool",
):
    scalers = fit_scalers(correct_features, incorrect_features, train_idx, list(weights), incorrect_cap)
    train_correct, train_incorrect = combine(
        correct_features, incorrect_features, train_idx, scalers, weights, incorrect_cap, aggregation
    )
    test_correct, test_incorrect = combine(
        correct_features, incorrect_features, test_idx, scalers, weights, incorrect_cap, aggregation
    )
    train_labels = np.concatenate([np.zeros(len(train_correct)), np.ones(len(train_incorrect))])
    train_scores = np.concatenate([train_correct, train_incorrect])
    test_labels = np.concatenate([np.zeros(len(test_correct)), np.ones(len(test_incorrect))])
    test_scores = np.concatenate([test_correct, test_incorrect])
    threshold, train_auc = find_best_threshold(train_labels, train_scores)
    metrics = metrics_at_threshold(test_labels, test_scores, threshold)
    pred = (test_scores >= threshold).astype(int)
    tp = int(np.sum((test_labels == 1) & (pred == 1)))
    fn = int(np.sum((test_labels == 1) & (pred == 0)))
    fp = int(np.sum((test_labels == 0) & (pred == 1)))
    tn = int(np.sum((test_labels == 0) & (pred == 0)))
    return {
        "train_auc": train_auc,
        "threshold": threshold,
        "threshold_selected_on_train": threshold,
        **metrics,
        "test_positive": int(np.sum(test_labels == 1)),
        "test_negative": int(np.sum(test_labels == 0)),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
    }


def summarize_rows(rows):
    return {
        key: summarize(row[key] for row in rows)
        for key in ["train_auc", "auc", "precision", "recall", "f1", "threshold_selected_on_train"]
    }


def run() -> dict[str, object]:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    questions, correct, incorrect = load_truthfulqa()
    if len(questions) != 817:
        raise RuntimeError(f"Expected 817 TruthfulQA questions, found {len(questions)}")
    nli_model = CrossEncoder(NLI_MODEL, activation_fn=None)
    emb_model = SentenceTransformer(EMBEDDING_MODEL)
    correct_features, incorrect_features = build_feature_store(questions, correct, incorrect, nli_model, emb_model)

    rng = np.random.default_rng(SEED_BASE)
    idx = np.arange(len(questions))
    rng.shuffle(idx)
    train_idx = np.sort(idx[:408])
    test_idx = np.sort(idx[408:])
    holdout = evaluate_split(correct_features, incorrect_features, train_idx, test_idx, FULL_WEIGHTS, MAX_INCORRECT)
    train_incorrect_candidates = int(sum(min(MAX_INCORRECT, len(incorrect[i])) for i in train_idx))
    test_incorrect_candidates = int(sum(min(MAX_INCORRECT, len(incorrect[i])) for i in test_idx))

    subset_rng = np.random.default_rng(SEED_BASE)
    subset_idx = np.arange(len(questions))
    subset_rng.shuffle(subset_idx)
    subset_idx = np.sort(subset_idx[:240])
    split_rng = np.random.default_rng(SEED_BASE + 7000)
    split_bank = []
    for _ in range(100):
        local = subset_idx.copy()
        split_rng.shuffle(local)
        split_bank.append((np.sort(local[:120]), np.sort(local[120:])))

    schemes: dict[str, dict[str, object]] = {}
    for cfg in ABLATION_SCHEMES:
        rows = [
            evaluate_split(
                correct_features,
                incorrect_features,
                tr,
                te,
                cfg["weights"],
                cfg["incorrect_cap"],
                cfg["aggregation"],
            )
            for tr, te in split_bank
        ]
        schemes[cfg["name"]] = {
            "config": {
                "weights": dict(cfg["weights"]),
                "aggregation": cfg["aggregation"],
                "incorrect_cap": cfg["incorrect_cap"],
            },
            "summary": summarize_rows(rows),
        }

    result = {
        "dataset": "truthfulqa/truthful_qa: generation/validation",
        "n_questions": len(questions),
        "seed_base": SEED_BASE,
        "nli_model": NLI_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "full_weights": FULL_WEIGHTS,
        "single_holdout": {
            "train_questions": 408,
            "test_questions": 409,
            "evaluation_unit": "question level after max pooling incorrect candidates",
            "positive_class": "pooled benchmark incorrect",
            "raw_train_correct_candidates": 408,
            "raw_train_incorrect_candidates": train_incorrect_candidates,
            "raw_train_pairs": 408 + train_incorrect_candidates,
            "raw_test_correct_candidates": 409,
            "raw_test_incorrect_candidates": test_incorrect_candidates,
            "raw_test_pairs": 409 + test_incorrect_candidates,
            "pooled_train_observations": 816,
            "pooled_test_observations": 818,
            **holdout,
        },
        "bounded_repeated_audit": {
            "subset_questions": 240,
            "train_questions": 120,
            "test_questions": 120,
            "n_repetitions": 100,
            "default_max_incorrect_answers_per_question": MAX_INCORRECT,
            "full_multi_signal": schemes["full_multi_signal_maxpool"]["summary"],
            "contradiction_only": schemes["contradiction_only_maxpool"]["summary"],
            "schemes": schemes,
        },
    }
    write_json("phase_b_results.json", result)
    return result


if __name__ == "__main__":
    payload = run()
    h = payload["single_holdout"]
    print(f"Phase B hold-out AUC={h['auc']:.4f}, F1={h['f1']:.4f}")
