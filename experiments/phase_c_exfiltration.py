"""Phase C template-disjoint exfiltration evaluation.

The script compares the DATM centroid score with a hybrid lexical logistic
regression baseline and a transparent regex proxy on the original 15 + 15
constructed template set. Splits are performed at template identity level.
"""

from __future__ import annotations

import re

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import (
    find_best_threshold,
    load_texts,
    metrics_at_threshold,
    score_centroid_detector,
    split_indices,
    summarize,
    write_json,
)

SEED_BASE = 42
RESAMPLING_SEED = SEED_BASE + 8000
N_REPETITIONS = 200
TRAIN_TEMPLATES_PER_CLASS = 10
EMBEDDING_MODEL = "all-mpnet-base-v2"

REGEX_PATTERNS = [
    r"void\s+\w+\(", r"class\s+\w+\s*\{", r"def\s+\w+\(", r"#include\s*<",
    r"module\s+\w+\(", r"func\s+\w+\(", r"const\s+\w+\s*=", r"import\s+\w+",
    r"create\s+table", r"select\s+", r"\{.*\}", r"0x[0-9a-f]+", r"api[_-]?key",
    r"jwt[_-]?secret", r"confidential", r"internal", r"proprietary",
]


def score_hybrid_lexical(
    train_benign: list[str],
    train_attack: list[str],
    test_benign: list[str],
    test_attack: list[str],
) -> dict[str, float]:
    train_texts = train_benign + train_attack
    test_texts = test_benign + test_attack
    train_labels = np.asarray([0] * len(train_benign) + [1] * len(train_attack))
    test_labels = np.asarray([0] * len(test_benign) + [1] * len(test_attack))

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), lowercase=True, sublinear_tf=True
    )
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    train_char = char_vectorizer.fit_transform(train_texts)
    test_char = char_vectorizer.transform(test_texts)
    train_word = word_vectorizer.fit_transform(train_texts)
    test_word = word_vectorizer.transform(test_texts)

    model = LogisticRegression(
        max_iter=5000,
        solver="liblinear",
        class_weight="balanced",
        C=2.0,
        random_state=SEED_BASE,
    )
    model.fit(hstack([train_char, train_word]), train_labels)
    train_scores = model.predict_proba(hstack([train_char, train_word]))[:, 1]
    test_scores = model.predict_proba(hstack([test_char, test_word]))[:, 1]

    metrics = metrics_at_threshold(test_labels, test_scores, 0.5)
    return {
        "train_auc": float(roc_auc_score(train_labels, train_scores)),
        "threshold": 0.5,
        **metrics,
    }


def regex_score(text: str) -> float:
    lowered = text.lower()
    return float(sum(bool(re.search(pattern, lowered)) for pattern in REGEX_PATTERNS))


def score_regex(
    train_benign: list[str],
    train_attack: list[str],
    test_benign: list[str],
    test_attack: list[str],
) -> dict[str, float]:
    train_labels = np.asarray([0] * len(train_benign) + [1] * len(train_attack))
    test_labels = np.asarray([0] * len(test_benign) + [1] * len(test_attack))
    train_scores = np.asarray([regex_score(t) for t in train_benign + train_attack], dtype=float)
    test_scores = np.asarray([regex_score(t) for t in test_benign + test_attack], dtype=float)
    threshold, train_auc = find_best_threshold(train_labels, train_scores)
    return {"train_auc": train_auc, "threshold": threshold, **metrics_at_threshold(test_labels, test_scores, threshold)}


def summarize_rows(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        key: summarize(row[key] for row in rows)
        for key in ["auc", "precision", "recall", "f1", "threshold"]
    }


def run() -> dict[str, object]:
    from sentence_transformers import SentenceTransformer

    groups = load_texts("phase_c_original_templates.csv")
    benign = groups["benign"]
    attack = groups["exfiltration"]
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    benign_emb = encoder.encode(benign, show_progress_bar=False, batch_size=64)
    attack_emb = encoder.encode(attack, show_progress_bar=False, batch_size=64)

    rng = np.random.default_rng(RESAMPLING_SEED)
    datm_rows: list[dict[str, float]] = []
    lexical_rows: list[dict[str, float]] = []
    regex_rows: list[dict[str, float]] = []

    for _ in range(N_REPETITIONS):
        b_train, b_test = split_indices(len(benign), TRAIN_TEMPLATES_PER_CLASS, rng)
        a_train, a_test = split_indices(len(attack), TRAIN_TEMPLATES_PER_CLASS, rng)
        datm_rows.append(
            score_centroid_detector(
                benign_emb[b_train], attack_emb[a_train], benign_emb[b_test], attack_emb[a_test]
            )
        )
        lexical_rows.append(
            score_hybrid_lexical(
                [benign[i] for i in b_train], [attack[i] for i in a_train],
                [benign[i] for i in b_test], [attack[i] for i in a_test],
            )
        )
        regex_rows.append(
            score_regex(
                [benign[i] for i in b_train], [attack[i] for i in a_train],
                [benign[i] for i in b_test], [attack[i] for i in a_test],
            )
        )

    result = {
        "seed": RESAMPLING_SEED,
        "embedding_model": EMBEDDING_MODEL,
        "n_repetitions": N_REPETITIONS,
        "train_templates_per_class": TRAIN_TEMPLATES_PER_CLASS,
        "test_templates_per_class": len(benign) - TRAIN_TEMPLATES_PER_CLASS,
        "datm_centroid": summarize_rows(datm_rows),
        "hybrid_lexical": summarize_rows(lexical_rows),
        "regex_proxy": summarize_rows(regex_rows),
    }
    write_json("phase_c_original_resampling.json", result)
    return result


if __name__ == "__main__":
    payload = run()
    print("Phase C original-template resampling complete")
    print(f"DATM mean AUC: {payload['datm_centroid']['auc']['mean']:.4f}")
    print(f"Hybrid mean AUC: {payload['hybrid_lexical']['auc']['mean']:.4f}")
