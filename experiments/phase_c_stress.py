"""Phase C stress evaluations on paraphrastic and marker-scrubbed templates."""

from __future__ import annotations

import numpy as np

from common import load_texts, score_centroid_detector, split_indices, summarize, write_json
from phase_c_exfiltration import score_hybrid_lexical, score_regex

SEED_BASE = 42
EMBEDDING_MODEL = "all-mpnet-base-v2"
N_REPETITIONS = 200


def summarize_rows(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        key: summarize(row[key] for row in rows)
        for key in ["auc", "precision", "recall", "f1", "threshold"]
    }


def run_one(csv_name: str, n_train: int, seed: int, encoder) -> dict[str, object]:
    groups = load_texts(csv_name)
    benign = groups["benign"]
    attack = groups["exfiltration"]
    benign_emb = encoder.encode(benign, show_progress_bar=False, batch_size=64)
    attack_emb = encoder.encode(attack, show_progress_bar=False, batch_size=64)
    rng = np.random.default_rng(seed)

    datm_rows: list[dict[str, float]] = []
    lexical_rows: list[dict[str, float]] = []
    regex_rows: list[dict[str, float]] = []
    for _ in range(N_REPETITIONS):
        b_train, b_test = split_indices(len(benign), n_train, rng)
        a_train, a_test = split_indices(len(attack), n_train, rng)
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

    return {
        "seed": seed,
        "n_repetitions": N_REPETITIONS,
        "train_templates_per_class": n_train,
        "test_templates_per_class": len(benign) - n_train,
        "datm_centroid": summarize_rows(datm_rows),
        "hybrid_lexical": summarize_rows(lexical_rows),
        "regex_proxy": summarize_rows(regex_rows),
    }


def run() -> dict[str, object]:
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(EMBEDDING_MODEL)
    result = {
        "embedding_model": EMBEDDING_MODEL,
        "paraphrastic": run_one(
            "phase_c_paraphrastic_templates.csv", n_train=8, seed=SEED_BASE + 3000, encoder=encoder
        ),
        "marker_scrubbed": run_one(
            "phase_c_marker_scrubbed_templates.csv", n_train=8, seed=SEED_BASE + 9000, encoder=encoder
        ),
    }
    write_json("phase_c_stress_results.json", result)
    return result


if __name__ == "__main__":
    payload = run()
    marker = payload["marker_scrubbed"]
    print("Phase C stress evaluations complete")
    print(f"Marker-scrubbed DATM mean AUC: {marker['datm_centroid']['auc']['mean']:.4f}")
    print(f"Marker-scrubbed hybrid mean AUC: {marker['hybrid_lexical']['auc']['mean']:.4f}")
