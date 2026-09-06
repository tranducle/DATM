"""Phase A boundary-mapping evaluations and stress tests."""

from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

from common import DATA_DIR, load_texts, score_centroid_detector, split_indices, summarize, write_json

SEED_BASE = 42
EMBEDDING_MODEL = "all-mpnet-base-v2"


def deterministic_augment(prompt: str, prefix: str) -> list[str]:
    return [
        prompt,
        prompt.lower(),
        f"{prompt} Please respond in detail.",
        f"{prefix} {prompt}",
        prompt.replace("?", "").replace(".", "") + " please",
    ]


def encode_families(model, prompts: list[str], prefix: str) -> list[np.ndarray]:
    return [
        model.encode(deterministic_augment(prompt, prefix), show_progress_bar=False, batch_size=32)
        for prompt in prompts
    ]


def run_two_class_resampling(
    model,
    csv_name: str,
    benign_label: str,
    attack_label: str,
    n_train: int,
    n_repetitions: int,
    seed: int,
    prefix: str,
) -> dict[str, object]:
    groups = load_texts(csv_name)
    benign = groups[benign_label]
    attack = groups[attack_label]
    benign_bank = encode_families(model, benign, prefix)
    attack_bank = encode_families(model, attack, prefix)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_repetitions):
        b_train, b_test = split_indices(len(benign), n_train, rng)
        a_train, a_test = split_indices(len(attack), n_train, rng)
        rows.append(
            score_centroid_detector(
                np.concatenate([benign_bank[i] for i in b_train]),
                np.concatenate([attack_bank[i] for i in a_train]),
                np.concatenate([benign_bank[i] for i in b_test]),
                np.concatenate([attack_bank[i] for i in a_test]),
            )
        )
    return {
        "seed": seed,
        "n_repetitions": n_repetitions,
        "train_prompt_families_per_class": n_train,
        "test_prompt_families_per_class": len(benign) - n_train,
        "auc": summarize(row["auc"] for row in rows),
        "precision": summarize(row["precision"] for row in rows),
        "recall": summarize(row["recall"] for row in rows),
        "f1": summarize(row["f1"] for row in rows),
        "threshold": summarize(row["threshold"] for row in rows),
    }


def load_multidomain() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    benign: dict[str, list[str]] = defaultdict(list)
    negative: dict[str, list[str]] = defaultdict(list)
    with (DATA_DIR / "phase_a_multidomain_prompts.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            target = benign if row["class"] == "benign" else negative
            target[row["domain"]].append(row["text"])
    return dict(benign), dict(negative)


def run_multidomain(model) -> dict[str, object]:
    benign_domains, negative_domains = load_multidomain()
    benign_bank = {
        domain: encode_families(model, prompts, "Business request:")
        for domain, prompts in benign_domains.items()
    }
    negative_bank = {
        domain: encode_families(model, prompts, "Quick request:")
        for domain, prompts in negative_domains.items()
    }
    rng = np.random.default_rng(SEED_BASE + 4000)
    rows = []
    for _ in range(150):
        train_benign = []
        test_benign = []
        for families in benign_bank.values():
            train_idx, test_idx = split_indices(len(families), 6, rng)
            train_benign.extend(families[i] for i in train_idx)
            test_benign.extend(families[i] for i in test_idx)
        train_negative = []
        test_negative = []
        for families in negative_bank.values():
            train_idx, test_idx = split_indices(len(families), 6, rng)
            train_negative.extend(families[i] for i in train_idx)
            test_negative.extend(families[i] for i in test_idx)
        rows.append(
            score_centroid_detector(
                np.concatenate(train_benign), np.concatenate(train_negative),
                np.concatenate(test_benign), np.concatenate(test_negative),
            )
        )
    return {
        "seed": SEED_BASE + 4000,
        "n_repetitions": 150,
        "benign_domains": sorted(benign_domains),
        "negative_domains": sorted(negative_domains),
        "auc": summarize(row["auc"] for row in rows),
        "precision": summarize(row["precision"] for row in rows),
        "recall": summarize(row["recall"] for row in rows),
        "f1": summarize(row["f1"] for row in rows),
    }


def run() -> dict[str, object]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    result = {
        "embedding_model": EMBEDDING_MODEL,
        "boundary_family_resampling": run_two_class_resampling(
            model, "phase_a_boundary_prompts.csv", "automotive", "negative",
            n_train=14, n_repetitions=200, seed=SEED_BASE, prefix="Automotive request:",
        ),
        "multidomain_benign_stress": run_multidomain(model),
        "in_domain_malicious_stress": run_two_class_resampling(
            model, "phase_a_in_domain_stress.csv", "benign", "malicious",
            n_train=14, n_repetitions=150, seed=SEED_BASE + 10000, prefix="Automotive request:",
        ),
    }
    write_json("phase_a_results.json", result)
    return result


if __name__ == "__main__":
    payload = run()
    print("Phase A evaluations complete")
    for name, block in payload.items():
        if isinstance(block, dict) and "auc" in block:
            print(f"{name}: mean AUC={block['auc']['mean']:.4f}, mean F1={block['f1']['mean']:.4f}")
