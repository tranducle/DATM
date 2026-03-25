#!/usr/bin/env python3
"""
DATM Phase B — Intent Auditing / Hallucination Detection (Eq. 3)
================================================================
Standalone experiment for autonomous optimization via auto_experiment_tools.py.

Target metric: phase_b_auc (maximize)

Current approach: Cross-encoder NLI contradiction scores on TruthfulQA.
The auto-experiment LLM can modify this file to try different strategies:
  - Different NLI models (larger cross-encoders)
  - Feature engineering (combining NLI logits, length ratios, etc.)
  - Ensemble scoring
  - Different distance/divergence metrics
  - Using more incorrect answers per question
  - Combining embedding-based and NLI-based signals

Output format: phase_b_auc: X.XXXX
"""

import os, json, time, warnings
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = Path(__file__).parent / "results"
FIGURE_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIGURABLE PARAMETERS (LLM can tune these)
# ============================================================

# NLI cross-encoder model for contradiction detection
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-large"

# Embedding model for cosine-based features
EMBEDDING_MODEL = "all-mpnet-base-v2"

# How many incorrect answers per question to use (1-10)
MAX_INCORRECT_PER_Q = 10

# Signal combination weights  
# Available signals: contradiction, entailment, neutral, entail_gap, cosine_dist, length_ratio
SIGNAL_WEIGHTS = {
    "contradiction": 1.0,
    "entail_gap": 0.45,
    "cosine_dist": 4.4,
    "length_ratio": 1.45,
}

# Whether to normalize signals before combining
NORMALIZE_SIGNALS = True

# ============================================================
# MAIN EXPERIMENT
# ============================================================

def load_truthfulqa():
    """Load TruthfulQA dataset."""
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    
    questions, correct_answers, incorrect_answers_list = [], [], []
    for row in ds:
        q = str(row.get('question', '')).strip()
        best = str(row.get('best_answer', '')).strip()
        inc_list = row.get('incorrect_answers', [])
        if q and best and inc_list and isinstance(inc_list, list) and len(inc_list) > 0:
            questions.append(q)
            correct_answers.append(best)
            # Take up to MAX_INCORRECT_PER_Q incorrect answers
            inc_subset = [str(a).strip() for a in inc_list[:MAX_INCORRECT_PER_Q] if str(a).strip()]
            incorrect_answers_list.append(inc_subset)
    
    return questions, correct_answers, incorrect_answers_list


def compute_nli_scores(questions, answers, nli_model):
    """Compute NLI scores (entail, neutral, contradict) for Q-A pairs."""
    pairs = [[q, a] for q, a in zip(questions, answers)]
    scores = nli_model.predict(pairs, batch_size=128)  # shape: (N, 3)
    return scores  # columns: [entailment, neutral, contradiction]


def compute_cosine_distances(questions, answers, emb_model):
    """Compute cosine distance between embedded Q and embedded A."""
    from sklearn.metrics.pairwise import cosine_similarity
    q_emb = emb_model.encode(questions, show_progress_bar=False, batch_size=512)
    a_emb = emb_model.encode(answers, show_progress_bar=False, batch_size=512)
    # Per-pair cosine distance
    sims = np.array([cosine_similarity(q.reshape(1,-1), a.reshape(1,-1))[0,0]
                     for q, a in zip(q_emb, a_emb)])
    return 1 - sims  # distance


def compute_length_ratio(questions, answers):
    """Length ratio between answer and question (proxy for verbosity)."""
    ratios = []
    for q, a in zip(questions, answers):
        q_len = max(len(q.split()), 1)
        a_len = len(a.split())
        ratios.append(a_len / q_len)
    return np.array(ratios)


def build_combined_signal(nli_scores, cosine_dists, length_ratios, weights):
    """Combine multiple signals into a single divergence score."""
    signals = {}
    
    # NLI-based signals
    signals["contradiction"] = nli_scores[:, 2]  # contradiction logit
    signals["entail_gap"] = nli_scores[:, 2] - nli_scores[:, 0]  # contra - entail
    
    # Embedding-based
    if cosine_dists is not None:
        signals["cosine_dist"] = cosine_dists
    
    # Length-based
    if length_ratios is not None:
        signals["length_ratio"] = length_ratios
    
    # Normalize if configured
    if NORMALIZE_SIGNALS:
        for key in signals:
            s = signals[key]
            std = np.std(s)
            if std > 0:
                signals[key] = (s - np.mean(s)) / std
    
    # Weighted combination
    combined = np.zeros(len(nli_scores))
    total_weight = 0
    for key, weight in weights.items():
        if weight > 0 and key in signals:
            combined += weight * signals[key]
            total_weight += weight
    
    if total_weight > 0:
        combined /= total_weight
    
    return combined


def run_experiment():
    """Run the Phase B experiment and print the AUC metric."""
    # Print params for snapshot tracking
    import json as _json
    _params = {
        "NLI_MODEL_NAME": NLI_MODEL_NAME,
        "EMBEDDING_MODEL": EMBEDDING_MODEL,
        "MAX_INCORRECT_PER_Q": MAX_INCORRECT_PER_Q,
        "SIGNAL_WEIGHTS": SIGNAL_WEIGHTS,
        "NORMALIZE_SIGNALS": NORMALIZE_SIGNALS,
        "aggregation": "max-pool",
    }
    print(f"PARAMS: {_json.dumps(_params)}")
    
    print(f"[Phase B] Loading models...")
    
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    nli_model = CrossEncoder(NLI_MODEL_NAME, default_activation_function=None, device=device, model_kwargs={"torch_dtype": torch.float16})
    
    # Only load embedding model if needed
    use_embeddings = SIGNAL_WEIGHTS.get("cosine_dist", 0) > 0
    emb_model = SentenceTransformer(EMBEDDING_MODEL, device=device, model_kwargs={"torch_dtype": torch.float16}) if use_embeddings else None
    
    print(f"[Phase B] Loading TruthfulQA...")
    questions, correct_answers, incorrect_answers_list = load_truthfulqa()
    print(f"[Phase B] {len(questions)} questions loaded")
    
    # --- Correct answer scores ---
    print(f"[Phase B] Scoring correct answers...")
    correct_nli = compute_nli_scores(questions, correct_answers, nli_model)
    
    correct_cosine = None
    if use_embeddings:
        correct_cosine = compute_cosine_distances(questions, correct_answers, emb_model)
    
    correct_length = compute_length_ratio(questions, correct_answers)
    
    correct_signal = build_combined_signal(
        correct_nli, correct_cosine, correct_length, SIGNAL_WEIGHTS
    )
    
    # --- Incorrect answer scores (may have multiple per question) ---
    print(f"[Phase B] Scoring incorrect answers...")
    
    # Aggregate per question: compute combined signal for each incorrect answer,
    # then mean-pool the combined scores for robust aggregation
    incorrect_signals_per_q = []
    for q, inc_list in zip(questions, incorrect_answers_list):
        if len(inc_list) == 0:
            continue
        
        # Compute scores for all incorrect answers for this question
        inc_nli = compute_nli_scores([q] * len(inc_list), inc_list, nli_model)
        
        inc_cosine = None
        if use_embeddings:
            inc_cosine = compute_cosine_distances([q] * len(inc_list), inc_list, emb_model)
        
        inc_length = compute_length_ratio([q] * len(inc_list), inc_list)
        
        # Route through build_combined_signal for consistent normalization
        inc_combined = build_combined_signal(inc_nli, inc_cosine, inc_length, SIGNAL_WEIGHTS)
        
        # Max-pool the combined signals across all incorrect answers for this question (captures strongest contradiction signal)
        incorrect_signals_per_q.append(np.max(inc_combined))
    
    incorrect_signal = np.array(incorrect_signals_per_q)
    
    # --- ROC AUC ---
    labels = np.concatenate([np.zeros(len(correct_signal)), np.ones(len(incorrect_signal))])
    all_signal = np.concatenate([correct_signal, incorrect_signal])
    
    fpr, tpr, _ = roc_curve(labels, all_signal)
    roc_auc = auc(fpr, tpr)
    
    # Optimal threshold
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    
    # F1 at optimal threshold
    best_thresh_idx = np.argmax(j_scores)
    thresholds = np.linspace(np.min(all_signal), np.max(all_signal), 200)
    best_f1 = 0
    for t in thresholds:
        pred = (all_signal > t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(labels, pred, average='binary', zero_division=0)
        best_f1 = max(best_f1, f1)
    
    # Print metrics (auto_experiment_tools parses these)
    print(f"phase_b_auc: {roc_auc:.6f}")
    print(f"phase_b_f1: {best_f1:.6f}")
    print(f"correct_mean: {np.mean(correct_signal):.6f}")
    print(f"incorrect_mean: {np.mean(incorrect_signal):.6f}")
    print(f"separation: {np.mean(incorrect_signal) - np.mean(correct_signal):.6f}")
    print(f"n_correct: {len(correct_signal)}")
    print(f"n_incorrect: {len(incorrect_signal)}")
    
    # Save detailed results
    results = {
        "auc": float(roc_auc),
        "f1": float(best_f1),
        "correct_mean": float(np.mean(correct_signal)),
        "incorrect_mean": float(np.mean(incorrect_signal)),
        "separation": float(np.mean(incorrect_signal) - np.mean(correct_signal)),
        "nli_model": NLI_MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "signal_weights": SIGNAL_WEIGHTS,
        "normalize": NORMALIZE_SIGNALS,
        "max_incorrect_per_q": MAX_INCORRECT_PER_Q,
    }
    
    with open(OUTPUT_DIR / "phase_b_experiment_result.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return roc_auc


if __name__ == "__main__":
    auc_score = run_experiment()
    print(f"\n[Phase B] Final AUC: {auc_score:.4f}")
