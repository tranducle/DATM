# Reproducibility Guide

This document records the computational provenance for the DATM study. It is organized around the exact inputs, models, split rules, thresholds, seeds, scripts, and canonical outputs used to support the reported bounded computational illustrations.

## Global conventions

- Base seed: `42`
- Embedding model for Phases A and C: `all-mpnet-base-v2`
- NLI model for Phase B: `cross-encoder/nli-deberta-v3-large`
- Threshold-dependent metrics use a threshold selected on the training split and then frozen on the corresponding test split unless a fixed threshold is explicitly stated.
- The positive class is the negative/adversarial class in Phase A, benchmark-incorrect responses in Phase B, and exfiltration in Phase C.
- Small constructed datasets are included directly under `data/`. TruthfulQA is public and is loaded from its public dataset source rather than redistributed.

The machine-readable configuration is `configs/reproducibility.yaml`.

## Phase A: Boundary mapping

### Core prompt-family evaluation

**Input:** `data/phase_a_boundary_prompts.csv`

The dataset contains 20 automotive prompt families and 20 negative prompt families. Each seed prompt is deterministically expanded into five surface variants: the original prompt, lowercase text, an appended detail request, a prefixed domain request, and a punctuation-stripped polite form. Family-disjoint evaluation prevents surface variants from the same seed from crossing the train/test boundary.

**Script:** `experiments/phase_a_boundary.py`

**Primary resampling:**

- 200 repetitions
- 14 prompt families per class for training
- 6 prompt families per class for testing
- benign centroid fitted only from training automotive embeddings
- threshold selected on the training split by maximizing Youden J
- threshold frozen for the corresponding test split

### Multi-domain benign-space stress

**Input:** `data/phase_a_multidomain_prompts.csv`

The stress set broadens benign traffic across automotive, HR, finance, and IT-operations families and compares it with several negative domains. The derived random seed is `4042`.

### In-domain malicious stress

**Input:** `data/phase_a_in_domain_stress.csv`

This stress set keeps both classes inside the automotive domain and tests whether the fixed-centroid score can distinguish benign automotive requests from policy-violating automotive requests. It uses 150 repetitions, 14 training families per class, 6 test families per class, and derived seed `10042`.

**Canonical summary:** `results/paper_results_manifest.json`

## Phase B: Response-integrity proxy

**Dataset:** Hugging Face dataset `truthfulqa/truthful_qa`, configuration `generation`, split `validation`

The expected usable set contains 817 questions. For each question, the public `best_answer` is treated as the benchmark-correct response and up to 10 entries from `incorrect_answers` are treated as benchmark-incorrect candidate responses.

**Script:** `experiments/phase_b_response_integrity.py`

The multi-signal score uses:

- contradiction logit, weight `1.0`
- contradiction-minus-entailment gap, weight `0.45`
- question-answer cosine distance, weight `4.4`
- answer-to-question length ratio, weight `1.45`

Signal standardization is fitted using training questions only. Incorrect-answer scores are max-pooled at the question level.

### Single hold-out

- base seed `42`
- 408 training questions
- 409 test questions
- threshold selected on training data by maximizing Youden J
- threshold frozen on test data

### Bounded repeated audit

- fixed 240-question subset selected from the public benchmark
- 100 train/test re-splits
- 120 training and 120 test questions per split
- up to 3 incorrect answers per question
- split-bank seed `7042`
- compares the fixed four-signal fusion with contradiction-only scoring

**Canonical summary:** `results/paper_results_manifest.json`

## Phase C: Exfiltration-template evaluation

### Original template task

**Input:** `data/phase_c_original_templates.csv`

The dataset contains 15 unique benign templates and 15 unique exfiltration templates. Evaluation uses template identity as the independent unit.

**Script:** `experiments/phase_c_exfiltration.py`

- 200 balanced resamples
- 10 training templates per class
- 5 test templates per class
- derived seed `8042`
- DATM centroid threshold selected by Youden J on training data and frozen on test data
- hybrid lexical comparator uses character 3-to-5 grams and word 1-to-2 grams with balanced logistic regression and fixed probability threshold `0.5`
- regex proxy is retained as a transparent weak baseline

**Detailed output:** `results/phase_c_operating_points.json`

### Paraphrastic stress

**Input:** `data/phase_c_paraphrastic_templates.csv`

- 12 templates per class
- 8 training templates and 4 test templates per class
- 200 resamples
- derived seed `3042`

This set is retained as an auxiliary stress test. Its contrast with regex is interpreted only as evidence of regex brittleness, not as evidence of general semantic superiority.

### Marker-scrubbed stress

**Input:** `data/phase_c_marker_scrubbed_templates.csv`

- 12 templates per class
- 8 training templates and 4 test templates per class
- 200 balanced resamples
- derived seed `9042`

The marker-scrubbed condition removes obvious lexical markers that made the original synthetic task easier. The detailed operating-point output is in `results/phase_c_operating_points.json`.

### Auxiliary matched-threshold diagnostic

`results/phase_c_matched_thresholds.json` records an auxiliary diagnostic in which both DATM and the hybrid lexical comparator select Youden J on the training split. This diagnostic is provided for transparency and is not substituted for the primary fixed-0.5 hybrid comparison reported in the study.

## Phase D: Monte Carlo sensitivity analysis

**Script:** `experiments/phase_d_sensitivity.py`

Configuration:

- seed `42`
- embedding dimension `384`
- 300 Monte Carlo runs per `(sigma, rho)` cell
- 500 benign and 200 adversarial synthetic samples per run
- benign Gaussian scale `0.10`
- adversarial Gaussian scale `0.15`
- adversarial mean-shift levels `0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50`
- candidate threshold grid from `0.05` to `<0.50` in increments of `0.01`
- dimension-normalized Euclidean distance
- standard F1 computed within each Monte Carlo run from the declared 200 adversarial and 500 benign samples, then averaged across runs for each `(sigma, rho)` cell

**Reference sweep:** `results/phase_d_reference_sweep.csv`

The simulation is a stylized sensitivity analysis. It should not be interpreted as an empirical deployment benchmark.

## Provenance map

| Evidence block | Input | Script | Seed | Threshold rule | Canonical output |
|---|---|---|---:|---|---|
| Phase A family resampling | `data/phase_a_boundary_prompts.csv` | `experiments/phase_a_boundary.py` | 42 | Train Youden J | `results/paper_results_manifest.json` |
| Phase A multi-domain stress | `data/phase_a_multidomain_prompts.csv` | `experiments/phase_a_boundary.py` | 4042 | Train Youden J | `results/paper_results_manifest.json` |
| Phase A in-domain stress | `data/phase_a_in_domain_stress.csv` | `experiments/phase_a_boundary.py` | 10042 | Train Youden J | `results/paper_results_manifest.json` |
| Phase B hold-out | Public TruthfulQA | `experiments/phase_b_response_integrity.py` | 42 | Train Youden J | `results/paper_results_manifest.json` |
| Phase B repeated audit | Public TruthfulQA | `experiments/phase_b_response_integrity.py` | 7042 split bank | Train Youden J | `results/paper_results_manifest.json` |
| Phase C original comparator | `data/phase_c_original_templates.csv` | `experiments/phase_c_exfiltration.py` | 8042 | DATM: train Youden J; hybrid: 0.5 | `results/phase_c_operating_points.json` |
| Phase C paraphrastic stress | `data/phase_c_paraphrastic_templates.csv` | `experiments/phase_c_stress.py` | 3042 | DATM: train Youden J; hybrid: 0.5 | `results/paper_results_manifest.json` |
| Phase C marker-scrubbed stress | `data/phase_c_marker_scrubbed_templates.csv` | `experiments/phase_c_stress.py` | 9042 | DATM: train Youden J; hybrid: 0.5 | `results/phase_c_operating_points.json` |
| Phase D sensitivity sweep | Synthetic Gaussian draws | `experiments/phase_d_sensitivity.py` | 42 | Explicit rho grid | `results/phase_d_reference_sweep.csv` |

## Scope and interpretation

The repository is designed to make the reported computational analyses inspectable and reproducible. It does not reproduce the retrospective enterprise incident evidence because those cases are based on cited public sources rather than generated datasets. It also does not claim that the bounded proxy and synthetic tasks establish enterprise deployment effectiveness, external validity, or calibrated production risk estimates.
