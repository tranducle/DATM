# Reproducibility Guide

This document records the computational provenance for the DATM study. It is organized around the exact inputs, models, split rules, thresholds, seeds, scripts, and canonical outputs used to support the reported bounded computational illustrations.

## Global conventions

- Base seed: `42`
- Embedding model for Phases A and C: `all-mpnet-base-v2`
- NLI model for Phase B: `cross-encoder/nli-deberta-v3-large`
- Threshold-dependent metrics use a threshold selected on the training split and then frozen on the corresponding test split unless a fixed threshold is explicitly stated.
- Thresholded predictions classify a score equal to the selected threshold as positive. This matches the threshold semantics used by scikit-learn ROC curves.
- The positive class is the negative/adversarial class in Phase A, benchmark-incorrect responses in Phase B, and exfiltration in Phase C.
- Small constructed datasets are included directly under `data/`. TruthfulQA is public and is loaded from its public dataset source rather than redistributed.

The machine-readable configuration is `configs/reproducibility.yaml`.

## Phase A: Boundary mapping

### Core prompt-family evaluation

**Input:** `data/phase_a_boundary_prompts.csv`

The dataset contains 20 automotive prompt families and 20 negative prompt families. Each seed prompt is deterministically expanded into five surface variants: the original prompt, lowercase text, an appended detail request, a prefixed domain request, and a punctuation-stripped polite form. Family-disjoint evaluation prevents surface variants from the same seed from crossing the train/test boundary.

The prefixed variant uses `I'd like to know:` for automotive prompts and `Quick question:` for negative prompts. These class-specific prefixes are part of the recorded Phase A protocol.

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

The multi-signal score uses fixed proof-of-concept coefficients carried from an exploratory 50-iteration prototype run (held constant across the hold-out split and repeated ablation audit rather than refit per split):

- contradiction logit, weight `1.0` (normalized `0.137`)
- contradiction-minus-entailment gap, weight `0.45` (normalized `0.062`)
- question-answer cosine distance, weight `4.4` (normalized `0.603`)
- answer-to-question length ratio, weight `1.45` (normalized `0.199`)

Signal standardization is fitted using training questions only. Incorrect-answer scores are max-pooled at the question level.

The original Phase B run recorded the model identifiers but did not content-address the exact remote model snapshots and full software environment. A current revalidation reproduced the held-out precision, recall, F1, and confusion counts exactly, while the continuous AUC shifted slightly. The committed manifest therefore remains the canonical numerical record for Phase B, and the repository treats exact score-level replay across model or library revisions as a reproducibility limitation.

### Single hold-out

- base seed `42`
- 408 training questions
- 409 test questions
- raw pairwise feature scoring covers 2,036 training pairs and 2,074 test pairs
- incorrect candidate scores are max-pooled within each question before held-out classification
- pooled evaluation contains 816 training observations and 818 test observations
- held-out test balance is 409 benchmark-correct negatives and 409 pooled benchmark-incorrect positives
- threshold selected on training data by maximizing Youden J
- threshold frozen on test data
- at threshold `-0.2024061428888519`, the test confusion matrix is TP 342, FN 67, FP 196, TN 213

### Bounded repeated audit

- fixed 240-question subset selected from the public benchmark with base seed `42`
- 100 train/test re-splits
- 120 training and 120 test questions per split
- up to 10 listed incorrect answers per question by default (with caps of 3 and 1 evaluated as explicit ablation variants)
- split-bank seed `7042`
- evaluates nine variants: `full_multi_signal_maxpool`, `contradiction_only_maxpool`, `drop_length_ratio`, `drop_cosine_dist`, `drop_entail_gap`, `mean_pool_aggregation`, `top2_mean_aggregation`, `incorrect_cap_3`, and `incorrect_cap_1`
- reports mean AUC and mean F1 with empirical 5th to 95th percentile ranges across the 100 re-splits (descriptive re-split stability ranges, not confidence intervals)

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
- regex proxy is retained as a transparent discrete lexical baseline

For the regex proxy, the graded match-count score yields mean AUC 0.970 across the 200 resamples. The training-selected threshold is 1.0, and ROC-consistent thresholding gives mean precision 1.000, recall 0.940, and F1 0.967.

**Detailed output:** `results/phase_c_operating_points.json`

### Paraphrastic stress

**Input:** `data/phase_c_paraphrastic_templates.csv`

- 12 templates per class
- 8 training templates and 4 test templates per class
- 200 resamples
- derived seed `3042`

The historical lexical comparator for this condition is a character 3-to-5 gram logistic regression, not the word-plus-character hybrid used for the original and marker-scrubbed comparisons. The regex proxy uses the same training-selected Youden threshold rule as the original regex comparison.

With ROC-consistent threshold semantics, the paraphrastic regex proxy has mean AUC 0.735, precision 0.873, recall 0.553, and F1 0.655. DATM retains higher mean AUC 0.953 and F1 0.855, but the corrected regex operating point is materially stronger than the previously reported strict-threshold result.

### Marker-scrubbed stress

**Input:** `data/phase_c_marker_scrubbed_templates.csv`

- 12 templates per class
- 8 training templates and 4 test templates per class
- 200 balanced resamples
- derived seed `9042`

The marker-scrubbed condition removes obvious lexical markers that made the original synthetic task easier. The detailed operating-point output is in `results/phase_c_operating_points.json`.

## Phase D: controlled Gaussian sensitivity analysis

Script: `experiments/phase_d_sensitivity.py`

Configuration:

* base seed `42`
* embedding dimension `384`
* 300 Monte Carlo runs for each condition and mean-shift level
* 500 benign and 200 adversarial synthetic samples per run
* benign Gaussian scale `0.10`
* equal-variance adversarial scale `0.10`
* unequal-variance comparator adversarial scale `0.15`
* mean-shift levels `0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50`
* candidate threshold grid from `0.05` to less than `0.50` in increments of `0.01`
* dimension-normalized Euclidean distance
* common random numbers across the paired variance conditions
* ROC AUC as the primary threshold-independent read-out
* TPR, FPR, precision, recall, F1, and the mean-F1-at-least-0.95 threshold interval as secondary read-outs

The equal-variance condition isolates mean displacement. The unequal-variance comparator measures how much the scale difference contributes to separation. At zero mean shift, the equal-variance condition has mean AUC 0.516, while the unequal-variance comparator has mean AUC 1.000. Under equal variance, mean AUC is 0.988 at mean shift 0.05 and 1.000 from mean shift 0.10 onward. No equal-variance threshold reaches mean F1 at least 0.95 at mean shifts 0 or 0.05; the first qualifying interval is rho 0.11 to 0.13 at mean shift 0.10.

Canonical outputs:

* `results/phase_d_auc_summary.csv`
* `results/phase_d_threshold_summary.csv`
* `results/phase_d_threshold_bands.csv`
* `results/phase_d_condition_contrasts.csv`
* `results/phase_d_sensitivity.json`

The analysis was validated with Python 3.12.10, NumPy 2.3.5, PyTorch 2.6.0 with CUDA 12.4 build metadata, and scikit-learn 1.7.2, using CPU execution for Phase D. The committed global requirements remain the shared environment definition for all phases.

The simulation remains a stylized sensitivity analysis. High-dimensional radial concentration makes modest per-coordinate shifts highly separable in this geometry, so the result is not an empirical deployment benchmark.

## Provenance map

* Retrospective case strand: structured five-field extraction worksheet in `cases/case_extraction_worksheet.md` and `cases/case_extraction_worksheet.json`, validated by `tools/test_case_worksheet.py` (strictly bounded to 1 primary legal ruling for Air Canada, 1 secondary Business Insider article for Chevrolet, and 2 secondary Register articles for Samsung; no primary dealership transcript or internal Samsung memo is held or cited).
* Phase A family resampling: input `data/phase_a_boundary_prompts.csv`, script `experiments/phase_a_boundary.py`, seed 42, training-split Youden J, canonical output `results/paper_results_manifest.json`.
* Phase A multi-domain stress: input `data/phase_a_multidomain_prompts.csv`, script `experiments/phase_a_boundary.py`, seed 4042, training-split Youden J, canonical output `results/paper_results_manifest.json`.
* Phase A in-domain stress: input `data/phase_a_in_domain_stress.csv`, script `experiments/phase_a_boundary.py`, seed 10042, training-split Youden J, canonical output `results/paper_results_manifest.json`.
* Phase B hold-out: public TruthfulQA, script `experiments/phase_b_response_integrity.py`, seed 42, training-split Youden J, canonical output `results/paper_results_manifest.json`.
* Phase B repeated audit: public TruthfulQA, script `experiments/phase_b_response_integrity.py`, split-bank seed 7042, training-split Youden J, canonical output `results/paper_results_manifest.json`.
* Phase C original comparator: input `data/phase_c_original_templates.csv`, script `experiments/phase_c_exfiltration.py`, seed 8042, DATM training-split Youden J and hybrid threshold 0.5, canonical output `results/phase_c_operating_points.json`.
* Phase C paraphrastic stress: input `data/phase_c_paraphrastic_templates.csv`, script `experiments/phase_c_stress.py`, seed 3042, canonical output `results/paper_results_manifest.json`.
* Phase C marker-scrubbed stress: input `data/phase_c_marker_scrubbed_templates.csv`, script `experiments/phase_c_stress.py`, seed 9042, canonical output `results/phase_c_operating_points.json`.
* Phase D sensitivity: synthetic Gaussian draws, script `experiments/phase_d_sensitivity.py`, seed 42, explicit rho grid, canonical outputs `results/phase_d_auc_summary.csv` and `results/phase_d_threshold_bands.csv`.

## Scope and interpretation

The repository makes the reported computational analyses inspectable and reproducible and exposes the structured five-field manual extraction worksheet (`cases/case_extraction_worksheet.md` and `cases/case_extraction_worksheet.json`) used for the retrospective case strand. Only Air Canada rests on a primary legal ruling; Chevrolet and Samsung rely solely on secondary press reports, and the worksheet does not include primary corporate transcripts, internal memos, or private incident datasets. The bounded proxy and synthetic tasks do not establish enterprise deployment effectiveness, external validity, or calibrated production risk estimates.
