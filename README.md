# Dynamic Algorithmic Threat Modeling (DATM)

This repository contains the computational reproducibility materials for a study of Dynamic Algorithmic Threat Modeling (DATM) for enterprise generative-AI governance. It is limited to experimental code, constructed prompt and template sets, configuration, canonical numerical outputs, and the source for the Phase D sensitivity figure.

The computational evidence is bounded. It covers controlled domain-boundary tests, a TruthfulQA response-integrity proxy, synthetic exfiltration-template evaluations, and a controlled Gaussian sensitivity analysis. These materials reproduce the reported computational illustrations. They are not a production security system or evidence of enterprise deployment readiness.

## Repository structure

```text
.
| README.md
| REPRODUCIBILITY.md
| requirements.txt
| configs/
| data/
| experiments/
| figures/
| results/
| tools/
```

The `data/` directory contains synthetic study inputs. The `experiments/` directory contains one script per computational phase. Canonical numerical outputs are under `results/`. The `figures/` directory contains the standalone TikZ source and exact CSV inputs for the Phase D sensitivity figure.

## Experimental phases

Phase A evaluates a fixed benign-centroid distance score using constructed automotive and negative prompt families, followed by multi-domain benign-space and in-domain malicious stress tests.

Phase B uses the public TruthfulQA validation split as a response-integrity proxy. The analysis compares a fixed multi-signal score with contradiction-only scoring under question-level hold-out and repeated bounded audits.

Phase C evaluates constructed benign and exfiltration templates with template-disjoint resampling. It includes a hybrid lexical comparator, a paraphrastic stress set, and a marker-scrubbed stress set.

Phase D separates two factors in a synthetic Gaussian model. The primary condition uses equal benign and adversarial scales of 0.10. An unequal-variance comparator uses adversarial scale 0.15. Both conditions share the same standard-normal draws within each paired run. ROC AUC is the primary read-out, while threshold-level metrics describe the range of candidate rho values for which mean F1 reaches at least 0.95.

## Quick start

Create a Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run individual phases from the repository root:

```bash
python experiments/phase_a_boundary.py
python experiments/phase_b_response_integrity.py
python experiments/phase_c_exfiltration.py
python experiments/phase_c_stress.py
python experiments/phase_d_sensitivity.py
```

Phase B downloads TruthfulQA through the Hugging Face `datasets` interface. The NLI and embedding checkpoints are obtained from their public model repositories unless already cached.

Run the repository checks with:

```bash
python tools/test_phase_b_reporting.py
python tools/test_phase_d_metrics.py
python tools/validate_repository.py
```

## Reproducibility conventions

The base random seed is 42. Derived seeds, split sizes, model identifiers, score weights, repetition counts, and threshold rules are recorded in `configs/reproducibility.yaml` and described in `REPRODUCIBILITY.md`.

Phase D uses common random numbers across the two variance conditions. For a fixed mean shift and run index, the same benign and adversarial standard-normal draws are reused before applying the condition-specific adversarial scale. This pairing isolates the effect of scale while reducing Monte Carlo noise in the condition contrast.

Rerunning Phase D writes regenerated outputs under `results/generated/`. The committed files directly under `results/` are the canonical summaries used for the reported analysis.

## Data handling

All prompt and template CSV files in `data/` are synthetic study inputs. Strings that resemble secrets, credentials, or internal identifiers are fictitious examples created for the experiments. TruthfulQA is not redistributed.

## Canonical results

`results/paper_results_manifest.json` is the compact numerical manifest for the study. Phase D uses `results/phase_d_auc_summary.csv`, `results/phase_d_threshold_summary.csv`, `results/phase_d_threshold_bands.csv`, `results/phase_d_condition_contrasts.csv`, and `results/phase_d_sensitivity.json`.

The Phase D figure source is `figures/phase_d_sensitivity.tex`. It reads the two CSV files in the same directory and can be compiled independently with LuaLaTeX.

See `REPRODUCIBILITY.md` for the provenance map linking inputs, scripts, seeds, threshold rules, and result artifacts.
