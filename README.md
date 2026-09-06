# Dynamic Algorithmic Threat Modeling (DATM)

This repository contains the computational reproducibility materials for a study of **Dynamic Algorithmic Threat Modeling (DATM)** for enterprise generative-AI governance. The repository is intentionally limited to experimental code, constructed prompt/template sets, configuration, and canonical numerical outputs.

The computational evidence is bounded. It includes controlled domain-boundary tests, a TruthfulQA response-integrity proxy, synthetic exfiltration-template evaluations, and a stylized Monte Carlo sensitivity analysis. These materials support reproducibility of the reported computational illustrations; they do not constitute a production security system or evidence of enterprise deployment readiness.

## Repository structure

```text
.
├── README.md
├── REPRODUCIBILITY.md
├── requirements.txt
├── configs/
│   └── reproducibility.yaml
├── data/
│   ├── phase_a_boundary_prompts.csv
│   ├── phase_a_in_domain_stress.csv
│   ├── phase_a_multidomain_prompts.csv
│   ├── phase_c_original_templates.csv
│   ├── phase_c_paraphrastic_templates.csv
│   └── phase_c_marker_scrubbed_templates.csv
├── experiments/
│   ├── common.py
│   ├── phase_a_boundary.py
│   ├── phase_b_response_integrity.py
│   ├── phase_c_exfiltration.py
│   ├── phase_c_stress.py
│   └── phase_d_sensitivity.py
└── results/
    ├── paper_results_manifest.json
    ├── phase_c_operating_points.json
    ├── phase_c_matched_thresholds.json
    └── phase_d_reference_sweep.csv
```

## Experimental phases

**Phase A: Boundary mapping.** Uses constructed automotive and negative prompt families to evaluate a fixed benign-centroid distance score, followed by multi-domain benign-space and in-domain malicious stress tests.

**Phase B: Response-integrity proxy.** Uses the public TruthfulQA validation split. Question-answer pairs are scored with contradiction, contradiction-minus-entailment, cosine-distance, and length-ratio signals. The main reproducibility script reports a question-level hold-out and a bounded repeated audit that compares the multi-signal score with contradiction-only scoring.

**Phase C: Exfiltration-template evaluation.** Uses constructed benign and exfiltration templates. Evaluation is template-disjoint. The repository includes the original template task, a stronger hybrid lexical comparator, a paraphrastic stress set, and a marker-scrubbed stress set.

**Phase D: Sensitivity analysis.** Uses synthetic Gaussian embeddings to examine how detection behavior changes across attack-shift levels and candidate boundary thresholds.

## Quick start

Create a Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run individual phases from the repository root:

```bash
python experiments/phase_a_boundary.py
python experiments/phase_b_response_integrity.py
python experiments/phase_c_exfiltration.py
python experiments/phase_c_stress.py
python experiments/phase_d_sensitivity.py
```

Phase B downloads TruthfulQA through the Hugging Face `datasets` interface. The NLI and embedding checkpoints are also obtained from their public model repositories unless they are already cached.

Validate repository structure and publication scope with:

```bash
python tools/validate_repository.py
```

## Reproducibility conventions

The base random seed is **42**. Derived seeds, split sizes, model identifiers, score weights, repetition counts, and threshold-selection rules are recorded in `configs/reproducibility.yaml` and described in detail in `REPRODUCIBILITY.md`.

Threshold-dependent metrics are evaluated using thresholds selected only from training data unless a fixed threshold is explicitly specified. In Phase C, the hybrid lexical comparator uses a fixed probability threshold of 0.5 in the primary operating-point comparison. A separate matched-threshold diagnostic is retained as an auxiliary result and is not substituted for the primary comparison.

## Data handling

All prompt and template CSV files in `data/` are synthetic study inputs. Strings that resemble secrets, credentials, or internal identifiers are fictitious examples created for the experiments. TruthfulQA is not redistributed in this repository.

## Canonical results

`results/paper_results_manifest.json` is the compact numerical manifest for the study. `results/phase_c_operating_points.json` contains the detailed operating-point statistics for the original and marker-scrubbed Phase C evaluations. `results/phase_d_reference_sweep.csv` preserves the reference Monte Carlo sweep used for the reported sensitivity analysis.

See `REPRODUCIBILITY.md` for the provenance map linking inputs, scripts, seeds, threshold rules, and result artifacts.
