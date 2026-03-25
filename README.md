# Dynamic Algorithmic Threat Modeling (DATM)

This repository contains the empirical validation codebase for the research paper: **Resolving Generative Dissonance: A Dynamic Algorithmic Threat Modeling (DATM) Framework for Probabilistic Enterprise Architectures**.

Due to the double-blind review process, author and institutional information have been temporarily removed.

## Repository Contents

- `datm_poc.py`: The Proof-of-Concept implementation of the DATM framework.
- `phase_ac_upgraded.py`: Simulates Phase A (Baseline Vulnerability Assessment) and Phase C (Post-DATM Mitigation Verification) on enterprise cases.
- `phase_b_experiment.py`: Simulates Phase B, demonstrating automated topological exploitation on the TruthfulQA dataset.
- `phase_d_experiment.py`: Simulates Phase D, conducting extensive algorithmic stress testing.
- `experiment_config.yaml` / `experiment_config_d.yaml`: Configuration files controlling iteration scaling, batch sizes, and model parameters.
- `generate_all_figs.py` / `generate_paper_figs.py`: Code for generating the empirical learning curves and visualization figures presented in the manuscript.
- `results/`: Contains the empirical output logs, metrics, and TSV files tracking the security efficacy over time.

## Requirements
- Python 3.9+
- The exact dependencies will be specified in a `requirements.txt` soon.

## Usage
To replicate the core validation experiments:
```bash
python phase_b_experiment.py
python phase_d_experiment.py
```
To generate the visualizations:
```bash
python generate_paper_figs.py
```

## Note
This is not a production-grade defense system. The code is provided solely to replicate the algorithmic experiments verifying the DATM framework's capability to bound GenAI topological vulnerabilities.
