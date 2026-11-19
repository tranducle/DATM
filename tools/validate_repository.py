#!/usr/bin/env python3
"""Validate the DATM reproducibility repository structure and publication scope."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md",
    "REPRODUCIBILITY.md",
    "requirements.txt",
    "configs/reproducibility.yaml",
    "data/phase_a_boundary_prompts.csv",
    "data/phase_a_multidomain_prompts.csv",
    "data/phase_a_in_domain_stress.csv",
    "data/phase_c_original_templates.csv",
    "data/phase_c_paraphrastic_templates.csv",
    "data/phase_c_marker_scrubbed_templates.csv",
    "experiments/common.py",
    "experiments/phase_a_boundary.py",
    "experiments/phase_b_response_integrity.py",
    "experiments/phase_c_exfiltration.py",
    "experiments/phase_c_stress.py",
    "experiments/phase_d_sensitivity.py",
    "tools/test_phase_a_protocol.py",
    "tools/test_phase_b_reporting.py",
    "tools/test_phase_c_regex.py",
    "tools/test_threshold_semantics.py",
    "tools/test_phase_d_metrics.py",
    "results/paper_results_manifest.json",
    "results/phase_c_operating_points.json",
    "results/phase_d_auc_summary.csv",
    "results/phase_d_threshold_summary.csv",
    "results/phase_d_threshold_bands.csv",
    "results/phase_d_condition_contrasts.csv",
    "results/phase_d_sensitivity.json",
    "figures/phase_d_sensitivity.tex",
    "figures/phase_d_equal_variance.csv",
    "figures/phase_d_unequal_variance.csv",
]

FORBIDDEN_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".svg", ".docx", ".pptx"}
ALLOWED_TEX = {"figures/phase_d_sensitivity.tex"}


def main() -> int:
    findings: list[str] = []

    for rel in REQUIRED:
        if not (ROOT / rel).is_file():
            findings.append(f"missing required file: {rel}")

    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() == ".tex" and rel not in ALLOWED_TEX:
            findings.append(f"forbidden TeX artifact: {rel}")
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name == ".env":
            findings.append(f"forbidden artifact: {rel}")

    if findings:
        print("Repository validation failed:")
        for finding in findings:
            print(f"  - {finding}")
        return 1

    print(f"Repository validation passed: {len(REQUIRED)} required artifacts present; no prohibited binary or document artifacts found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
