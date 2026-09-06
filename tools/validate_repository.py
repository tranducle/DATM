#!/usr/bin/env python3
"""Validate the DATM reproducibility repository structure and publication scope."""

from __future__ import annotations

import re
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
    "results/paper_results_manifest.json",
    "results/phase_c_operating_points.json",
    "results/phase_c_matched_thresholds.json",
    "results/phase_d_reference_sweep.csv",
]

FORBIDDEN_SUFFIXES = {".tex", ".pdf", ".png", ".jpg", ".jpeg", ".svg", ".docx", ".pptx"}
TEXT_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".csv", ".json", ".txt"}
FORBIDDEN_PATTERNS = [
    r"/" + r"Users/",
    r"review" + r"er[ _-]*[0-9]",
    r"local" + r" use",
    r"autonomous" + r" optimization",
    r"agent" + r" workflow",
    r"chain" + r" of thought",
    r"SUBMITTED/" + r"SUBMIT_TO_ITM",
    r"sk-proj-[A-Za-z0-9_-]{20,}",
    r"ghp_[A-Za-z0-9]{20,}",
    r"AKIA[0-9A-Z]{16}",
]


def main() -> int:
    findings: list[str] = []

    for rel in REQUIRED:
        if not (ROOT / rel).is_file():
            findings.append(f"missing required file: {rel}")

    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(ROOT)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name == ".env":
            findings.append(f"forbidden artifact: {rel}")
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name == ".gitignore":
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in FORBIDDEN_PATTERNS:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    findings.append(f"forbidden pattern {pattern!r} in {rel}")

    if findings:
        print("Repository validation failed:")
        for finding in findings:
            print(f"  - {finding}")
        return 1

    print(f"Repository validation passed: {len(REQUIRED)} required artifacts present; no prohibited files or text patterns found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
