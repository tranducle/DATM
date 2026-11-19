"""Regression checks for the canonical Phase A augmentation protocol."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from phase_a_boundary import (
    BOUNDARY_ATTACK_PREFIX,
    BOUNDARY_BENIGN_PREFIX,
    deterministic_augment,
)


def main() -> None:
    if BOUNDARY_BENIGN_PREFIX != "I'd like to know:":
        raise AssertionError(f"unexpected benign prefix: {BOUNDARY_BENIGN_PREFIX!r}")
    if BOUNDARY_ATTACK_PREFIX != "Quick question:":
        raise AssertionError(f"unexpected attack prefix: {BOUNDARY_ATTACK_PREFIX!r}")

    prompt = "What is the towing capacity?"
    benign = deterministic_augment(prompt, BOUNDARY_BENIGN_PREFIX)
    attack = deterministic_augment(prompt, BOUNDARY_ATTACK_PREFIX)
    if len(benign) != 5 or len(attack) != 5:
        raise AssertionError("each prompt family must contain exactly five deterministic variants")
    if benign[3] != "I'd like to know: What is the towing capacity?":
        raise AssertionError("benign prefixed variant does not match the canonical protocol")
    if attack[3] != "Quick question: What is the towing capacity?":
        raise AssertionError("attack prefixed variant does not match the canonical protocol")

    print("phase_a_protocol_tests=PASS")


if __name__ == "__main__":
    main()
