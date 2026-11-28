#!/usr/bin/env python3
"""Regression checks for the retrospective five-field case-extraction worksheet."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "cases" / "case_extraction_worksheet.json"
MD_PATH = ROOT / "cases" / "case_extraction_worksheet.md"

EXPECTED_FIELDS = [
    "actor_and_access_path",
    "prompt_or_payload_characteristics",
    "documented_organizational_consequences",
    "intact_deterministic_controls",
    "datm_relevant_runtime_behavior",
]

EXPECTED_CASES = {
    "air_canada": {
        "evidence_tier": "primary_legal_record",
        "primary_material_available": True,
        "primary_source_count": 1,
        "secondary_source_count": 0,
        "bib_keys": ["moffatt2024aircanada"],
    },
    "chevrolet": {
        "evidence_tier": "secondary_news_report",
        "primary_material_available": False,
        "primary_source_count": 0,
        "secondary_source_count": 1,
        "bib_keys": ["coldewey2023chevy"],
    },
    "samsung": {
        "evidence_tier": "secondary_trade_press",
        "primary_material_available": False,
        "primary_source_count": 0,
        "secondary_source_count": 2,
        "bib_keys": ["dobberstein2023samsung", "karabus2023samsungban"],
    },
}


def main() -> int:
    assert JSON_PATH.is_file(), f"Missing {JSON_PATH}"
    assert MD_PATH.is_file(), f"Missing {MD_PATH}"

    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    md_text = MD_PATH.read_text(encoding="utf-8")

    assert data["required_extraction_fields"] == EXPECTED_FIELDS

    summary = data["corpus_evidence_summary"]
    assert summary["total_cases"] == 3
    assert summary["cases_with_primary_material"] == 1
    assert summary["cases_with_secondary_reporting_only"] == 2
    assert summary["total_cited_sources"] == 4
    assert summary["total_primary_sources"] == 1
    assert summary["total_secondary_sources"] == 3

    cases = data["cases"]
    assert len(cases) == 3
    seen = {}
    for entry in cases:
        cid = entry["case_id"]
        assert cid in EXPECTED_CASES, f"Unexpected case_id: {cid}"
        seen[cid] = entry
        exp = EXPECTED_CASES[cid]
        assert entry["evidence_tier"] == exp["evidence_tier"]
        assert entry["primary_material_available"] is exp["primary_material_available"]
        assert entry["primary_source_count"] == exp["primary_source_count"]
        assert entry["secondary_source_count"] == exp["secondary_source_count"]

        bib_keys = [src["bib_key"] for src in entry["cited_sources"]]
        assert bib_keys == exp["bib_keys"], f"{cid}: expected {exp['bib_keys']}, got {bib_keys}"
        assert len(entry["cited_sources"]) == exp["primary_source_count"] + exp["secondary_source_count"]
        for src in entry["cited_sources"]:
            assert src["url"].startswith("https://"), f"Non-HTTPS source URL in {cid}"
            assert src["bib_key"] in md_text, f"{src['bib_key']} missing from Markdown worksheet"

        extracted = entry["extracted_fields"]
        assert list(extracted.keys()) == EXPECTED_FIELDS
        for field_name in EXPECTED_FIELDS:
            val = extracted[field_name]
            assert isinstance(val, str) and len(val.strip()) >= 25, f"Empty/short field {cid}.{field_name}"

        assert len(entry["unavailable_primary_artifacts"]) >= 1
        assert len(entry["provenance_limit"].strip()) >= 30

    assert set(seen.keys()) == set(EXPECTED_CASES.keys())

    # Confirm missing primary artifacts are explicitly acknowledged for secondary-only cases
    chevy_missing = " ".join(seen["chevrolet"]["unavailable_primary_artifacts"]).lower()
    samsung_missing = " ".join(seen["samsung"]["unavailable_primary_artifacts"]).lower()
    assert "transcript" in chevy_missing
    assert "internal" in samsung_missing and "memo" in samsung_missing

    # Confirm neither artifact uses prohibited dashes or overclaimed provenance labels
    double_hyphen = "-" * 2
    mdash_ent = "&" + "mdash;"
    ndash_ent = "&" + "ndash;"
    prohibited_dash = re.compile(rf"[\u2013\u2014]|{double_hyphen}|{mdash_ent}|{ndash_ent}")
    for path in (JSON_PATH, MD_PATH):
        text = path.read_text(encoding="utf-8")
        assert not prohibited_dash.search(text), f"Prohibited dash sequence in {path.name}"
        assert "auditable chain of evidence" not in text.lower(), f"Overclaimed chain phrase in {path.name}"
        assert "forensic reconstruction" not in text.lower(), f"Overclaimed forensic phrase in {path.name}"
        assert "v(t)" not in text.lower(), f"Retired Eq. 1 symbol V(t) present in {path.name}"
        assert "equation 2" not in text.lower(), f"Stale Equation 2 reference present in {path.name}"

    print("Case extraction worksheet checks passed: 3 cases, 5 fields each, 1 primary / 2 secondary-only verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
