"""Ensure committed metrics files do not invent physical Franka results."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def test_metrics_json_marks_physical_unmeasured():
    data = json.loads((RESULTS / "metrics.json").read_text())
    phys = data.get("physical_franka") or data.get("cpu_benchmarks", {})
    # Support both layouts produced over time
    if "physical_franka" in data:
        for key, val in data["physical_franka"].items():
            assert "not yet measured" in str(val).lower(), (key, val)
    else:
        assert "physical" in json.dumps(data).lower() or "not yet measured" in json.dumps(data).lower()


def test_metrics_csv_no_fake_franka_numbers():
    text = (RESULTS / "metrics.csv").read_text().lower()
    # Must mention unmeasured hardware or not claim franka success rates as numbers
    assert "not yet measured" in text or "physical_franka" in text


def test_benchmark_scripts_document_limitations():
    readme = (ROOT / "benchmarks" / "README.md").read_text().lower()
    assert "not yet measured" in readme or "without franka" in readme
