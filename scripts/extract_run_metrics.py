#!/usr/bin/env python3
"""Extract run-level metrics from a diagnostics JSONL produced on Scholar.

Reads a JSONL file (one diagnostics record per line, as written by the ROS 2
teleop pipeline node) and writes a summary JSON using the same pure-Python
aggregator that is unit-tested in CI.

Usage:
  python scripts/extract_run_metrics.py <diagnostics.jsonl> [--output summary.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vision_arm_control" / "src"))

from vision_arm_control.pipeline_diagnostics import summarize_records  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)
    records = []
    for line in args.jsonl.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    summary = summarize_records(records)
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"wrote {args.output}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
