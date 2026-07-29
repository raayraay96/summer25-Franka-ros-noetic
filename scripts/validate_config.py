#!/usr/bin/env python3
"""Validate teleop YAML config(s) against the versioned schema (Phase 8).

Usage:
  python scripts/validate_config.py <config.yaml> [more.yaml ...]
  python scripts/validate_config.py            # validates all shipped fixtures

Exits non-zero with a useful message on the first invalid config.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vision_arm_control" / "src"))

from vision_arm_control.config_schema import ConfigError, validate_teleop_config  # noqa: E402

FIXTURES = ROOT / "src" / "vision_arm_control" / "config" / "fixtures"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    paths = [Path(p) for p in argv] or sorted(FIXTURES.glob("*.yaml"))
    if not paths:
        print("no configs to validate", file=sys.stderr)
        return 1
    for p in paths:
        cfg = yaml.safe_load(p.read_text())
        try:
            validate_teleop_config(cfg)
        except ConfigError as exc:
            print(f"[INVALID] {p}: {exc}", file=sys.stderr)
            return 1
        print(f"[ok] {p}")
    print(f"validated {len(paths)} config(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
