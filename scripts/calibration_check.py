#!/usr/bin/env python3
"""Validate camera intrinsics / extrinsics configuration files.

Does not perform a full calibration procedure; it checks file integrity and
prints whether values are labeled example vs measured.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=Path("src/vision_arm_control/config/camera_intrinsics.yaml"),
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("src/vision_arm_control/config/mapping.yaml"),
    )
    args = parser.parse_args()

    errors = 0
    for path in (args.intrinsics, args.mapping):
        if not path.is_file():
            print(f"[FAIL] missing {path}")
            errors += 1
            continue
        data = yaml.safe_load(path.read_text())
        print(f"[OK] parsed {path}")
        print(yaml.safe_dump(data, sort_keys=False))

    if args.intrinsics.is_file():
        status = yaml.safe_load(args.intrinsics.read_text())["camera_intrinsics"].get("status")
        if status != "measured":
            print(
                f"[WARN] intrinsics status={status!r} — treat as EXAMPLE, not measured calibration."
            )
    if args.mapping.is_file():
        status = yaml.safe_load(args.mapping.read_text())["frames"].get("camera_to_base_status")
        if status != "measured":
            print(
                f"[WARN] camera_to_base status={status!r} — EXAMPLE extrinsics only."
            )

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
