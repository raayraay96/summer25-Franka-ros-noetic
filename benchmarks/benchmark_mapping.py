#!/usr/bin/env python3
"""Benchmark pure mapping throughput (no ROS, no GPU)."""
from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vision_arm_control" / "src"))

from vision_arm_control.coordinate_mapping import (  # noqa: E402
    CameraIntrinsics,
    map_wrist_end_effector_position,
)


def main(n: int = 5000) -> dict:
    K = CameraIntrinsics(fx=600, fy=600, cx=320, cy=240, width=640, height=480)
    T = np.eye(4)
    times = []
    for i in range(n):
        wx = 0.5 + 0.1 * np.sin(i * 0.01)
        t0 = time.perf_counter()
        map_wrist_end_effector_position(
            wrist_norm_xy=(wx, 0.5),
            image_size=(640, 480),
            intrinsics=K,
            T_base_camera=T,
            depth_m=None,
            relative_depth=None,
            relative_depth_scale=1.0,
            relative_depth_offset=0.8,
            use_metric_depth=False,
            shoulder_norm_xy=(0.45, 0.35),
            mapping_mode="shoulder_relative",
        )
        times.append((time.perf_counter() - t0) * 1000.0)
    result = {
        "metric": "mapping_latency_ms",
        "n": n,
        "mean_ms": statistics.mean(times),
        "p50_ms": statistics.median(times),
        "p95_ms": sorted(times)[int(0.95 * (n - 1))],
        "max_ms": max(times),
        "hardware": platform.processor() or platform.machine(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python benchmarks/benchmark_mapping.py",
        "notes": "CPU-only coordinate mapping; not full perception pipeline",
    }
    return result


if __name__ == "__main__":
    out = main()
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    path = results_dir / "mapping_benchmark.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"Wrote {path}")
