#!/usr/bin/env python3
"""End-to-end pipeline latency scaffold (mapping + filters only without ROS)."""
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

from vision_arm_control.coordinate_mapping import CameraIntrinsics, map_wrist_end_effector_position  # noqa: E402
from vision_arm_control.trajectory_filter import LowPassFilter3D, VelocityLimiter3D  # noqa: E402
from vision_arm_control.workspace_limits import DEFAULT_WORKSPACE, validate_or_clamp_position  # noqa: E402


def main(n: int = 2000) -> dict:
    K = CameraIntrinsics(fx=600, fy=600, cx=320, cy=240, width=640, height=480)
    T = np.eye(4)
    lpf = LowPassFilter3D(0.3)
    vlim = VelocityLimiter3D(0.2)
    times = []
    rejects = 0
    for i in range(n):
        t0 = time.perf_counter()
        target = map_wrist_end_effector_position(
            wrist_norm_xy=(0.5 + 0.2 * np.sin(i * 0.05), 0.5),
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
        pos, ok, _ = validate_or_clamp_position(target, DEFAULT_WORKSPACE, mode="reject")
        if not ok or pos is None:
            rejects += 1
        else:
            filt = lpf.update(pos)
            vlim.update(filt, dt=0.02)
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "metric": "pipeline_cpu_stage_latency_ms",
        "n": n,
        "mean_ms": statistics.mean(times),
        "p50_ms": statistics.median(times),
        "reject_rate": rejects / float(n),
        "hardware": platform.processor() or platform.machine(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python benchmarks/benchmark_latency.py",
        "notes": (
            "CPU stages only (map+filter+workspace). Full ROS perception+control "
            "and physical Franka latency: Not yet measured."
        ),
    }


if __name__ == "__main__":
    out = main()
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / "latency_benchmark.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
