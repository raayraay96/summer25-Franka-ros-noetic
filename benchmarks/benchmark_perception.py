#!/usr/bin/env python3
"""Perception benchmark scaffold.

Runs only if MediaPipe (and optionally PyTorch models) are available.
Does not invent metrics when dependencies are missing.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def try_mediapipe(n_frames: int = 50) -> dict:
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        return {
            "metric": "perception_latency_ms",
            "status": "Not yet measured",
            "reason": f"dependency_missing: {exc}",
            "date_utc": datetime.now(timezone.utc).isoformat(),
        }

    pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    # Synthetic frames (not a real camera dataset)
    times = []
    detected = 0
    for i in range(n_frames):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a simple stick figure-ish blob — detection not guaranteed
        cv2.circle(frame, (320, 200), 30, (255, 255, 255), -1)
        t0 = time.perf_counter()
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        times.append((time.perf_counter() - t0) * 1000.0)
        if res.pose_landmarks:
            detected += 1
    pose.close()
    return {
        "metric": "perception_latency_ms",
        "status": "measured_synthetic_frames",
        "n_frames": n_frames,
        "mean_ms": float(np.mean(times)),
        "p50_ms": float(np.median(times)),
        "detection_rate": detected / float(n_frames),
        "input": "synthetic_blank_with_circle",
        "hardware": platform.processor() or platform.machine(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "notes": (
            "Synthetic frames; detection_rate is not representative of real human video. "
            "Physical Franka / real camera metrics: Not yet measured."
        ),
        "command": "python benchmarks/benchmark_perception.py",
    }


def main() -> None:
    out = try_mediapipe()
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    path = results / "perception_benchmark.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
