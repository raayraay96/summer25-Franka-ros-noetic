#!/usr/bin/env python3
"""Deterministic research benchmark suite for retargeting v1.1.

Compares shoulder_relative vs sew_orientation with reject / clamp / cbf_qp.

Usage:
  python3 benchmarks/benchmark_retargeting_v11.py \\
    --config benchmarks/config/v11.yaml \\
    --output results/v1.1

  # CI smoke:
  python3 benchmarks/benchmark_retargeting_v11.py --smoke --output results/v1.1
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vision_arm_control" / "src"))

from vision_arm_control.landmarks.confidence import ConfidenceGateConfig  # noqa: E402
from vision_arm_control.landmarks.schema import ArmLandmarks, LandmarkPoint  # noqa: E402
from vision_arm_control.pipeline import PipelineConfig, TeleopPipeline  # noqa: E402
from vision_arm_control.retargeting.base import RetargetingConfig  # noqa: E402
from vision_arm_control.safety_filters.base import (  # noqa: E402
    SafetyFilterConfig,
    SphericalObstacle,
)
from vision_arm_control.workspace_limits import AxisAlignedBounds  # noqa: E402


def _percentile(xs: Sequence[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = int(round(p * (len(s) - 1)))
    return float(s[max(0, min(k, len(s) - 1))])


def _lm_at(
    shoulder: Sequence[float],
    elbow: Sequence[float],
    wrist: Sequence[float],
    ts: float,
    conf: float = 1.0,
) -> ArmLandmarks:
    return ArmLandmarks(
        shoulder=LandmarkPoint(*shoulder, confidence=conf, name="right_shoulder"),
        elbow=LandmarkPoint(*elbow, confidence=conf, name="right_elbow"),
        wrist=LandmarkPoint(*wrist, confidence=conf, name="right_wrist"),
        timestamp=ts,
    )


def trajectory_samples(name: str, t: float, rng: np.random.Generator) -> Tuple[Optional[ArmLandmarks], dict]:
    """Synthetic SEW landmarks in image-normalized coordinates.

    Returns (landmarks_or_None, meta).
    """
    # Base shoulder fixed; animate elbow/wrist
    s = np.array([0.45, 0.35, 0.0])
    meta = {"traj": name, "t": t}

    if name == "horizontal_sweep":
        phase = 0.5 + 0.12 * math.sin(2 * math.pi * 0.25 * t)
        e = s + np.array([0.08, 0.05, 0.0])
        w = s + np.array([phase - 0.45 + 0.15, 0.12, 0.0])
        return _lm_at(s, e, w, t), meta

    if name == "vertical_sweep":
        phase = 0.45 + 0.10 * math.sin(2 * math.pi * 0.25 * t)
        e = s + np.array([0.06, 0.06, 0.0])
        w = s + np.array([0.12, phase - 0.35 + 0.10, 0.02])
        return _lm_at(s, e, w, t), meta

    if name == "circle":
        ang = 2 * math.pi * 0.3 * t
        e = s + np.array([0.07, 0.05, 0.0])
        w = s + np.array([0.12 * math.cos(ang), 0.10 * math.sin(ang) + 0.08, 0.0])
        return _lm_at(s, e, w, t), meta

    if name == "figure_eight":
        ang = 2 * math.pi * 0.25 * t
        e = s + np.array([0.07, 0.05, 0.0])
        w = s + np.array([0.12 * math.sin(ang), 0.08 * math.sin(2 * ang) + 0.08, 0.0])
        return _lm_at(s, e, w, t), meta

    if name == "step_input":
        offset = 0.18 if t >= 1.0 else 0.05
        e = s + np.array([0.06, 0.05, 0.0])
        w = s + np.array([offset, 0.10, 0.0])
        return _lm_at(s, e, w, t), meta

    if name == "near_full_extension":
        e = s + np.array([0.12, 0.02, 0.0])
        w = s + np.array([0.24, 0.03, 0.0])
        return _lm_at(s, e, w, t), meta

    if name == "noisy_landmarks":
        e = s + np.array([0.08, 0.05, 0.0]) + rng.normal(0, 0.005, size=3)
        w = s + np.array([0.14, 0.12, 0.0]) + rng.normal(0, 0.01, size=3)
        return _lm_at(s, e, w, t), meta

    if name == "one_second_dropout":
        if 1.0 <= t < 2.0:
            return None, {**meta, "dropout": True}
        e = s + np.array([0.08, 0.05, 0.0])
        w = s + np.array([0.14 + 0.05 * math.sin(t), 0.12, 0.0])
        return _lm_at(s, e, w, t), meta

    if name == "workspace_boundary_crossing":
        # Large horizontal swing that maps near/outside workspace
        phase = 0.5 + 0.45 * math.sin(2 * math.pi * 0.2 * t)
        e = s + np.array([0.10, 0.05, 0.0])
        w = np.array([phase, 0.50, 0.0])
        return _lm_at(s, e, w, t), meta

    if name == "obstacle_intersection":
        # Drive toward region that maps near demo sphere
        frac = min(t / 3.0, 1.0)
        e = s + np.array([0.08, 0.04, 0.0])
        w = s + np.array([0.05 + 0.20 * frac, 0.05, -0.05 * frac])
        return _lm_at(s, e, w, t), meta

    raise ValueError(f"unknown trajectory {name}")


def build_pipeline(retargeter: str, safety: str, cfg: dict) -> TeleopPipeline:
    ws = cfg["workspace"]
    bounds = AxisAlignedBounds(**{k: float(ws[k]) for k in ws})
    cbf = cfg.get("cbf_qp", {})
    obstacles = [
        SphericalObstacle(
            id=o["id"],
            center=o["center"],
            radius_m=float(o["radius_m"]),
            margin_m=float(o.get("margin_m", 0.0)),
        )
        for o in cbf.get("obstacles", [])
    ]
    safety_cfg = SafetyFilterConfig(
        mode=safety,
        workspace=bounds,
        workspace_margin_m=float(cbf.get("workspace_margin_m", 0.03)),
        dt=float(cfg["dt"]),
        alpha=float(cbf.get("alpha", 4.0)),
        max_cartesian_velocity_mps=float(cbf.get("max_cartesian_velocity_mps", 0.2)),
        stop_on_solver_failure=bool(cbf.get("stop_on_solver_failure", True)),
        obstacles=obstacles if safety == "cbf_qp" else [],
    )
    require_elbow = retargeter == "sew_orientation"
    return TeleopPipeline(
        PipelineConfig(
            retargeting=RetargetingConfig(
                name=retargeter,
                treat_z_as_image_relative=True,
            ),
            safety=safety_cfg,
            gate=ConfidenceGateConfig(
                hold_interval_sec=0.25,
                stale_timeout_sec=0.5,
                recovery_blend_sec=0.3,
                require_elbow=require_elbow,
            ),
            backend="dry_run",
            initial_position=tuple(cfg["initial_position"]),
        )
    )


def discrete_jerk_proxy(positions: List[np.ndarray], dt: float) -> float:
    """Mean ||Δa||/dt where a is discrete acceleration — jerk proxy."""
    if len(positions) < 4 or dt <= 0:
        return 0.0
    v = [(positions[i + 1] - positions[i]) / dt for i in range(len(positions) - 1)]
    a = [(v[i + 1] - v[i]) / dt for i in range(len(v) - 1)]
    j = [float(np.linalg.norm(a[i + 1] - a[i]) / dt) for i in range(len(a) - 1)]
    return float(statistics.mean(j)) if j else 0.0


def path_length(positions: List[np.ndarray]) -> float:
    if len(positions) < 2:
        return 0.0
    return float(sum(np.linalg.norm(positions[i + 1] - positions[i]) for i in range(len(positions) - 1)))


def orientation_alignment_error(directions: List[np.ndarray]) -> float:
    """Mean angle between consecutive lower-arm directions (rad)."""
    if len(directions) < 2:
        return 0.0
    errs = []
    for i in range(len(directions) - 1):
        a = directions[i]
        b = directions[i + 1]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            continue
        c = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        errs.append(math.acos(c))
    return float(statistics.mean(errs)) if errs else 0.0


def min_obstacle_clearance(positions: List[np.ndarray], obstacles: Sequence[SphericalObstacle]) -> float:
    if not positions or not obstacles:
        return float("nan")
    mins = []
    for p in positions:
        for o in obstacles:
            mins.append(float(np.linalg.norm(p - o.center_array()) - o.radius_m))
    return float(min(mins)) if mins else float("nan")


def run_one(
    retargeter: str,
    safety: str,
    traj: str,
    cfg: dict,
    rng: np.random.Generator,
) -> dict:
    pipe = build_pipeline(retargeter, safety, cfg)
    dt = float(cfg["dt"])
    duration = float(cfg["duration_sec"])
    n = int(round(duration / dt))
    map_lat = []
    saf_lat = []
    positions: List[np.ndarray] = []
    directions: List[np.ndarray] = []
    interventions = 0
    interv_mags = []
    solver_fail = 0
    dropped = 0
    violations = 0
    commanded = 0
    recovery_time = None
    saw_dropout = False
    reacq_t = None

    obstacles = []
    if safety == "cbf_qp":
        for o in cfg.get("cbf_qp", {}).get("obstacles", []):
            obstacles.append(
                SphericalObstacle(
                    id=o["id"],
                    center=o["center"],
                    radius_m=float(o["radius_m"]),
                    margin_m=float(o.get("margin_m", 0.0)),
                )
            )

    for i in range(n):
        t = i * dt
        lm, meta = trajectory_samples(traj, t, rng)
        if meta.get("dropout"):
            saw_dropout = True
        t0 = time.perf_counter()
        # retarget timing
        if lm is not None:
            rt0 = time.perf_counter()
            _ = pipe.retargeter.retarget(lm)
            map_lat.append((time.perf_counter() - rt0) * 1000.0)
        res = pipe.step(lm, now=t)
        step_ms = (time.perf_counter() - t0) * 1000.0
        if res.safety is not None:
            saf_lat.append(res.safety.compute_time_s * 1000.0)
            if res.safety.intervened:
                interventions += 1
                interv_mags.append(res.safety.intervention_magnitude)
            if "solver_failure" in res.safety.reason:
                solver_fail += 1
            violations += int(res.safety.constraint_violations)
        if not res.commanded:
            dropped += 1
        else:
            commanded += 1
            if res.safety and res.safety.position is not None:
                positions.append(res.safety.position.copy())
            elif res.gated.position is not None:
                positions.append(res.gated.position.copy())
            if res.gated.direction is not None:
                directions.append(np.asarray(res.gated.direction, dtype=np.float64))
            if saw_dropout and reacq_t is None and lm is not None:
                reacq_t = t
                recovery_time = 0.0
            elif reacq_t is not None and recovery_time == 0.0:
                recovery_time = t - reacq_t

    row = {
        "retargeter": retargeter,
        "safety": safety,
        "trajectory": traj,
        "n_steps": n,
        "mapping_latency_mean_ms": statistics.mean(map_lat) if map_lat else float("nan"),
        "mapping_latency_median_ms": statistics.median(map_lat) if map_lat else float("nan"),
        "mapping_latency_p95_ms": _percentile(map_lat, 0.95) if map_lat else float("nan"),
        "mapping_latency_p99_ms": _percentile(map_lat, 0.99) if map_lat else float("nan"),
        "mapping_latency_max_ms": max(map_lat) if map_lat else float("nan"),
        "safety_latency_mean_ms": statistics.mean(saf_lat) if saf_lat else float("nan"),
        "safety_latency_p95_ms": _percentile(saf_lat, 0.95) if saf_lat else float("nan"),
        "orientation_alignment_error_rad": orientation_alignment_error(directions),
        "ee_path_length_m": path_length(positions),
        "path_jerk_proxy": discrete_jerk_proxy(positions, dt),
        "constraint_violation_count": violations,
        "intervention_count": interventions,
        "intervention_pct": 100.0 * interventions / max(n, 1),
        "mean_intervention_magnitude": (statistics.mean(interv_mags) if interv_mags else 0.0),
        "solver_failure_count": solver_fail,
        "dropped_command_count": dropped,
        "commanded_count": commanded,
        "recovery_time_sec": recovery_time if recovery_time is not None else float("nan"),
        "joint_limit_violations": 0,  # no joint outputs in feature-level retarget
        "min_obstacle_clearance_m": min_obstacle_clearance(positions, obstacles),
        "step_wall_ms_mean": step_ms,  # last step only marker; full mean below
    }
    return row


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Retargeting v1.1 research benchmarks")
    ap.add_argument("--config", type=Path, default=ROOT / "benchmarks/config/v11.yaml")
    ap.add_argument("--output", type=Path, default=ROOT / "results/v1.1")
    ap.add_argument("--smoke", action="store_true", help="Fast CI smoke (short duration)")
    ap.add_argument("--repeat", type=int, default=1, help="Repeatability runs")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    if args.smoke or cfg.get("smoke"):
        cfg["duration_sec"] = 1.0
        cfg["trajectories"] = [
            "horizontal_sweep",
            "one_second_dropout",
            "obstacle_intersection",
        ]
        cfg["combinations"] = [
            {"retargeter": "shoulder_relative", "safety": "reject"},
            {"retargeter": "sew_orientation", "safety": "cbf_qp"},
        ]

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    seed = int(cfg.get("seed", 42))
    rows: List[dict] = []
    for rep in range(int(args.repeat)):
        rng = np.random.default_rng(seed + rep)
        for combo in cfg["combinations"]:
            for traj in cfg["trajectories"]:
                row = run_one(combo["retargeter"], combo["safety"], traj, cfg, rng)
                row["repeat"] = rep
                rows.append(row)

    # Repeatability: std of path length across repeats when repeat>1
    repeatability = {}
    if args.repeat > 1:
        from collections import defaultdict

        groups = defaultdict(list)
        for r in rows:
            key = (r["retargeter"], r["safety"], r["trajectory"])
            groups[key].append(r["ee_path_length_m"])
        for k, vals in groups.items():
            repeatability[str(k)] = {
                "path_length_mean": statistics.mean(vals),
                "path_length_stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            }

    env = {
        "hostname": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "smoke": bool(args.smoke or cfg.get("smoke")),
        "command": (
            "python3 benchmarks/benchmark_retargeting_v11.py"
            + (" --smoke" if args.smoke else "")
            + f" --config {args.config} --output {out_dir}"
        ),
        "notes": (
            "CPU pure-Python retargeting + safety filter benchmarks. "
            "Not physical Franka. Not full SEW-Mimic joint solve. "
            "CBF-QP is experimental kinematic filter only."
        ),
    }

    summary = {
        "version": "v1.1",
        "environment": env,
        "n_rows": len(rows),
        "combinations": cfg["combinations"],
        "trajectories": cfg["trajectories"],
        "aggregate_by_combo": {},
        "repeatability": repeatability,
    }
    from collections import defaultdict

    agg = defaultdict(list)
    for r in rows:
        key = f"{r['retargeter']}+{r['safety']}"
        agg[key].append(r)
    for key, group in agg.items():
        summary["aggregate_by_combo"][key] = {
            "mapping_latency_mean_ms": statistics.mean(
                x["mapping_latency_mean_ms"] for x in group if not math.isnan(x["mapping_latency_mean_ms"])
            ),
            "safety_latency_mean_ms": statistics.mean(
                x["safety_latency_mean_ms"] for x in group if not math.isnan(x["safety_latency_mean_ms"])
            ),
            "mean_ee_path_length_m": statistics.mean(x["ee_path_length_m"] for x in group),
            "total_interventions": sum(x["intervention_count"] for x in group),
            "total_solver_failures": sum(x["solver_failure_count"] for x in group),
            "total_dropped_commands": sum(x["dropped_command_count"] for x in group),
            "mean_intervention_pct": statistics.mean(x["intervention_pct"] for x in group),
        }

    # Write outputs
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    (out_dir / "config-snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out_dir / "benchmark-summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fieldnames = list(rows[0].keys()) if rows else []
    with (out_dir / "benchmark-runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Simple plot if matplotlib available
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = list(summary["aggregate_by_combo"].keys())
        path_lens = [summary["aggregate_by_combo"][k]["mean_ee_path_length_m"] for k in labels]
        interv = [summary["aggregate_by_combo"][k]["mean_intervention_pct"] for k in labels]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].barh(labels, path_lens, color="#3b82f6")
        ax[0].set_xlabel("Mean EE path length (m)")
        ax[0].set_title("Path length by combo")
        ax[1].barh(labels, interv, color="#f59e0b")
        ax[1].set_xlabel("Mean intervention %")
        ax[1].set_title("Safety interventions")
        fig.suptitle("v1.1 retargeting benchmarks (CPU pure-Python)")
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / "combo_comparison.png", dpi=120)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        (out_dir / "logs" / "plot_skip.txt").write_text(f"plot skipped: {exc}\n")

    print(json.dumps(summary["aggregate_by_combo"], indent=2))
    print(f"Wrote results to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
