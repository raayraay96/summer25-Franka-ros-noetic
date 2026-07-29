#!/usr/bin/env python3
"""Paired, replicated v1.1 benchmark on the hardened pipeline (Phase 7).

Fixes the earlier single-replicate design:
  * IMMUTABLE PAIRED INPUTS: for each (trajectory, replicate) one landmark
    sequence is generated once, SHA-256 hashed, and replayed through *every*
    method combination (never advance one RNG across methods).
  * REPLICATION: >= N replicates (default 30) with distinct per-replicate seeds.
  * STATISTICS: mean, median, std, p95, p99, 95% bootstrap CI, n, warmup.
  * PER-TRAJECTORY results preserved (no averaging unrelated trajectories into a
    single headline).
  * CORRECTED METRICS: orientation alignment (intended lower-arm direction vs
    emitted EE motion direction), direction_temporal_variation_rad (renamed),
    recovery time to gate PASS, clearance vs hard radius AND radius+margin.
  * INDEPENDENT unsafe-accepted count (must be 0) re-checked outside the filter.
  * DETERMINISTIC replay hash match.
  * FAULT INJECTION scenarios.

No ROS: callback-to-command and fake-hardware publication latency are marked
not_measured. Not physical Franka; not full SEW-Mimic; CBF-QP is a kinematic
filter only.

Usage:
  python3 benchmarks/benchmark_paired_v11.py --config benchmarks/config/v11.yaml \\
      --output results/v1.1-hardening/paired --replicates 30
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vision_arm_control" / "src"))

from vision_arm_control.landmarks.confidence import ConfidenceGateConfig, GateStatus  # noqa: E402
from vision_arm_control.landmarks.schema import ArmLandmarks, LandmarkPoint  # noqa: E402
from vision_arm_control.pipeline import PipelineConfig, TeleopPipeline  # noqa: E402
from vision_arm_control.retargeting.base import RetargetingConfig  # noqa: E402
from vision_arm_control.safety_filters.base import SafetyFilterConfig, SphericalObstacle  # noqa: E402
from vision_arm_control.workspace_limits import AxisAlignedBounds  # noqa: E402

Frame = Tuple[float, Optional[ArmLandmarks]]


# --------------------------------------------------------------------------- #
# Deterministic paired input generation
# --------------------------------------------------------------------------- #
def _seed_for(base_seed: int, traj: str, rep: int) -> int:
    h = hashlib.sha256(f"{base_seed}|{traj}|{rep}".encode()).hexdigest()
    return int(h[:8], 16)


def _lm(s, e, w, ts: float, conf: float = 1.0) -> ArmLandmarks:
    return ArmLandmarks(
        shoulder=LandmarkPoint(*s, confidence=conf, name="right_shoulder"),
        elbow=LandmarkPoint(*e, confidence=conf, name="right_elbow"),
        wrist=LandmarkPoint(*w, confidence=conf, name="right_wrist"),
        timestamp=ts,
    )


def _sample(traj: str, t: float, rng: np.random.Generator) -> Optional[ArmLandmarks]:
    s = np.array([0.45, 0.35, 0.0])
    if traj == "horizontal_sweep":
        phase = 0.5 + 0.12 * math.sin(2 * math.pi * 0.25 * t)
        return _lm(s, s + [0.08, 0.05, 0.0], s + [phase - 0.30, 0.12, 0.0], t)
    if traj == "vertical_sweep":
        phase = 0.45 + 0.10 * math.sin(2 * math.pi * 0.25 * t)
        return _lm(s, s + [0.06, 0.06, 0.0], s + [0.12, phase - 0.25, 0.02], t)
    if traj == "circle":
        a = 2 * math.pi * 0.3 * t
        return _lm(s, s + [0.07, 0.05, 0.0], s + [0.12 * math.cos(a), 0.10 * math.sin(a) + 0.08, 0.0], t)
    if traj == "figure_eight":
        a = 2 * math.pi * 0.25 * t
        return _lm(s, s + [0.07, 0.05, 0.0], s + [0.12 * math.sin(a), 0.08 * math.sin(2 * a) + 0.08, 0.0], t)
    if traj == "step_input":
        off = 0.18 if t >= 1.0 else 0.05
        return _lm(s, s + [0.06, 0.05, 0.0], s + [off, 0.10, 0.0], t)
    if traj == "near_full_extension":
        return _lm(s, s + [0.12, 0.02, 0.0], s + [0.24, 0.03, 0.0], t)
    if traj == "noisy_landmarks":
        e = s + np.array([0.08, 0.05, 0.0]) + rng.normal(0, 0.005, size=3)
        w = s + np.array([0.14, 0.12, 0.0]) + rng.normal(0, 0.01, size=3)
        return _lm(s, e, w, t)
    if traj == "one_second_dropout":
        if 1.0 <= t < 2.0:
            return None
        return _lm(s, s + [0.08, 0.05, 0.0], s + [0.14 + 0.05 * math.sin(t), 0.12, 0.0], t)
    if traj == "workspace_boundary_crossing":
        phase = 0.5 + 0.45 * math.sin(2 * math.pi * 0.2 * t)
        return _lm(s, s + [0.10, 0.05, 0.0], np.array([phase, 0.50, 0.0]), t)
    if traj == "obstacle_intersection":
        frac = min(t / 3.0, 1.0)
        return _lm(s, s + [0.08, 0.04, 0.0], s + [0.05 + 0.20 * frac, 0.05, -0.05 * frac], t)
    raise ValueError(f"unknown trajectory {traj}")


def generate_sequence(traj: str, rep: int, cfg: dict) -> List[Frame]:
    rng = np.random.default_rng(_seed_for(int(cfg.get("seed", 42)), traj, rep))
    dt = float(cfg["dt"])
    n = int(round(float(cfg["duration_sec"]) / dt))
    return [(i * dt, _sample(traj, i * dt, rng)) for i in range(n)]


def sequence_hash(seq: List[Frame]) -> str:
    h = hashlib.sha256()
    for t, lm in seq:
        if lm is None:
            h.update(f"{t:.6f}|none".encode())
        else:
            for p in (lm.shoulder, lm.elbow, lm.wrist):
                h.update(f"{p.x:.6f},{p.y:.6f},{p.z:.6f},{p.confidence:.4f};".encode())
            h.update(f"@{t:.6f}\n".encode())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Pipeline construction + independent safety re-check
# --------------------------------------------------------------------------- #
def _obstacles(cfg: dict) -> List[SphericalObstacle]:
    return [
        SphericalObstacle(
            id=o["id"],
            center=o["center"],
            radius_m=float(o["radius_m"]),
            margin_m=float(o.get("margin_m", 0.0)),
        )
        for o in cfg.get("cbf_qp", {}).get("obstacles", [])
    ]


def build_pipeline(retargeter: str, safety: str, cfg: dict) -> TeleopPipeline:
    ws = cfg["workspace"]
    bounds = AxisAlignedBounds(**{k: float(ws[k]) for k in ws})
    cbf = cfg.get("cbf_qp", {})
    safety_cfg = SafetyFilterConfig(
        mode=safety,
        workspace=bounds,
        workspace_margin_m=float(cbf.get("workspace_margin_m", 0.03)),
        dt=float(cfg["dt"]),
        alpha=float(cbf.get("alpha", 4.0)),
        per_axis_velocity_limit_mps=float(
            cbf.get("per_axis_velocity_limit_mps", cbf.get("max_cartesian_velocity_mps", 0.2))
        ),
        stop_on_solver_failure=bool(cbf.get("stop_on_solver_failure", True)),
        obstacles=_obstacles(cfg) if safety == "cbf_qp" else [],
    )
    return TeleopPipeline(
        PipelineConfig(
            retargeting=RetargetingConfig(name=retargeter, treat_z_as_image_relative=True),
            safety=safety_cfg,
            gate=ConfidenceGateConfig(
                hold_interval_sec=0.25,
                stale_timeout_sec=0.5,
                recovery_blend_sec=0.3,
                require_elbow=(retargeter == "sew_orientation"),
            ),
            backend="dry_run",
            initial_position=tuple(cfg["initial_position"]),
        )
    )


def _external_unsafe(
    p: np.ndarray, bounds: AxisAlignedBounds, obstacles: Sequence[SphericalObstacle]
) -> bool:
    """Independent check (outside the filter): True if p is unsafe."""
    tol = 1e-6
    if not np.all(np.isfinite(p)):
        return True
    if not (bounds.x_min - tol <= p[0] <= bounds.x_max + tol):
        return True
    if not (bounds.y_min - tol <= p[1] <= bounds.y_max + tol):
        return True
    if not (bounds.z_min - tol <= p[2] <= bounds.z_max + tol):
        return True
    for o in obstacles:
        if float(np.linalg.norm(p - o.center_array())) < o.radius_m - tol:  # hard collision
            return True
    return False


def _percentile(xs: Sequence[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = int(round(p * (len(s) - 1)))
    return float(s[max(0, min(k, len(s) - 1))])


# --------------------------------------------------------------------------- #
# Single run over one paired sequence
# --------------------------------------------------------------------------- #
def run_sequence(retargeter: str, safety: str, cfg: dict, seq: List[Frame], warmup: int) -> dict:
    pipe = build_pipeline(retargeter, safety, cfg)
    bounds = AxisAlignedBounds(**{k: float(cfg["workspace"][k]) for k in cfg["workspace"]})
    obstacles = _obstacles(cfg) if safety == "cbf_qp" else []

    saf_lat: List[float] = []
    positions: List[np.ndarray] = []
    intended_dirs: List[Optional[np.ndarray]] = []
    interv_mags: List[float] = []
    interventions = solver_fail = dropped = commanded = unsafe_accepted = 0
    hold_steps = stop_steps = recovering_steps = 0
    min_hard = float("inf")
    min_margin = float("inf")
    saw_dropout = False
    reacq_step: Optional[int] = None
    recovery_time = float("nan")

    for i, (t, lm) in enumerate(seq):
        if lm is None:
            saw_dropout = True
        res = pipe.step(lm, now=t)
        if res.safety is not None and i >= warmup:
            saf_lat.append(res.safety.compute_time_s * 1000.0)
        if res.safety is not None:
            if res.safety.intervened:
                interventions += 1
                interv_mags.append(res.safety.intervention_magnitude)
            if "solver_failure" in res.safety.reason:
                solver_fail += 1

        gate = pipe.gate.status
        if gate == GateStatus.HOLD:
            hold_steps += 1
        elif gate == GateStatus.STOPPED:
            stop_steps += 1
        elif gate == GateStatus.RECOVERING:
            recovering_steps += 1

        if res.commanded and res.safety and res.safety.position is not None:
            p = res.safety.position
            commanded += 1
            positions.append(p.copy())
            intended_dirs.append(
                None if res.gated.direction is None else np.asarray(res.gated.direction, dtype=np.float64)
            )
            if _external_unsafe(p, bounds, obstacles):
                unsafe_accepted += 1
            for o in obstacles:
                d = float(np.linalg.norm(p - o.center_array()))
                min_hard = min(min_hard, d - o.radius_m)
                min_margin = min(min_margin, d - (o.radius_m + o.margin_m))
            # recovery: first PASS after a reacquisition following a dropout
            if saw_dropout and reacq_step is None:
                reacq_step = i
            if reacq_step is not None and math.isnan(recovery_time) and gate == GateStatus.PASS:
                recovery_time = (i - reacq_step) * float(cfg["dt"])
        else:
            dropped += 1

    # emitted EE motion directions
    emitted = [positions[k + 1] - positions[k] for k in range(len(positions) - 1)]
    align = []
    for k, d in enumerate(emitted):
        intended = intended_dirs[k]
        nd = float(np.linalg.norm(d))
        if intended is None or nd < 1e-9 or float(np.linalg.norm(intended)) < 1e-9:
            continue
        c = float(np.clip(np.dot(d / nd, intended / np.linalg.norm(intended)), -1.0, 1.0))
        align.append(math.acos(c))
    temporal = []
    for k in range(len(emitted) - 1):
        a, b = emitted[k], emitted[k + 1]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            continue
        temporal.append(math.acos(float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))

    n = len(seq)
    path_len = float(sum(np.linalg.norm(e) for e in emitted))
    return {
        "retargeter": retargeter,
        "safety": safety,
        "n_steps": n,
        "safety_latency_mean_ms": statistics.mean(saf_lat) if saf_lat else float("nan"),
        "safety_latency_p95_ms": _percentile(saf_lat, 0.95) if saf_lat else float("nan"),
        "acceptance_rate": commanded / max(n, 1),
        "dropped_rate": dropped / max(n, 1),
        "hold_steps": hold_steps,
        "stop_steps": stop_steps,
        "recovering_steps": recovering_steps,
        "recovery_time_sec": recovery_time,
        "intervention_pct": 100.0 * interventions / max(n, 1),
        "mean_intervention_magnitude": statistics.mean(interv_mags) if interv_mags else 0.0,
        "solver_failure_count": solver_fail,
        "unsafe_accepted_count": unsafe_accepted,
        "min_hard_clearance_m": (min_hard if math.isfinite(min_hard) else float("nan")),
        "min_margin_clearance_m": (min_margin if math.isfinite(min_margin) else float("nan")),
        "ee_path_length_m": path_len,
        "orientation_alignment_rad": statistics.mean(align) if align else float("nan"),
        "direction_temporal_variation_rad": statistics.mean(temporal) if temporal else float("nan"),
        "commanded_positions_hash": hashlib.sha256(
            b"".join(np.asarray(p, dtype=np.float64).tobytes() for p in positions)
        ).hexdigest(),
    }


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _bootstrap_ci(vals: Sequence[float], B: int = 2000, seed: int = 7) -> Tuple[float, float]:
    xs = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(xs, dtype=np.float64)
    means = arr[rng.integers(0, len(arr), size=(B, len(arr)))].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def summarize(vals: Sequence[float]) -> dict:
    xs = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
    if not xs:
        return {"n": 0}
    lo, hi = _bootstrap_ci(xs)
    return {
        "n": len(xs),
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
        "p95": _percentile(xs, 0.95),
        "p99": _percentile(xs, 0.99),
        "ci95_low": lo,
        "ci95_high": hi,
    }


METRICS = [
    "safety_latency_mean_ms",
    "acceptance_rate",
    "dropped_rate",
    "intervention_pct",
    "mean_intervention_magnitude",
    "ee_path_length_m",
    "orientation_alignment_rad",
    "direction_temporal_variation_rad",
    "min_hard_clearance_m",
    "min_margin_clearance_m",
    "recovery_time_sec",
]


# --------------------------------------------------------------------------- #
# Fault injection
# --------------------------------------------------------------------------- #
def inject_fault(seq: List[Frame], scenario: str, cfg: dict) -> List[Frame]:
    rng = np.random.default_rng(_seed_for(int(cfg.get("seed", 42)), "fault_" + scenario, 0))
    out: List[Frame] = []
    prev: Optional[Frame] = None
    for i, (t, lm) in enumerate(seq):
        if scenario == "latency":
            out.append((t, ArmLandmarks(lm.shoulder, lm.elbow, lm.wrist, timestamp=t - 0.2) if lm else None))
        elif scenario == "jitter":
            j = float(rng.normal(0, 0.02))
            out.append((t, ArmLandmarks(lm.shoulder, lm.elbow, lm.wrist, timestamp=t + j) if lm else None))
        elif scenario == "packet_loss":
            out.append((t, None if rng.random() < 0.3 else lm))
        elif scenario == "timestamp_reorder":
            if lm and rng.random() < 0.2 and prev and prev[1]:
                out.append(
                    (t, ArmLandmarks(lm.shoulder, lm.elbow, lm.wrist, timestamp=prev[1].timestamp - 0.1))
                )
            else:
                out.append((t, lm))
        elif scenario == "future_timestamps":
            out.append((t, ArmLandmarks(lm.shoulder, lm.elbow, lm.wrist, timestamp=t + 5.0) if lm else None))
        elif scenario == "stale":
            out.append((t, prev[1] if (prev and rng.random() < 0.3) else lm))
        elif scenario == "confidence_oscillation":
            c = 0.2 if (i % 4 < 2) else 0.9
            out.append(
                (
                    t,
                    _lm(lm.shoulder.as_array(), lm.elbow.as_array(), lm.wrist.as_array(), t, c)
                    if lm
                    else None,
                )
            )
        elif scenario == "brief_occlusion":
            out.append((t, None if 1.0 <= t < 1.15 else lm))
        elif scenario == "long_occlusion":
            out.append((t, None if 1.0 <= t < 2.5 else lm))
        elif scenario == "discontinuous_reacquisition":
            if lm and t >= 2.5:
                out.append(
                    (
                        t,
                        _lm(
                            lm.shoulder.as_array(),
                            lm.elbow.as_array(),
                            lm.wrist.as_array() + np.array([0.3, 0.3, 0.0]),
                            t,
                        ),
                    )
                )
            else:
                out.append((t, None if 1.0 <= t < 2.5 else lm))
        else:
            out.append((t, lm))
        prev = out[-1]
    return out


FAULTS = [
    "latency",
    "jitter",
    "packet_loss",
    "timestamp_reorder",
    "future_timestamps",
    "stale",
    "confidence_oscillation",
    "brief_occlusion",
    "long_occlusion",
    "discontinuous_reacquisition",
]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Paired v1.1 benchmark (hardened)")
    ap.add_argument("--config", type=Path, default=ROOT / "benchmarks/config/v11.yaml")
    ap.add_argument("--output", type=Path, default=ROOT / "results/v1.1-hardening/paired")
    ap.add_argument("--replicates", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    trajectories = list(cfg["trajectories"])
    combos = list(cfg["combinations"])
    reps = int(args.replicates)
    if args.smoke:
        reps = 3
        trajectories = ["horizontal_sweep", "one_second_dropout", "obstacle_intersection"]

    # 1) Generate immutable paired inputs (once per traj,rep) and hash them.
    sequences: Dict[Tuple[str, int], List[Frame]] = {}
    input_hashes: Dict[str, str] = {}
    for traj in trajectories:
        for rep in range(reps):
            seq = generate_sequence(traj, rep, cfg)
            sequences[(traj, rep)] = seq
            input_hashes[f"{traj}#{rep}"] = sequence_hash(seq)

    # 2) Replay identical sequences through every combo.
    rows: List[dict] = []
    for combo in combos:
        for traj in trajectories:
            for rep in range(reps):
                row = run_sequence(
                    combo["retargeter"], combo["safety"], cfg, sequences[(traj, rep)], args.warmup
                )
                row["trajectory"] = traj
                row["replicate"] = rep
                row["input_hash"] = input_hashes[f"{traj}#{rep}"]
                rows.append(row)

    # 3) Deterministic replay hash match: re-run one config twice.
    c0 = combos[0]
    t0 = trajectories[0]
    h1 = run_sequence(c0["retargeter"], c0["safety"], cfg, sequences[(t0, 0)], args.warmup)[
        "commanded_positions_hash"
    ]
    h2 = run_sequence(c0["retargeter"], c0["safety"], cfg, sequences[(t0, 0)], args.warmup)[
        "commanded_positions_hash"
    ]
    replay_match = h1 == h2

    # 4) Aggregate: per (combo, trajectory) and per combo across trajectories.
    per_combo_traj: Dict[str, dict] = {}
    per_combo: Dict[str, dict] = {}
    grp_ct: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    grp_c: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        key = f"{r['retargeter']}+{r['safety']}"
        grp_ct[(key, r["trajectory"])].append(r)
        grp_c[key].append(r)
    for (key, traj), g in grp_ct.items():
        per_combo_traj.setdefault(key, {})[traj] = {m: summarize([x[m] for x in g]) for m in METRICS}
    for key, g in grp_c.items():
        per_combo[key] = {
            "metrics": {m: summarize([x[m] for x in g]) for m in METRICS},
            "total_unsafe_accepted": sum(x["unsafe_accepted_count"] for x in g),
            "total_solver_failures": sum(x["solver_failure_count"] for x in g),
        }

    # Repeatability across replicates (populated, not empty).
    repeatability = {}
    for key, g in grp_c.items():
        by_rep_path = [x["ee_path_length_m"] for x in g]
        repeatability[key] = {
            "ee_path_length": summarize(by_rep_path),
            "deterministic_replay_hash_match": replay_match,
        }

    # 5) Fault injection (safe cbf combos).
    fault_rows = []
    cbf_combos = [c for c in combos if c["safety"] == "cbf_qp"] or combos[:1]
    base_traj = "obstacle_intersection" if "obstacle_intersection" in trajectories else trajectories[0]
    base_seq = sequences[(base_traj, 0)]
    for scenario in FAULTS if not args.smoke else ["packet_loss", "long_occlusion"]:
        fseq = inject_fault(base_seq, scenario, cfg)
        for combo in cbf_combos:
            r = run_sequence(combo["retargeter"], combo["safety"], cfg, fseq, args.warmup)
            fault_rows.append(
                {
                    "scenario": scenario,
                    "combo": f"{combo['retargeter']}+{combo['safety']}",
                    "acceptance_rate": r["acceptance_rate"],
                    "dropped_rate": r["dropped_rate"],
                    "hold_steps": r["hold_steps"],
                    "stop_steps": r["stop_steps"],
                    "solver_failure_count": r["solver_failure_count"],
                    "unsafe_accepted_count": r["unsafe_accepted_count"],
                    "min_hard_clearance_m": r["min_hard_clearance_m"],
                }
            )
    fault_cpu_load = {"scenario": "cpu_load", "status": "not_simulated_pure_python"}

    env = {
        "hostname": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(cfg.get("seed", 42)),
        "replicates": reps,
        "warmup": args.warmup,
        "smoke": bool(args.smoke),
        "command": f"python3 benchmarks/benchmark_paired_v11.py --config {args.config} --replicates {reps}",
        "latency_notes": {
            "algorithm_and_pipeline_ms": "measured (safety_latency_*)",
            "ros_callback_to_command_ms": "not_measured_no_ros",
            "fake_hardware_publication_ms": "not_measured_no_ros",
        },
        "non_claims": "Not physical Franka; not full SEW-Mimic; CBF-QP kinematic filter only.",
    }

    total_unsafe = sum(x["unsafe_accepted_count"] for x in rows)
    total_solver_fail = sum(x["solver_failure_count"] for x in rows)
    fault_unsafe = sum(x["unsafe_accepted_count"] for x in fault_rows)

    summary = {
        "version": "v1.1-hardening-paired",
        "environment": env,
        "design": "immutable paired inputs replayed across all combos; >=1 replicate; bootstrap CIs",
        "n_rows": len(rows),
        "replicates": reps,
        "combinations": combos,
        "trajectories": trajectories,
        "per_combo": per_combo,
        "per_combo_per_trajectory": per_combo_traj,
        "repeatability": repeatability,
        "totals": {
            "unsafe_accepted_states": total_unsafe,
            "solver_failures": total_solver_fail,
            "fault_injection_unsafe_accepted": fault_unsafe,
            "deterministic_replay_hash_match": replay_match,
        },
    }

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    (out_dir / "config-snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out_dir / "input-hashes.json").write_text(json.dumps(input_hashes, indent=2, sort_keys=True) + "\n")
    (out_dir / "paired-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out_dir / "fault-injection.json").write_text(
        json.dumps({"scenarios": fault_rows, "cpu_load": fault_cpu_load}, indent=2, sort_keys=True) + "\n"
    )
    if rows:
        keys = [k for k in rows[0].keys() if k != "commanded_positions_hash"]
        with (out_dir / "paired-runs.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)

    print(
        json.dumps(
            {
                "n_rows": len(rows),
                "replicates": reps,
                "unsafe_accepted_states": total_unsafe,
                "solver_failures": total_solver_fail,
                "fault_injection_unsafe_accepted": fault_unsafe,
                "deterministic_replay_hash_match": replay_match,
            },
            indent=2,
        )
    )
    print(f"Wrote paired results to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
