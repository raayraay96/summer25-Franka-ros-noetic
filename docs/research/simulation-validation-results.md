# Simulation-Validation Results (Phase 7 + S2-A ROS 2 dry_run)

Independent post-program engineering by Eric Raymond (2026). All numbers below
were generated in this repository on documented hosts; no University of Wyoming,
HUMANS MOVE, Purdue, or paper-author involvement. **Simulation / dry_run only —
no physical Franka.**

## Environments

### Pure-Python (paired benchmark, QP oracle)

Cursor Cloud VM, `Linux 6.12.94+ x86_64`, Python 3.10.20, numpy 1.24.4.
Full provenance: `results/v1.1-hardening/paired/environment.json`.

### ROS 2 Humble dry_run (S2-A + S2-B)

- **S2-A** (end-to-end RViz): Scholar `scholar-b000.rcac.purdue.edu`, SLURM job
  **459478**, container `ros2_humble_franka.sif`, ROS 2 Humble, Python 3.10.12,
  git SHA `5403fe775f80c1fac604d695178314662179f9da`, date UTC
  `2026-07-29T21:05:20+00:00`. Provenance:
  `results/v1.1-hardening/ros2/environment.json`. `physical_hardware: false`.
  Mock landmarks; teleop pipeline dry_run (no physical output).
- **S2-B** (MoveIt IK): Scholar + same container family
  `ros2_humble_franka.sif`, focused SLURM job **459481** (re-run after headless
  `move_group` launch fix). Dry_run only; no physical Franka.

## 1. QP solver correctness (Phase 1)

6000 seeded QPs vs a trusted OSQP + SciPy-linprog oracle: **0** feasibility /
objective / distance / KKT mismatches, **0** cases where the custom solver is
worse. Evidence: `results/v1.1-hardening/qp-cross-validation.json`,
`docs/research/qp-correctness-audit.md`.

## 2. Paired, replicated pipeline benchmark (Phase 7)

Design: one immutable landmark sequence per `(trajectory, replicate)`
(SHA-256 hashed in `input-hashes.json`) replayed through **all 6** method
combinations; **30 replicates**; 10 trajectories; 3-step warmup. Command:

```bash
python benchmarks/benchmark_paired_v11.py --config benchmarks/config/v11.yaml \
    --replicates 30 --output results/v1.1-hardening/paired
```

Evidence: `results/v1.1-hardening/paired/paired-summary.json` (per-combo and
per-combo-per-trajectory mean/median/std/p95/p99/95%-bootstrap-CI),
`paired-runs.csv` (1800 rows), `fault-injection.json`.

### Safety (the headline)

| Aggregate | Value |
|---|---:|
| Total paired runs | 1800 |
| **Unsafe accepted next states (independent re-check)** | **0** |
| Solver failures | 0 |
| Min hard obstacle clearance (cbf_qp) | **+0.014 m** (never inside the hard radius) |
| Deterministic replay hash match | true |

### Per-combo (30 replicates × 10 trajectories, mean [95% bootstrap CI])

| Combo | Intervention % | Safety latency (ms) | Unsafe accepted |
|---|---|---:|---:|
| shoulder_relative + reject | 1.4 [0.9, 1.8] | 0.001 | 0 |
| shoulder_relative + clamp | 1.4 [0.9, 1.8] | 0.002 | 0 |
| shoulder_relative + cbf_qp | 86.6 [85.0, 88.2] | 8.79 | 0 |
| sew_orientation + reject | 1.4 [0.9, 1.8] | 0.002 | 0 |
| sew_orientation + clamp | 1.4 [0.9, 1.8] | 0.002 | 0 |
| sew_orientation + cbf_qp | 86.2 [84.5, 87.9] | 8.77 | 0 |

Notes (honest):
- The high cbf_qp intervention % is expected under the tight per-axis velocity
  box + barrier model on these aggressive synthetic trajectories; it is a
  behavior characterization, **not** a superiority claim.
- cbf_qp safety latency (~8.8 ms mean) is the cost of the **exact** active-set
  solver (~13 constraints, up to C(13,3) active sets). It is well under the
  50 ms step period but materially higher than the pre-hardening early-exit
  solver; reported transparently.
- `min_margin_clearance` is negative (~-0.036 m) on the `obstacle_intersection`
  trajectory: the aggressive input drives the EE into the soft `radius+margin`
  band, but the **hard** collision radius is never entered (clearance +0.014 m),
  which matches the discrete control-barrier rule (recovery-only motion inside
  the margin).
- Orientation-alignment (~1.4–1.5 rad) is a frame-consistent proxy between the
  intended lower-arm direction and the emitted EE motion direction under the
  uncalibrated identity-ish axis map; it is not an achieved-EE-orientation
  metric (no IK/FK) and not a paper-parity claim.

## 3. Fault injection (Phase 7)

Base trajectory `obstacle_intersection`, both cbf_qp combos. Evidence:
`results/v1.1-hardening/paired/fault-injection.json`.

| Scenario | Acceptance | Unsafe accepted | Behavior |
|---|---:|---:|---|
| latency / jitter / packet_loss / timestamp_reorder / future_timestamps / stale | 1.00 (0.97 conf-osc) | 0 | gated safely |
| confidence_oscillation | 0.97 | 0 | drops low-confidence frames |
| long_occlusion | 0.69 | 0 | holds then stops during occlusion |
| discontinuous_reacquisition | 0.69 | 0 | holds/stops, no unsafe jump |
| cpu_load | — | — | not_simulated_pure_python |

**Every fault scenario produced 0 unsafe accepted states** and hard clearance
+0.014 m. Timestamp reordering / future timestamps / staleness are gated by the
`ConfidenceGate`; occlusions hold-then-stop per the default policy.

## 4. ROS 2 runtime (S2-A) — end-to-end dry_run (resolved)

Real ROS 2 Humble runtime on Scholar host `scholar-b000`, **dry_run only**,
mock landmarks, **no physical Franka**. SLURM job **459478**, container
`ros2_humble_franka.sif`, git SHA `5403fe775f80c1fac604d695178314662179f9da`,
provenance `results/v1.1-hardening/ros2/environment.json`
(`physical_hardware: false`).

### Media

| Artifact | Path |
|---|---|
| RViz recording (mp4) | `docs/media/v1.1-hardening/end-to-end-rviz.mp4` |
| GIF | `docs/media/v1.1-hardening/end-to-end-rviz.gif` |
| Thumbnail | `docs/media/v1.1-hardening/end-to-end-rviz-thumb.png` |
| Media README | `docs/media/v1.1-hardening/README.md` |

Launch path (recording): `teleop_pipeline_rviz.launch.py` — mock landmarks →
TeleopPipeline (shoulder_relative + cbf_qp) → geometric_unverified IK → RSP →
RViz. Labels: ROS 2 SIMULATION · MOCK LANDMARKS · PANDA FAKE HARDWARE · NO
PHYSICAL ROBOT · RETARGETER · SAFETY FILTER · live diagnostics topic.

### Scenario metrics (from committed `*_metrics.json`)

Evidence root: `results/v1.1-hardening/ros2/`. Numbers below are from the
JSON metrics files (rounded for readability where noted).

| Scenario | Strategy / safety | Command rate (Hz) | Mean latency (ms) | Intervention % | Acceptance | n_records |
|---|---|---:|---:|---:|---:|---:|
| scenario_a_shoulder_cbf | shoulder_relative + cbf_qp | ~20.05 | **0.388** | 0.0 | 1.0 | 428 |
| scenario_b_sew_cbf | sew_orientation + cbf_qp | ~20.05 | ~0.30 | 0.0 | 1.0 | 428 |
| scenario_c_obstacle_interv | obstacle intervention path | ~20.05 | ~11.2 | ~51.2 | 1.0 | 432 |
| rviz_demo (recorded) | shoulder_relative + cbf_qp | ~20.04 | ~11.6 | ~51.3 | 1.0 | 532 |

Exact file values (for audit): scenario_a `command_rate_hz=20.0469`,
`mean_latency_ms=0.3883`, `intervention_pct=0.0`, `n_records=428`;
scenario_b `mean_latency_ms=0.3002`, `n_records=428`; scenario_c
`mean_latency_ms=11.166`, `intervention_pct=51.157`, `n_records=432`;
rviz_demo `mean_latency_ms=11.628`, `intervention_pct=51.316`, `n_records=532`.
Solver failures: **0** across all four. Acceptance rate **1.0** on all four.

Topic rates on live diagnostics ~20 Hz for target/command/safety (and
joint_states when geometric IK / RSP path is active).

Supporting artifacts: `rviz_demo.jsonl` + `rviz_demo_metrics.json`;
scenario_a/b/c `*.jsonl`, `*_metrics.json`, `*_bag/` (rosbag2);
`environment.json`, `s2a_rviz_ok.txt`, `s2a_scenarios_ok.txt`.
Job log (host path): `.../franka-teleop-data/logs/v11-s2-459478.out`.

**Honest scope for S2-A:** dry_run, mock landmarks, no physical robot. IK used
for the RViz recording is **geometric_unverified** (or none on scenario A),
**not** verified MoveIt `compute_ik`.

## 5. S2-B — verified MoveIt IK (**resolved**; claim 11 **E**)

Focused re-run job **459481** after headless `move_group` launch fix (prior job
459478 had `/compute_ik` advertised but recorded 0 JointStates). Script:
`scripts/s2b_moveit_ik_evidence.py`. Launch: `panda_moveit_ik.launch.py`.
Service: `/compute_ik`. Environment: Scholar + `ros2_humble_franka.sif`.
Dry_run only; **no physical hardware**.

| Field | Value |
|---|---|
| Evidence file | `results/v1.1-hardening/ros2/s2b_joint_evidence.json` |
| status | `verified_moveit_jointstate_evidence` |
| n_samples | 16 |
| n_ok / n_fail | 16 / 0 |
| frame_id | `verified_moveit` (all samples) |
| within_limits | true (all samples) |
| dry_run / physical_hardware | true / false |
| scenario_s2b_moveit_ik.jsonl | 16 lines (`jsonl_n=16`, `jsonl_ok=16`) |
| bag | `scenario_s2b_moveit_ik_bag/` |

**16/16** limit-checked `sensor_msgs/JointState` samples via MoveIt
`compute_ik`, all with `frame_id=verified_moveit`.

**Honest scope:** Cartesian-target MoveIt IK evidence only. This does **not**
validate SEW joint retargeting, paper parity, or physical Franka execution.
`sew_orientation` remains feature-level only (`joint_positions=None`).

## What this does NOT establish

No physical Franka accuracy, no collision-free hardware execution, no metric
depth, no paper-parity, no full SEW-Mimic. S2-B is verified MoveIt Cartesian
`compute_ik` dry_run JointState evidence only — not SEW joint retargeting.
Paired pure-Python ROS callback / formal fake-hardware latency legs are still
not a separate multi-replicate study (S2-A reports dry_run diagnostic mean
latency only). See `claims-ledger.md` and `v1.1-final-limitations.md`.
