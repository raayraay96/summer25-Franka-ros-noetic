# Claims Ledger — v1.1-simulation-validated (candidate)

Independent post-program engineering by Eric Raymond (2026). No University of
Wyoming, HUMANS MOVE, Purdue, or cited-paper author supervised, approved, or
validated this work. Every allowed claim must be backed by committed evidence
generated in a documented environment.

Legend: **E** = evidence exists and was executed for this PR; **P** = pending /
not yet executed (blocker noted).

| # | Claim (allowed wording) | Status | Evidence | Environment | Prohibited stronger wording |
|---|---|---|---|---|---|
| 1 | The hardened CBF-QP solver returns a KKT-verified exact optimum on the modeled linear-constraint QP, matching a trusted OSQP/linprog oracle on 6000 seeded problems with 0 mismatches. | **E** | `results/v1.1-hardening/qp-cross-validation.json`, `scripts/qp_cross_validation.py`, `docs/research/qp-correctness-audit.md` | Cursor VM, Py3.10.20, numpy1.24.4, scipy1.10.1, osqp0.6.3 | "provably optimal for all QPs"; "formally verified solver" |
| 2 | Every accepted command is independently re-validated on the actual next state (finite, per-axis speed, workspace hard box, obstacle hard radius, non-decreasing margin). | **E** | `validate_next_state`, `tests/test_cbf_qp_properties.py` (`_external_safe` property over 500+ random calls) | same | "guarantees collision-free operation" |
| 3 | A sequential-projection result is never reported as optimal; the default stops on exact-solver failure. | **E** | `cbf_qp.py` status vocabulary; `test_projection_fallback_*` | same | "always finds a safe command" |
| 4 | The velocity limit is a per-axis box (`per_axis_velocity_limit_mps`), not a Euclidean speed cap. | **E** | `base.py`, `test_diagonal_motion_respects_per_axis_box` | same | "bounds Cartesian/Euclidean end-effector speed" |
| 5 | `sew_orientation` emits feature-level Cartesian + orientation-direction targets; no joint IK is produced. | **E** | `sew_orientation.py` (`joint_positions=None`, `ik_status=missing_ik_layer_feature_level_only`), `frame-contracts.md` | same | "reproduces SEW-Mimic"; "joint-space retargeting" |
| 6 | Image-relative z is relative monocular depth, de-weighted by the named `image_relative_z_scale`; not metric, no camera-to-base calibration implied. | **E** | `frame-contracts.md`, schema provenance fields | same | "metric depth"; "calibrated camera-to-base" |
| 7 | Pure-logic + property + oracle tests pass (143 total). | **E** | `pytest -q tests/` → 143 passed | Cursor VM Py3.10.20 | "fully verified system" |
| 8 | Noetic container + catkin + dry-run launch smoke pass. | **E** (CI) | GitHub Actions `noetic-container` (PR #5 head) | GitHub ubuntu-latest | "hardware-validated" |
| 9 | Paired 30-replicate benchmark on the hardened pipeline (identical inputs replayed across methods): 0 unsafe accepted next states in 1800 runs, 0 solver failures, deterministic replay match, per-combo bootstrap CIs, 10 fault-injection scenarios all 0 unsafe. | **E** | `results/v1.1-hardening/paired/`, `benchmarks/benchmark_paired_v11.py`, `simulation-validation-results.md` | Cursor VM, Py3.10.20 | any benchmark *superiority* claim; ROS callback/fake-hw latency (not measured) |
| 9b | Versioned config schema validation (fail-fast on unknown/missing keys, types, ranges, units, conflicts). | **E** | `config_schema.py`, `scripts/validate_config.py`, `tests/test_config_schema.py`, 5 fixtures | same | — |
| 10 | One end-to-end RViz recording from the real ROS 2 runtime, with live diagnostics (dry_run, mock landmarks, no physical Franka). | **E** | Media: `docs/media/v1.1-hardening/end-to-end-rviz.mp4`, `.gif`, `.thumb.png`, `README.md`. Runtime: `results/v1.1-hardening/ros2/rviz_demo.jsonl` + `rviz_demo_metrics.json`; `scenario_a_shoulder_cbf` / `scenario_b_sew_cbf` / `scenario_c_obstacle_interv` (`*.jsonl`, `*_metrics.json`, `*_bag/`); `environment.json`; `s2a_rviz_ok.txt`. | Scholar host `scholar-b000`, SLURM job **459478**, container `ros2_humble_franka.sif`, ROS 2 Humble, git SHA `5403fe775f80c1fac604d695178314662179f9da` | "verified end-to-end robot simulation"; "physical robot demo"; "verified MoveIt IK" (S2-A used geometric_unverified / none) |
| 11 | Verified simulated Panda IK (MoveIt/compute_ik) producing JointState evidence with `frame_id=verified_moveit`. | **E** | `results/v1.1-hardening/ros2/s2b_joint_evidence.json` (`status=verified_moveit_jointstate_evidence`, `n_samples=16`, `n_ok=16`, `n_fail=0`, all `frame_id=verified_moveit`, `within_limits=true`, `dry_run=true`, `physical_hardware=false`); `scenario_s2b_moveit_ik.jsonl` (16 lines); `scenario_s2b_moveit_ik_bag/`. Script: `scripts/s2b_moveit_ik_evidence.py`. Launch: `ros2_ws/.../panda_moveit_ik.launch.py`. Service `/compute_ik`. Focused re-run after headless `move_group` launch fix (prior job 459478 had service up but 0 samples). | Scholar + `ros2_humble_franka.sif`, SLURM job **459481** | "joint retargeting validated"; "SEW joint retargeting"; "full SEW-Mimic"; physical Franka (this is Cartesian-target MoveIt IK dry_run only; `sew_orientation` remains feature-level) |
| 12 | Physical Franka tracking / collision-free execution / metric accuracy. | **P** (out of scope) | none | — | any physical-hardware claim |

## Standing prohibited claims (never make)

- Physical Franka validation, collision-free hardware execution, manufacturer-certified safety.
- Full SEW-Mimic reproduction or its optimality guarantee.
- Metric monocular depth; calibrated camera-to-base geometry.
- Formal continuous- or discrete-time forward invariance for the implementation.
- Benchmark superiority without paired inputs + uncertainty.
- "Production-ready" because CI is green.
- University of Wyoming / HUMANS MOVE / Purdue / paper-author involvement in this post-program work.
- Physical joint retargeting / SEW-Mimic joint IK (claim 11 is dry_run MoveIt Cartesian `compute_ik` only; `sew_orientation` stays feature-level with `joint_positions=None`).
