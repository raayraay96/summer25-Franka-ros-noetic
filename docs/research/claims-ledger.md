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
| 10 | One end-to-end RViz recording from the real ROS 2 runtime, with live diagnostics. | **P** | none — blocked: no ROS 2 / display in the VM | — | "verified end-to-end robot simulation" |
| 11 | Verified simulated Panda IK (MoveIt/compute_ik) producing JointState evidence. | **P** | none — blocked: no ROS/MoveIt in the VM | — | "joint retargeting validated" |
| 12 | Physical Franka tracking / collision-free execution / metric accuracy. | **P** (out of scope) | none | — | any physical-hardware claim |

## Standing prohibited claims (never make)

- Physical Franka validation, collision-free hardware execution, manufacturer-certified safety.
- Full SEW-Mimic reproduction or its optimality guarantee.
- Metric monocular depth; calibrated camera-to-base geometry.
- Formal continuous- or discrete-time forward invariance for the implementation.
- Benchmark superiority without paired inputs + uncertainty.
- "Production-ready" because CI is green.
- University of Wyoming / HUMANS MOVE / Purdue / paper-author involvement in this post-program work.
