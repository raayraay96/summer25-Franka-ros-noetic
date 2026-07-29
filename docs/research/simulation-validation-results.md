# Simulation-Validation Results (Phase 7 executed)

Independent post-program engineering by Eric Raymond (2026). All numbers below
were generated in this repository on the audit host; no University of Wyoming,
HUMANS MOVE, Purdue, or paper-author involvement. **Simulation / pure-Python
only — no physical Franka, no ROS runtime.**

## Environment

Cursor Cloud VM, `Linux 6.12.94+ x86_64`, Python 3.10.20, numpy 1.24.4.
Full provenance: `results/v1.1-hardening/paired/environment.json`.
Latency legs `ros_callback_to_command_ms` and `fake_hardware_publication_ms` are
**not measured** (no ROS in this VM).

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

## What this does NOT establish

No physical Franka accuracy, no collision-free hardware execution, no metric
depth, no verified IK/JointState, no ROS callback/fake-hardware latency, no
paper-parity. See `claims-ledger.md` and `v1.1-final-limitations.md`.
