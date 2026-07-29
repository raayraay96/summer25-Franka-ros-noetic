# Adversarial Red-Team Review — v1.1 (hardening pass)

Independent post-program engineering by Eric Raymond (2026). Severity: S1 =
release-blocking safety/correctness; S2 = blocks the "simulation-validated"
release name; S3 = minor.

## Answers to the mandated adversarial questions

1. **Can any accepted command violate the independently evaluated safety set?**
   Not in the modeled kinematic set: every accepted command is re-checked by
   `validate_next_state` (final authority) and a *separate* external validator
   in tests confirms this over 500+ random calls and 6 obstacle-approach
   directions. Caveat: the model is a point EE with spherical obstacles and an
   axis-aligned box — real geometry/perception are out of scope (S2, documented).
2. **Can malformed data produce movement?** No. NaN/Inf inputs, invalid `dt`,
   and solver failure all return `accepted=False, position=None`
   (`test_nan_and_inf_reject`, `test_invalid_dt_rejects`).
3. **Can timestamp manipulation bypass staleness?** The `ConfidenceGate` rejects
   timestamp rewind beyond tolerance and holds/stops on staleness
   (`test_confidence_gate.py`). Deeper adversarial timestamp fuzzing is not yet
   in the fault-injection matrix (S2).
4. **Can the solver report success on a nonoptimal/infeasible point?** No:
   `optimal` requires KKT verification; projection is relabeled and off by
   default; infeasible returns `None`. Cross-validated on 6000 QPs (S-none).
5. **Are benchmarks paired?** Yes now — `benchmark_paired_v11.py` replays one
   immutable per-`(trajectory, replicate)` input (SHA-256 hashed) across all
   methods; 30 replicates; 1800 runs; **0 unsafe accepted**; deterministic
   replay match. The old single-replicate `results/v1.1/` are superseded (was
   S2-C; now resolved for the pure-Python legs).
6. **Are confidence intervals reproducible?** Yes — per-combo bootstrap 95% CIs
   in `results/v1.1-hardening/paired/paired-summary.json` (seeded).
7. **Does the video execute the actual repository runtime?** Yes for S2-A:
   `docs/media/v1.1-hardening/end-to-end-rviz.{mp4,gif}` records the real ROS 2
   Humble pipeline on Scholar (job 459478, container `ros2_humble_franka.sif`,
   dry_run, mock landmarks, no physical robot). Live diagnostics JSONL + metrics
   under `results/v1.1-hardening/ros2/`. Pure-Python demos remain honestly
   labeled as such.
8. **Are reported joint values produced by verified IK?** Yes for S2-B dry_run
   Cartesian targets: job **459481** recorded 16/16 MoveIt `/compute_ik`
   JointStates with `frame_id=verified_moveit`, all `within_limits=true`,
   `status=verified_moveit_jointstate_evidence` in
   `results/v1.1-hardening/ros2/s2b_joint_evidence.json` (claim 11 **E**).
   Scope is simulated dry_run only (no physical Franka). This is **not** SEW
   joint retargeting: `sew_orientation` remains feature-level
   (`joint_positions=None`).
9. **Are frames and units explicit?** Yes — `frame-contracts.md` + schema
   provenance fields; the `z*0.5` heuristic is now the named
   `image_relative_z_scale` (S-none).
10. **Does any text imply hardware validation?** Not in the code/docs added
    here; the claims ledger enumerates prohibited wording. S2-A evidence is
    explicitly dry_run / mock landmarks / no physical Franka.
11. **Do all README numbers have committed evidence?** The README re-audited to
    the paired numbers; S2-A media README cites job 459478 and SHA
    `5403fe775f80c1fac604d695178314662179f9da`.
12. **Can a reviewer reproduce the headline result from one command?** Yes for
    the QP correctness headline: `python scripts/qp_cross_validation.py`
    (exit 0 = PASS). S2-A ROS 2 path is reproducible via
    `sbatch scholar/record_v11_hardening.slurm` on Scholar with the committed
    container (dry_run only).

## Open issues

### Severity 1 (release-blocking) — none identified in the executed scope
No S1 remained after Phase 1–2: the previously weak obstacle test and the
suboptimal/mislabeled solver were the S1 risks, and both are fixed and
cross-validated. (The model-vs-reality gap is real but is an explicit scope
boundary, tracked as S2, not a defect in the modeled filter.)

### Severity 2 (blocks "simulation-validated" release name)
- **S2-A (resolved)** End-to-end ROS 2 fake-hardware RViz recording from the real
  runtime with live diagnostics. Evidence: `docs/media/v1.1-hardening/end-to-end-rviz.{mp4,gif}`,
  `results/v1.1-hardening/ros2/` (JSONL, metrics, bags), SLURM job **459478**,
  container `ros2_humble_franka.sif`, host `scholar-b000.rcac.purdue.edu`,
  git SHA `5403fe775f80c1fac604d695178314662179f9da`. Scenario A
  (`shoulder_relative+cbf_qp`): command rate ~20.05 Hz, mean latency
  ~0.39 ms, 0 solver failures, acceptance 1.0 (dry_run, mock landmarks,
  `physical_hardware=false`). RViz demo used `ik=geometric_unverified` — **not**
  verified MoveIt. No physical Franka.
- **S2-B (resolved)** Verified simulated Panda MoveIt IK / JointState evidence
  on focused re-run job **459481** (after headless `move_group` launch fix;
  prior job 459478 had `/compute_ik` advertised but 0 samples). Evidence:
  - `results/v1.1-hardening/ros2/s2b_joint_evidence.json`:
    `status=verified_moveit_jointstate_evidence`, `n_samples=16`, `n_ok=16`,
    `n_fail=0`, all samples `frame_id=verified_moveit`, `within_limits=true`,
    `dry_run=true`, `physical_hardware=false`.
  - `scenario_s2b_moveit_ik.jsonl` (16 lines); `scenario_s2b_moveit_ik_bag/`.
  - Script: `scripts/s2b_moveit_ik_evidence.py`; launch
    `panda_moveit_ik.launch.py`; service `/compute_ik`.
  - Environment: Scholar + container `ros2_humble_franka.sif`.
  - Scope honesty: Cartesian-target MoveIt `compute_ik` dry_run only — **not**
    physical Franka, **not** SEW joint retargeting. `sew_orientation` remains
    feature-level (`joint_positions=None`).
- **S2-C (resolved, pure-Python legs)** Benchmark is now paired, 30 replicates,
  bootstrap CIs, per-trajectory, populated `repeatability`, 0 unsafe accepted,
  fault injection. ROS callback/fake-hardware latency legs remain not measured
  as a formal paired latency study (S2-A reports diagnostic mean latency on
  the dry_run pipeline only).
- **S2-D (resolved)** Versioned config-schema validation implemented
  (`config_schema.py`, `scripts/validate_config.py`, CI step, 18 tests).
  Fault-injection results are recorded (`fault-injection.json`).
- **S2-E (resolved)** README re-audited to the fresh paired numbers; the stale
  single-replicate table is explicitly marked superseded. (Case study still
  references older framing — minor, S3.)

### Severity 3
- Oracle cross-validation is CPU-bound (~28 s for 6000 problems); CI uses an
  800-problem smoke. Acceptable.

## Release readiness

**S2-A and S2-B both resolved** (simulation / dry_run scope). The mathematical
safety core is audited, corrected, and cross-validated; the benchmark is paired
and statistically characterized; config schema and fault injection are done
(S2-C/D/E resolved); **S2-A** has a real ROS 2 dry_run RViz recording and
diagnostics on Scholar (job **459478**, `ros2_humble_franka.sif`); **S2-B** has
verified MoveIt `/compute_ik` JointState evidence (job **459481**,
`status=verified_moveit_jointstate_evidence`, 16/16 limit-checked,
`frame_id=verified_moveit`). Engineering evidence gates for the
simulation-validated candidate are met; PR merge remains **optional / pending
human authorization** (draft wording may be lifted when a human authorizes).
Still **no** physical Franka claims, **no** paper parity, **no** metric depth,
**no** full SEW-Mimic; `sew_orientation` stays feature-level. Dry_run / no
physical Franka wording remains mandatory for all ROS evidence.
