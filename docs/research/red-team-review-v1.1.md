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
5. **Are benchmarks paired?** Not yet. Committed `results/v1.1/` are
   single-replicate and are now **stale** vs the hardened solver (S2).
6. **Are confidence intervals reproducible?** Not yet produced (S2).
7. **Does the video execute the actual repository runtime?** There is no new
   runtime video. Existing demos are pure-Python (honestly labeled) (S2).
8. **Are reported joint values produced by verified IK?** No joint values are
   reported; `joint_positions=None` everywhere (S-none; correctly gated).
9. **Are frames and units explicit?** Yes — `frame-contracts.md` + schema
   provenance fields; the `z*0.5` heuristic is now the named
   `image_relative_z_scale` (S-none).
10. **Does any text imply hardware validation?** Not in the code/docs added
    here; the claims ledger enumerates prohibited wording (S-none, pending
    README pass — see below).
11. **Do all README numbers have committed evidence?** The README still shows
    the pre-hardening single-replicate benchmark table; those numbers are now
    stale (S2 — README not yet re-audited, deliberately deferred until the
    paired benchmark exists).
12. **Can a reviewer reproduce the headline result from one command?** Yes for
    the QP correctness headline: `python scripts/qp_cross_validation.py`
    (exit 0 = PASS). Not yet for a runtime demo (S2).

## Open issues

### Severity 1 (release-blocking) — none identified in the executed scope
No S1 remained after Phase 1–2: the previously weak obstacle test and the
suboptimal/mislabeled solver were the S1 risks, and both are fixed and
cross-validated. (The model-vs-reality gap is real but is an explicit scope
boundary, tracked as S2, not a defect in the modeled filter.)

### Severity 2 (blocks "simulation-validated" release name)
- **S2-A** No end-to-end ROS 2 fake-hardware RViz recording from the real
  runtime. Blocker: no ROS 2 / MoveIt / display in the audit VM. Smallest
  repair: run the ROS 2 fake-hardware launch on a ROS 2 host (or in the Noetic
  container extended with a ROS 2 layer) and record with the diagnostics topic.
- **S2-B** No verified simulated Panda IK / JointState evidence. Blocker: same.
  Smallest repair: MoveIt `compute_ik` on a Panda fake-hardware bringup; keep
  `sew_orientation` feature-only until then (already enforced in code).
- **S2-C** Benchmark not paired, single replicate, empty `repeatability`, no
  CIs, and now stale vs the hardened solver. Smallest repair: implement the
  paired protocol in `paired-experiment-protocol.md` (immutable per-seed input
  sequences replayed across methods; ≥30 reps; bootstrap CIs; per-trajectory).
- **S2-D** No fault-injection results and no versioned config schema validation.
- **S2-E** README/case-study still cite pre-hardening numbers; must be
  re-audited only after S2-C produces fresh paired evidence.

### Severity 3
- Oracle cross-validation is CPU-bound (~28 s for 6000 problems); CI uses an
  800-problem smoke. Acceptable.

## Release readiness

**Not release-ready as `v1.1-simulation-validated`.** The mathematical safety
core is now audited, corrected, and cross-validated (the highest-risk area), but
S2-A through S2-E remain. The PR must stay **draft**.
