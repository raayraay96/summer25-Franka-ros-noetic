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
- **S2-A (open)** No end-to-end ROS 2 fake-hardware RViz recording from the real
  runtime. Blocker: no ROS 2 / MoveIt / display in the audit VM. Smallest
  repair: run the ROS 2 fake-hardware launch on a ROS 2 host (or in the Noetic
  container extended with a ROS 2 layer) and record with the diagnostics topic.
- **S2-B (open)** No verified simulated Panda IK / JointState evidence. Blocker:
  same. Smallest repair: MoveIt `compute_ik` on a Panda fake-hardware bringup;
  keep `sew_orientation` feature-only until then (already enforced in code).
- **S2-C (resolved, pure-Python legs)** Benchmark is now paired, 30 replicates,
  bootstrap CIs, per-trajectory, populated `repeatability`, 0 unsafe accepted,
  fault injection. ROS callback/fake-hardware latency legs remain not measured.
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

**Not release-ready as `v1.1-simulation-validated`.** The mathematical safety
core is audited, corrected, and cross-validated; the benchmark is now paired and
statistically characterized; config schema and fault injection are done
(S2-C/D/E resolved). The two remaining blockers — **S2-A** (end-to-end ROS 2
fake-hardware RViz demo) and **S2-B** (verified simulated Panda IK) — cannot be
executed in this environment (no ROS 2 / MoveIt / display). The PR must stay
**draft** until those run on a ROS host.
