# Paired Benchmark Protocol (Phase 7 design — not yet executed)

Independent post-program engineering by Eric Raymond (2026). This is the
**protocol**; the paired results are **not yet produced** (see status below).

## Problem with the current committed benchmark

`results/v1.1/` was generated with `n_rows=60` (6 method combos × 10
trajectories × **1** replicate), `repeatability: {}` empty, no std/p95/p99/CIs,
and its random stream advances across methods (inputs not provably paired). It
also predates the Phase 1–2 solver hardening, so it is now **stale**.

## Required design

1. **Immutable paired inputs.** For each `(trajectory, replicate, seed)`,
   generate one landmark sequence once, store its SHA-256, and replay the exact
   same sequence through every method combination. Never advance one RNG stream
   across methods.
2. **Replication.** ≥30 replicates for timing-sensitive metrics.
3. **Statistics per metric.** mean, median, std, p95, p99, 95% bootstrap CI,
   sample count, warmup count.
4. **Timing decomposition.** algorithm-only vs pipeline vs (on a ROS host)
   callback-to-command vs fake-hardware publication latency.
5. **Per-trajectory reporting** preserved; no single headline that averages
   unrelated trajectories without the per-trajectory table.
6. **Corrected metrics.**
   - orientation: angle between the intended transformed lower-arm direction
     and the emitted robot target direction (and, when IK/FK exists, achieved EE
     orientation);
   - rename temporal direction change to `direction_temporal_variation_rad`;
   - recovery time: from reacquisition until gate `PASS` and recovery blend
     `alpha == 1`;
   - clearance metrics include the configured margin AND the hard radius.
7. **Reported fields.** acceptance rate, dropped-command rate, hold/stop/recovery
   durations, intervention frequency+magnitude, min hard clearance, min
   safety-margin clearance, unsafe-accepted-state count (must be 0 by the
   independent validator), solver-failure count, solver-oracle mismatch count,
   deterministic replay hash match. Populate `repeatability`.
8. **Fault injection.** latency, jitter, packet loss, timestamp reordering,
   future timestamps, stale observations, confidence oscillation, brief/long
   occlusion, discontinuous reacquisition, CPU-load disturbance. Motivated by
   delayed-control literature — used to define fault cases, not to add features.

## Status

**NOT YET EXECUTED.** Blocker: the redesign is substantial and the ROS
callback-to-command / fake-hardware latency legs require a ROS host not present
in the audit VM. Until executed, no benchmark superiority is claimed and the
stale `results/v1.1/` numbers must not be cited as current (tracked S2-C in the
red-team review).
