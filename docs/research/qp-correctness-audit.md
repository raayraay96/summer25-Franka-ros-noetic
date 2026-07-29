# CBF-QP Correctness Audit (Phase 1)

Independent post-program engineering by Eric Raymond (2026). This audit treats
the original custom solver as **untrusted** and verifies the hardened solver
against two independent trusted tools.

## Problem

The Cartesian CBF safety layer solves, for a 3-vector command velocity `u`:

```
minimize    0.5 * ||u - u_nom||^2
subject to  a_i . u >= b_i    (workspace CBF, obstacle CBF, per-axis velocity box)
```

This is a strictly convex QP, so the minimizer is unique.

## Defects found in the audited baseline (`7b3ab90`)

1. **Non-exhaustive active-set search.** The old `solve_cbf_qp` enumerated
   active sets from the *violated-at-nominal* constraints first and `break`-ed
   on the first pool that yielded any feasible candidate. It could therefore
   return a feasible-but-**suboptimal** point while reporting `solved`.
2. **No KKT verification.** Dual multipliers were never checked for sign,
   stationarity, or complementary slackness.
3. **Projection fallback mislabeled as a solution.** `solved_projection` /
   `solved_projection_max_iter` were accepted as if optimal.
4. **Velocity semantics mismatch.** The nominal was pre-scaled by Euclidean
   norm, but the QP enforced a per-axis **box**; a diagonal correction could
   reach `sqrt(3) * vmax` Euclidean speed. The setting was mislabeled
   "cartesian velocity."
5. **No independent discrete-step validator.** Only the linear CBF residual on
   `u*` was checked, not the actual next state `p_next`.

## Corrections implemented

- **Exact active-set solver** (`solve_cbf_qp_ex`): enumerates *every* active set
  of size 0..3 over *all* constraints, solves each equality-constrained
  subproblem, keeps only primal-feasible candidates, and returns the
  minimum-cost one. Because the objective is strictly convex, the minimum-cost
  primal-feasible active-set solution is the global optimum.
- **Independent KKT verification** of the selected optimum: primal residual
  `<= 1e-7`, dual multipliers `>= -1e-7` (recovered from stationarity
  `u - u_nom = A^T lambda`), and stationarity residual `<= 1e-6`. Rank-deficient
  (duplicate / contradictory / near-singular) active sets are skipped.
- **Truthful status vocabulary**: `optimal` only for a KKT-verified exact
  optimum; `feasible_projection_fallback` for the sequential-projection path
  (never called optimal, only produced when `allow_projection_fallback=True`,
  default off ⇒ **stop** on exact-solver failure); `infeasible`,
  `invalid_nominal`, `degenerate_constraint` otherwise.
- **Velocity semantics fixed** (Phase 1, option C): renamed to
  `per_axis_velocity_limit_mps`, enforced as an exact per-axis box, nominal
  pre-clamped per-axis. The old `max_cartesian_velocity_mps` remains a
  documented back-compat alias denoting the same per-axis box, never a
  Euclidean ball.
- **Independent discrete-step validator** (`validate_next_state`): re-checks the
  actual `p_next` for finiteness, per-axis speed, workspace hard box, and a
  discrete control-barrier rule per obstacle (never enter the hard radius; do
  not decrease clearance to the `radius + margin` set). It is the **final
  authority** and can reject a command the QP "accepted."

Barrier semantics are documented as **continuous-time inspired, discrete-time
enforced** via the independent validator. No forward-invariance certificate is
claimed.

## Trusted oracle

Development/CI-only (`scripts/qp_oracle.py`), never imported by the runtime:

- **Feasibility classification**: SciPy `linprog` (HiGHS).
- **Optimum**: OSQP (operator-splitting QP) with `eps_abs = eps_rel = 1e-9` and
  polishing; the same linear-constraint QP expressed as `min 0.5 x'Px + q'x`
  with `P = I`, `q = -u_nom`, `l = b`, `u = +inf`.

Two independent tools are used so the custom solver is never graded by a copy of
itself.

## Thresholds (fixed BEFORE running — see `scripts/qp_cross_validation.py`)

| Threshold | Value |
|---|---|
| Objective abs / rel tolerance | `1e-6` / `1e-5` |
| Solution L2 distance tolerance | `1e-3` |
| Feasibility-classification agreement required | 100% |
| KKT-verified required for every `optimal` result | yes |
| Correctness gate (blocking) | custom never *worse* than oracle; 0 feasibility disagreements; 0 KKT failures |

## Results (this run)

Command:

```bash
python scripts/qp_cross_validation.py --generic 3000 --cbf 3000
```

Environment: Cursor Cloud VM, Python 3.10.20, numpy 1.24.4, scipy 1.10.1,
osqp 0.6.3. Full report: `results/v1.1-hardening/qp-cross-validation.json`.

| Metric | Value |
|---|---:|
| Total problems | 6000 |
| Feasibility mismatches | 0 |
| Objective mismatches (two-sided) | 0 |
| Solution-distance mismatches | 0 |
| KKT-verification failures | 0 |
| Custom worse than oracle | 0 |
| Max objective error (CBF family) | 4.7e-16 |
| Max solution distance (CBF family) | 4.7e-15 |
| Max solution distance (generic family) | 8.8e-7 |
| **Gate** | **PASS** |

An initial run used SciPy `trust-constr` as the oracle and reported ~164
objective and ~923 distance "mismatches"; inspection showed the custom solver's
objective was *lower* in every case (`j_custom < j_oracle`), i.e. `trust-constr`
stopped short of the optimum. Switching the oracle to OSQP (task-recommended)
gave machine-precision agreement, confirming the discrepancy was oracle
imprecision, not a solver defect.

## Reproduce

```bash
python -m pip install -r requirements-ci.txt   # includes scipy + osqp (dev/CI)
python scripts/qp_cross_validation.py          # exit 0 on PASS
```
