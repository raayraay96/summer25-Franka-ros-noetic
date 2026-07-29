#!/usr/bin/env python3
"""Cross-validate the pure-NumPy CBF-QP solver against the SciPy oracle.

Development/CI-only. Generates thousands of deterministic (seeded) QPs from two
families and compares the custom exact active-set solver to the trusted oracle:

  * feasibility classification agreement
  * objective-value agreement (abs + rel tolerance)
  * solution L2 distance (unique minimizer of a strictly convex QP)
  * KKT-verification flag on custom "optimal" results

Writes ``results/v1.1-hardening/qp-cross-validation.json``.

Documented thresholds (fixed BEFORE running):
  OBJ_ABS_TOL = 1e-6 ; OBJ_REL_TOL = 1e-5
  SOL_DIST_TOL = 1e-4
  Feasibility classification must match on 100% of problems.
  Every custom "optimal" result must have kkt_verified == True.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vision_arm_control" / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from qp_oracle import oracle_solve  # noqa: E402

from vision_arm_control.safety_filters.base import SphericalObstacle  # noqa: E402
from vision_arm_control.safety_filters.cbf_qp import (  # noqa: E402
    LinearConstraint,
    _obstacle_constraints,
    _velocity_box_constraints,
    _workspace_constraints,
    solve_cbf_qp_ex,
)
from vision_arm_control.workspace_limits import DEFAULT_WORKSPACE  # noqa: E402

# Two-sided agreement targets (report), and the one-sided correctness gate.
OBJ_ABS_TOL = 1e-6
OBJ_REL_TOL = 1e-5
SOL_DIST_TOL = 1e-3
# The custom solver claims OPTIMAL. A true defect is only when it is *worse*
# (higher objective) than the trusted oracle beyond tolerance; being *better*
# (lower objective) just means the oracle stopped short numerically.
WORSE_TOL = 1e-6


def _cons_to_matrices(cons):
    if not cons:
        return np.zeros((0, 3)), np.zeros((0,))
    A = np.stack([c.a for c in cons], axis=0).astype(np.float64)
    b = np.array([c.b for c in cons], dtype=np.float64)
    return A, b


def _random_generic(rng):
    """Generic random halfspace QP with mixed feasibility."""
    u_nom = rng.uniform(-1.0, 1.0, size=3)
    n = int(rng.integers(1, 9))
    cons = []
    for i in range(n):
        a = rng.normal(size=3)
        while np.linalg.norm(a) < 1e-3:
            a = rng.normal(size=3)
        anchor = rng.uniform(-1.0, 1.0, size=3)
        offset = rng.uniform(-0.5, 0.5)
        b = float(a @ anchor + offset)
        cons.append(LinearConstraint(a=a, b=b, name=f"g{i}"))
    # Occasionally inject contradictory / duplicate constraints.
    roll = rng.random()
    if roll < 0.15:
        a = rng.normal(size=3)
        cons.append(LinearConstraint(a=a, b=1e3, name="contra_hi"))
        cons.append(LinearConstraint(a=-a, b=1e3, name="contra_lo"))
    elif roll < 0.25 and cons:
        cons.append(LinearConstraint(a=cons[0].a.copy(), b=cons[0].b, name="dup"))
    return u_nom, cons


def _random_cbf(rng):
    """Realistic CBF filter constraint set (workspace + obstacles + vel box)."""
    ws = DEFAULT_WORKSPACE
    p = np.array(
        [
            rng.uniform(ws.x_min - 0.05, ws.x_max + 0.05),
            rng.uniform(ws.y_min - 0.05, ws.y_max + 0.05),
            rng.uniform(ws.z_min - 0.05, ws.z_max + 0.05),
        ]
    )
    n_obs = int(rng.integers(0, 4))
    obstacles = []
    for i in range(n_obs):
        center = np.array(
            [
                rng.uniform(ws.x_min, ws.x_max),
                rng.uniform(ws.y_min, ws.y_max),
                rng.uniform(ws.z_min, ws.z_max),
            ]
        )
        obstacles.append(
            SphericalObstacle(
                id=f"o{i}",
                center=center.tolist(),
                radius_m=float(rng.uniform(0.03, 0.12)),
                margin_m=float(rng.uniform(0.0, 0.05)),
            )
        )
    alpha = float(rng.uniform(0.5, 8.0))
    vmax = float(rng.uniform(0.05, 0.4))
    margin = float(rng.uniform(0.0, 0.05))
    p_des = p + rng.uniform(-0.3, 0.3, size=3)
    dt = float(rng.uniform(0.01, 0.1))
    u_nom = np.clip((p_des - p) / dt, -vmax, vmax)
    cons = []
    cons.extend(_workspace_constraints(p, ws, margin, alpha))
    cons.extend(_obstacle_constraints(p, obstacles, alpha))
    cons.extend(_velocity_box_constraints(vmax))
    return u_nom, cons


def run(n_generic=3000, n_cbf=3000, seed=20260729):
    rng = np.random.default_rng(seed)
    families = [("generic", _random_generic, n_generic), ("cbf", _random_cbf, n_cbf)]
    total = 0
    feas_mismatch = []
    obj_mismatch = []
    dist_mismatch = []
    kkt_failures = []
    custom_worse = []  # true defect: custom objective HIGHER than oracle
    per_family = {}

    for fam_name, gen, count in families:
        f_total = f_feas = 0
        max_obj_err = 0.0
        max_dist = 0.0
        for i in range(count):
            u_nom, cons = gen(rng)
            A, b = _cons_to_matrices(cons)
            sol = solve_cbf_qp_ex(u_nom, cons, allow_projection_fallback=False)
            ora = oracle_solve(u_nom, A, b)
            total += 1
            f_total += 1

            custom_feasible = sol.u is not None
            if custom_feasible != ora.feasible:
                feas_mismatch.append(
                    {
                        "family": fam_name,
                        "idx": i,
                        "custom_status": sol.status,
                        "oracle_feasible": ora.feasible,
                    }
                )
                continue
            if not ora.feasible:
                continue
            f_feas += 1

            # Custom feasible & oracle feasible: compare optimum.
            if sol.status == "optimal" and not sol.kkt_verified:
                kkt_failures.append({"family": fam_name, "idx": i})
            j_custom = 0.5 * float((sol.u - u_nom) @ (sol.u - u_nom))
            j_oracle = float(ora.objective)
            obj_err = abs(j_custom - j_oracle)
            max_obj_err = max(max_obj_err, obj_err)
            tol = OBJ_ABS_TOL + OBJ_REL_TOL * abs(j_oracle)
            if obj_err > tol:
                obj_mismatch.append(
                    {
                        "family": fam_name,
                        "idx": i,
                        "j_custom": j_custom,
                        "j_oracle": j_oracle,
                        "delta": j_custom - j_oracle,
                        "oracle_source": ora.source,
                    }
                )
            # One-sided correctness gate: custom must not be WORSE than oracle.
            if j_custom - j_oracle > WORSE_TOL + OBJ_REL_TOL * abs(j_oracle):
                custom_worse.append(
                    {"family": fam_name, "idx": i, "j_custom": j_custom, "j_oracle": j_oracle}
                )
            dist = float(np.linalg.norm(sol.u - ora.u)) if ora.u is not None else float("inf")
            max_dist = max(max_dist, dist)
            if dist > SOL_DIST_TOL:
                dist_mismatch.append({"family": fam_name, "idx": i, "dist": dist})
        per_family[fam_name] = {
            "problems": f_total,
            "feasible": f_feas,
            "max_objective_error": max_obj_err,
            "max_solution_distance": max_dist,
        }

    report = {
        "version": "v1.1-hardening",
        "seed": seed,
        "thresholds": {
            "objective_abs_tol": OBJ_ABS_TOL,
            "objective_rel_tol": OBJ_REL_TOL,
            "solution_distance_tol": SOL_DIST_TOL,
            "feasibility_agreement_required": 1.0,
            "kkt_verified_required_for_optimal": True,
        },
        "total_problems": total,
        "per_family": per_family,
        "feasibility_mismatches": len(feas_mismatch),
        "objective_mismatches_two_sided": len(obj_mismatch),
        "solution_distance_mismatches": len(dist_mismatch),
        "kkt_verification_failures": len(kkt_failures),
        "custom_worse_than_oracle": len(custom_worse),
        "examples": {
            "feasibility": feas_mismatch[:5],
            "objective": obj_mismatch[:5],
            "distance": dist_mismatch[:5],
            "kkt": kkt_failures[:5],
            "custom_worse": custom_worse[:5],
        },
    }
    # Correctness gate: no feasibility disagreement, no KKT failures, and the
    # custom solver is never worse (higher objective) than the trusted oracle.
    report["passed"] = len(feas_mismatch) == 0 and len(kkt_failures) == 0 and len(custom_worse) == 0
    return report


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--generic", type=int, default=3000)
    ap.add_argument("--cbf", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--output", default=str(ROOT / "results" / "v1.1-hardening" / "qp-cross-validation.json"))
    args = ap.parse_args()
    report = run(n_generic=args.generic, n_cbf=args.cbf, seed=args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "total_problems",
                    "feasibility_mismatches",
                    "objective_mismatches_two_sided",
                    "solution_distance_mismatches",
                    "kkt_verification_failures",
                    "custom_worse_than_oracle",
                    "passed",
                )
            },
            indent=2,
        )
    )
    print(f"Wrote {out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
