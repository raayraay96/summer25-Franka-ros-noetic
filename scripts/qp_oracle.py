#!/usr/bin/env python3
"""Trusted QP oracle for cross-validating the pure-NumPy CBF-QP solver.

Development/CI-only. Two independent trusted tools are used so the custom solver
is never "graded by itself":

  * feasibility classification  -> SciPy ``linprog`` (HiGHS, exact simplex/IPM)
  * strictly convex optimum     -> OSQP (operator-splitting QP, polished)

The runtime filter never imports this module.

QP: minimize 0.5*||u - u_nom||^2  s.t.  A u >= b   (u in R^3)
Expressed for OSQP as: minimize 0.5 x'Px + q'x  s.t.  l <= A x <= u,
with P = I, q = -u_nom, l = b, u = +inf.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import osqp
import scipy.sparse as sp
from scipy.optimize import linprog


@dataclass
class OracleResult:
    feasible: bool
    u: Optional[np.ndarray]
    objective: float
    source: str = "osqp"


def _feasible_point(A: np.ndarray, b: np.ndarray, big: float = 1e3) -> Optional[np.ndarray]:
    """Independent feasibility classifier + Phase-1 feasible point via LP."""
    n = A.shape[1]
    res = linprog(
        c=np.zeros(n),
        A_ub=-A,
        b_ub=-b,
        bounds=[(-big, big)] * n,
        method="highs",
    )
    if res.status == 0:
        return np.asarray(res.x, dtype=np.float64)
    return None


def oracle_solve(u_nom: np.ndarray, A: np.ndarray, b: np.ndarray) -> OracleResult:
    """Independently solve the QP: linprog for feasibility, OSQP for optimum."""
    u_nom = np.asarray(u_nom, dtype=np.float64).reshape(-1)
    n = u_nom.shape[0]
    A = np.asarray(A, dtype=np.float64).reshape(-1, n) if np.size(A) else np.zeros((0, n))
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    if A.shape[0] == 0:
        return OracleResult(True, u_nom.copy(), 0.0, "trivial")

    if np.all(A @ u_nom - b >= -1e-9):
        return OracleResult(True, u_nom.copy(), 0.0, "nominal")

    lp_point = _feasible_point(A, b)
    if lp_point is None:
        return OracleResult(False, None, float("inf"), "linprog_infeasible")

    P = sp.eye(n, format="csc")
    q = -u_nom
    A_sp = sp.csc_matrix(A)
    lo = b
    hi = np.full(b.shape[0], np.inf)

    prob = osqp.OSQP()
    prob.setup(
        P=P,
        q=q,
        A=A_sp,
        l=lo,
        u=hi,
        eps_abs=1e-9,
        eps_rel=1e-9,
        eps_prim_inf=1e-9,
        eps_dual_inf=1e-9,
        max_iter=200000,
        polish=True,
        polish_refine_iter=10,
        verbose=False,
    )
    res = prob.solve()
    status = res.info.status

    def obj(u):
        d = u - u_nom
        return 0.5 * float(d @ d)

    if res.x is not None and np.all(np.isfinite(res.x)) and np.all(A @ res.x - b >= -1e-6):
        u = np.asarray(res.x, dtype=np.float64)
        # Keep the better of OSQP and the LP feasible point.
        if obj(lp_point) < obj(u) and np.all(A @ lp_point - b >= -1e-6):
            return OracleResult(True, lp_point, obj(lp_point), "linprog_point")
        return OracleResult(True, u, obj(u), f"osqp:{status}")
    return OracleResult(True, lp_point, obj(lp_point), "linprog_point")
