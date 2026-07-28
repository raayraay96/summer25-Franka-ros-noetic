"""Experimental Cartesian CBF-QP safety filter.

Inspired by CBF-QP safety layers used in human-to-humanoid imitation
(arXiv:2604.11447). This module implements a **small, auditable kinematic**
filter on Cartesian command velocity / target increments.

Scope (exactly what is modeled):
  - Axis-aligned workspace halfspace barriers with optional margin
  - Spherical obstacle barriers
  - Command-rate / max Cartesian velocity limit
  - Minimum-change objective relative to the nominal command

Non-claims:
  - Not manufacturer-certified safety
  - Not dynamics-level CBF with full robot inertia
  - No formal forward-invariance certificate beyond the discrete model
  - Solver failure ⇒ stop/reject (never pass unsafe command)

Solver: pure-NumPy active-set QP for 3 decision variables with linear
inequalities (exhaustive small active sets + sequential projection fallback).
No OSQP/CVXPY dependency.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .base import SafetyFilterConfig, SafetyFilterResult, SphericalObstacle


@dataclass
class LinearConstraint:
    """a · u >= b"""

    a: np.ndarray
    b: float
    name: str


def _workspace_constraints(p: np.ndarray, bounds, margin: float, alpha: float) -> List[LinearConstraint]:
    """CBF inequalities for axis-aligned box: ∇h·u + α h >= 0."""
    cons: List[LinearConstraint] = []
    pairs = [
        ("x_max", np.array([-1.0, 0.0, 0.0]), float(bounds.x_max - margin) - p[0]),
        ("x_min", np.array([1.0, 0.0, 0.0]), p[0] - float(bounds.x_min + margin)),
        ("y_max", np.array([0.0, -1.0, 0.0]), float(bounds.y_max - margin) - p[1]),
        ("y_min", np.array([0.0, 1.0, 0.0]), p[1] - float(bounds.y_min + margin)),
        ("z_max", np.array([0.0, 0.0, -1.0]), float(bounds.z_max - margin) - p[2]),
        ("z_min", np.array([0.0, 0.0, 1.0]), p[2] - float(bounds.z_min + margin)),
    ]
    for name, grad, h in pairs:
        cons.append(LinearConstraint(a=grad, b=-alpha * h, name=f"ws_{name}"))
    return cons


def _obstacle_constraints(
    p: np.ndarray, obstacles: Sequence[SphericalObstacle], alpha: float
) -> List[LinearConstraint]:
    cons: List[LinearConstraint] = []
    for obs in obstacles:
        c = obs.center_array()
        r = float(obs.radius_m) + float(obs.margin_m)
        d = p - c
        dist = float(np.linalg.norm(d))
        if dist < 1e-9:
            # Push outward along +x as a well-defined escape direction.
            n = np.array([1.0, 0.0, 0.0])
            h = -r
        else:
            n = d / dist
            h = dist - r
        cons.append(LinearConstraint(a=n, b=-alpha * h, name=f"obs_{obs.id}"))
    return cons


def _velocity_box_constraints(vmax: float) -> List[LinearConstraint]:
    cons: List[LinearConstraint] = []
    for i, axis in enumerate("xyz"):
        e = np.zeros(3)
        e[i] = 1.0
        cons.append(LinearConstraint(a=e, b=-vmax, name=f"vmax_{axis}_lo"))
        cons.append(LinearConstraint(a=-e, b=-vmax, name=f"vmax_{axis}_hi"))
    return cons


def _is_feasible(u: np.ndarray, constraints: Sequence[LinearConstraint], tol: float = 1e-7) -> bool:
    for c in constraints:
        if float(np.dot(c.a, u) - c.b) < -tol:
            return False
    return True


def _solve_equality_active(u_nom: np.ndarray, active: Sequence[LinearConstraint]) -> Optional[np.ndarray]:
    """Solve min ||u-u_nom||^2 s.t. a_i·u = b_i for active set (0..3 eqs)."""
    k = len(active)
    if k == 0:
        return u_nom.copy()
    A = np.stack([c.a for c in active], axis=0)  # k x 3
    b = np.array([c.b for c in active], dtype=np.float64)
    # u = u_nom + A^T λ, with A(u_nom + A^T λ) = b ⇒ (A A^T) λ = b - A u_nom
    try:
        M = A @ A.T
        rhs = b - A @ u_nom
        # Use least-squares for near-singular active sets
        lam, _, rank, _ = np.linalg.lstsq(M, rhs, rcond=1e-10)
        if rank < k:
            return None
        return u_nom + A.T @ lam
    except np.linalg.LinAlgError:
        return None


def solve_cbf_qp(
    u_nom: np.ndarray, constraints: Sequence[LinearConstraint]
) -> Tuple[Optional[np.ndarray], str, List[str]]:
    """Solve min ||u - u_nom||^2 s.t. a_i · u >= b_i for u in R^3.

    Strategy:
      1. If u_nom feasible → return it
      2. Exhaustive active sets of size 1..3 (exact for QP with few constraints)
      3. Sequential halfspace projection fallback
    """
    u_nom = np.asarray(u_nom, dtype=np.float64).copy()
    if u_nom.shape != (3,) or not np.all(np.isfinite(u_nom)):
        return None, "invalid_nominal", []

    cons = list(constraints)
    if _is_feasible(u_nom, cons):
        return u_nom, "solved", []

    best_u = None
    best_cost = float("inf")
    best_active: List[str] = []

    # Prefer violated constraints at u_nom; fall back to all if needed.
    violated_idx = [i for i, c in enumerate(cons) if float(np.dot(c.a, u_nom) - c.b) < -1e-9]
    candidate_sets = [violated_idx] if violated_idx else [list(range(len(cons)))]
    if violated_idx and len(violated_idx) < len(cons):
        candidate_sets.append(list(range(len(cons))))

    for pool in candidate_sets:
        n = len(pool)
        max_k = min(3, n)
        for k in range(1, max_k + 1):
            for idxs in itertools.combinations(pool, k):
                active = [cons[i] for i in idxs]
                u_cand = _solve_equality_active(u_nom, active)
                if u_cand is None or not np.all(np.isfinite(u_cand)):
                    continue
                if not _is_feasible(u_cand, cons, tol=1e-6):
                    continue
                cost = float(np.dot(u_cand - u_nom, u_cand - u_nom))
                if cost < best_cost:
                    best_cost = cost
                    best_u = u_cand
                    best_active = [c.name for c in active]
        if best_u is not None:
            break

    if best_u is not None:
        return best_u, "solved", best_active

    # Sequential projection fallback (find any feasible near u_nom)
    u = u_nom.copy()
    active_names: List[str] = []
    for _ in range(80):
        worst = None
        worst_val = 0.0
        for c in cons:
            val = float(np.dot(c.a, u) - c.b)
            if val < worst_val:
                worst_val = val
                worst = c
        if worst is None:
            return u, "solved_projection", active_names
        anorm2 = float(np.dot(worst.a, worst.a))
        if anorm2 < 1e-18:
            return None, "degenerate_constraint", active_names
        u = u + ((worst.b - np.dot(worst.a, u)) / anorm2) * worst.a
        if worst.name not in active_names:
            active_names.append(worst.name)

    if _is_feasible(u, cons, tol=1e-5):
        return u, "solved_projection_max_iter", active_names
    return None, "infeasible", active_names


class CBFQPFilter:
    """Experimental kinematic Cartesian CBF-QP filter."""

    def __init__(self, config: Optional[SafetyFilterConfig] = None) -> None:
        self.config = config or SafetyFilterConfig(mode="cbf_qp")
        self.name = "cbf_qp"
        self._last_position: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._last_position = None

    def filter(
        self,
        current_position: Sequence[float],
        desired_position: Sequence[float],
        dt: Optional[float] = None,
    ) -> SafetyFilterResult:
        t0 = time.perf_counter()
        cfg = self.config
        dt_use = float(cfg.dt if dt is None else dt)
        if dt_use <= 0.0:
            elapsed = time.perf_counter() - t0
            return SafetyFilterResult(
                accepted=False,
                position=None,
                velocity=None,
                reason="invalid_dt",
                mode=self.name,
                intervened=True,
                solver_status="error",
                compute_time_s=elapsed,
            )

        p = np.asarray(current_position, dtype=np.float64).reshape(3)
        p_des = np.asarray(desired_position, dtype=np.float64).reshape(3)
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(p_des))):
            elapsed = time.perf_counter() - t0
            return SafetyFilterResult(
                accepted=False,
                position=None,
                velocity=None,
                reason="non_finite_input",
                mode=self.name,
                intervened=True,
                solver_status="error",
                compute_time_s=elapsed,
            )

        u_nom = (p_des - p) / dt_use
        vmax = float(cfg.max_cartesian_velocity_mps)
        speed = float(np.linalg.norm(u_nom))
        if speed > vmax and speed > 1e-12:
            u_nom = u_nom * (vmax / speed)

        cons: List[LinearConstraint] = []
        cons.extend(_workspace_constraints(p, cfg.workspace, cfg.workspace_margin_m, cfg.alpha))
        cons.extend(_obstacle_constraints(p, cfg.obstacles, cfg.alpha))
        cons.extend(_velocity_box_constraints(vmax))

        u_star, status, active = solve_cbf_qp(u_nom, cons)
        elapsed = time.perf_counter() - t0

        if u_star is None:
            # Always reject on failure — never pass unsafe command.
            return SafetyFilterResult(
                accepted=False,
                position=None,
                velocity=None,
                reason=f"solver_failure:{status}",
                mode=self.name,
                intervened=True,
                active_constraints=active,
                solver_status=status,
                compute_time_s=elapsed,
                constraint_violations=1,
            )

        p_next = p + u_star * dt_use
        interv_mag = float(np.linalg.norm(u_star - u_nom))
        intervened = interv_mag > 1e-9 or len(active) > 0

        violations = 0
        for c in cons:
            if float(np.dot(c.a, u_star) - c.b) < -1e-6:
                violations += 1

        self._last_position = p_next.copy()
        return SafetyFilterResult(
            accepted=True,
            position=p_next,
            velocity=u_star,
            reason="ok" if not intervened else "cbf_intervened",
            mode=self.name,
            intervened=intervened,
            intervention_magnitude=interv_mag,
            active_constraints=active,
            solver_status=status,
            compute_time_s=elapsed,
            constraint_violations=violations,
        )
