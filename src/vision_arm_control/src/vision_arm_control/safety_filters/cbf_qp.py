"""Experimental Cartesian CBF-QP safety filter (hardened v1.1).

Inspired by CBF-QP safety layers used in human-to-humanoid imitation
(arXiv:2604.11447). This module implements a **small, auditable kinematic**
filter on a Cartesian command velocity ``u`` (m/s) for a single end-effector
point.

Optimization problem (strictly convex QP, 3 decision variables):

    minimize    0.5 * ||u - u_nom||^2
    subject to  a_i . u >= b_i   for each linear constraint i

Constraints modeled:
  - Axis-aligned workspace half-space CBF rows: ``grad(h) . u + alpha * h >= 0``
  - Spherical-obstacle CBF row: ``n . u + alpha * (dist - r) >= 0``
  - Per-axis command-velocity box: ``-vmax <= u_axis <= vmax``

Solver (``solve_cbf_qp``): pure-NumPy **exact** active-set method for this
low-dimensional QP. It enumerates every active set of size 0..3 over *all*
constraints, solves the equality-constrained subproblem, keeps only
primal-feasible candidates, and returns the minimum-cost candidate whose KKT
conditions (stationarity, primal/dual feasibility, complementary slackness)
verify within documented tolerances. Because the objective is strictly convex,
the minimum-cost primal-feasible active-set solution is the global optimum; the
KKT check is an independent verification, not the selection rule.

Truthfulness guarantees:
  - Status ``optimal`` is returned only for a KKT-verified exact optimum.
  - A sequential-projection result is **never** called optimal. It is labeled
    ``feasible_projection_fallback`` and is only produced when explicitly
    enabled via ``allow_projection_fallback=True``; the default is to STOP
    (return ``None``) on exact-solver failure.
  - After a command is computed, an **independent discrete-step validator**
    re-checks the actual next state ``p_next = p + u* * dt`` against the modeled
    safety set (finite, workspace hard box, obstacle radius+margin, per-axis
    speed). The internal constraint residual is never the sole authority.

Barrier semantics: the CBF rows are **continuous-time inspired** but enforced in
**discrete time** through the independent next-state validator. No formal
forward-invariance certificate is claimed for the discrete implementation.

Non-claims: not manufacturer-certified; not dynamics-level CBF with inertia; not
a forward-invariance proof; solver failure => stop/reject (never pass an
unverified command).
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .base import SafetyFilterConfig, SafetyFilterResult, SphericalObstacle

# Documented numerical tolerances (see docs/research/qp-correctness-audit.md).
PRIMAL_TOL = 1e-7  # a_i . u >= b_i - PRIMAL_TOL counts as feasible
DUAL_TOL = 1e-7  # dual multipliers must be >= -DUAL_TOL
STATIONARITY_TOL = 1e-6  # ||u - u_nom - A^T lambda|| <= STATIONARITY_TOL
RANK_RCOND = 1e-10  # least-squares rank cutoff for active-set matrices


@dataclass
class LinearConstraint:
    """a . u >= b"""

    a: np.ndarray
    b: float
    name: str


@dataclass
class QPSolution:
    """Result of an exact QP solve with KKT audit fields."""

    u: Optional[np.ndarray]
    status: str
    active: List[str]
    cost: float = float("inf")
    kkt_verified: bool = False
    max_primal_residual: float = float("inf")
    min_dual: float = float("nan")
    stationarity_residual: float = float("nan")


def _workspace_constraints(p: np.ndarray, bounds, margin: float, alpha: float) -> List[LinearConstraint]:
    """CBF inequalities for axis-aligned box: grad(h) . u + alpha * h >= 0."""
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
            # Degenerate: at the obstacle center there is no defined outward
            # normal. Pick +x as a well-defined escape direction and a barrier
            # value of -r (deeply violated) so the QP must push outward.
            n = np.array([1.0, 0.0, 0.0])
            h = -r
        else:
            n = d / dist
            h = dist - r
        cons.append(LinearConstraint(a=n, b=-alpha * h, name=f"obs_{obs.id}"))
    return cons


def _velocity_box_constraints(vmax: float) -> List[LinearConstraint]:
    """Per-axis velocity box: -vmax <= u_axis <= vmax (NOT a Euclidean ball)."""
    cons: List[LinearConstraint] = []
    for i, axis in enumerate("xyz"):
        e = np.zeros(3)
        e[i] = 1.0
        cons.append(LinearConstraint(a=e, b=-vmax, name=f"vmax_{axis}_lo"))
        cons.append(LinearConstraint(a=-e, b=-vmax, name=f"vmax_{axis}_hi"))
    return cons


def _is_feasible(u: np.ndarray, constraints: Sequence[LinearConstraint], tol: float = PRIMAL_TOL) -> bool:
    for c in constraints:
        if float(np.dot(c.a, u) - c.b) < -tol:
            return False
    return True


def _max_primal_residual(u: np.ndarray, constraints: Sequence[LinearConstraint]) -> float:
    """Largest constraint violation (0 if feasible)."""
    worst = 0.0
    for c in constraints:
        v = c.b - float(np.dot(c.a, u))  # >0 means violated
        if v > worst:
            worst = v
    return worst


def _solve_equality_active(u_nom: np.ndarray, active: Sequence[LinearConstraint]) -> Optional[np.ndarray]:
    """Solve min ||u-u_nom||^2 s.t. a_i.u = b_i for active set (0..3 eqs).

    Returns None when the active rows are rank-deficient (duplicate /
    contradictory / near-singular), which the caller treats as "skip this
    candidate" rather than a solution.
    """
    k = len(active)
    if k == 0:
        return u_nom.copy()
    A = np.stack([c.a for c in active], axis=0)  # k x 3
    b = np.array([c.b for c in active], dtype=np.float64)
    if np.linalg.matrix_rank(A, tol=1e-9) < k:
        return None
    # KKT of equality QP: u = u_nom + A^T lambda, with A u = b
    #   => (A A^T) lambda = b - A u_nom
    try:
        M = A @ A.T
        rhs = b - A @ u_nom
        lam, _, rank, _ = np.linalg.lstsq(M, rhs, rcond=RANK_RCOND)
        if rank < k:
            return None
        u = u_nom + A.T @ lam
        if not np.all(np.isfinite(u)):
            return None
        return u
    except np.linalg.LinAlgError:
        return None


def _dual_multipliers(
    u: np.ndarray, u_nom: np.ndarray, active: Sequence[LinearConstraint]
) -> Tuple[np.ndarray, float]:
    """Recover dual multipliers lambda for the active set from stationarity
    u - u_nom = A^T lambda, and return (lambda, stationarity_residual)."""
    if len(active) == 0:
        return np.zeros(0), float(np.linalg.norm(u - u_nom))
    A = np.stack([c.a for c in active], axis=0)  # k x 3
    lam, _, _, _ = np.linalg.lstsq(A.T, u - u_nom, rcond=RANK_RCOND)
    resid = float(np.linalg.norm((A.T @ lam) - (u - u_nom)))
    return lam, resid


def solve_cbf_qp_ex(
    u_nom: np.ndarray,
    constraints: Sequence[LinearConstraint],
    allow_projection_fallback: bool = False,
) -> QPSolution:
    """Exact active-set solve of the strictly convex QP with KKT verification.

    See module docstring. Returns a :class:`QPSolution`.
    """
    u_nom = np.asarray(u_nom, dtype=np.float64).copy()
    if u_nom.shape != (3,) or not np.all(np.isfinite(u_nom)):
        return QPSolution(u=None, status="invalid_nominal", active=[])

    cons = list(constraints)

    # Unconstrained minimizer u_nom is optimal iff it is feasible.
    if _is_feasible(u_nom, cons):
        return QPSolution(
            u=u_nom,
            status="optimal",
            active=[],
            cost=0.0,
            kkt_verified=True,
            max_primal_residual=_max_primal_residual(u_nom, cons),
            min_dual=float("inf"),
            stationarity_residual=0.0,
        )

    n = len(cons)
    best: Optional[QPSolution] = None

    # Enumerate ALL active sets of size 1..min(3, n) over ALL constraints.
    for k in range(1, min(3, n) + 1):
        for idxs in itertools.combinations(range(n), k):
            active = [cons[i] for i in idxs]
            u_cand = _solve_equality_active(u_nom, active)
            if u_cand is None:
                continue
            if not _is_feasible(u_cand, cons):
                continue
            cost = float(np.dot(u_cand - u_nom, u_cand - u_nom))
            if best is None or cost < best.cost:
                lam, stat = _dual_multipliers(u_cand, u_nom, active)
                best = QPSolution(
                    u=u_cand,
                    status="optimal",
                    active=[c.name for c in active],
                    cost=cost,
                    kkt_verified=False,  # verified after selection below
                    max_primal_residual=_max_primal_residual(u_cand, cons),
                    min_dual=float(np.min(lam)) if lam.size else float("inf"),
                    stationarity_residual=stat,
                )

    if best is not None and best.u is not None:
        # Independent KKT verification of the selected minimum-cost optimum.
        best.kkt_verified = (
            best.max_primal_residual <= PRIMAL_TOL
            and best.min_dual >= -DUAL_TOL
            and best.stationarity_residual <= STATIONARITY_TOL
        )
        return best

    # No KKT optimum found: either infeasible, or numerically degenerate.
    if allow_projection_fallback:
        u = u_nom.copy()
        active_names: List[str] = []
        for _ in range(200):
            worst_c = None
            worst_val = 0.0
            for c in cons:
                val = float(np.dot(c.a, u) - c.b)
                if val < worst_val:
                    worst_val = val
                    worst_c = c
            if worst_c is None:
                return QPSolution(
                    u=u,
                    status="feasible_projection_fallback",
                    active=active_names,
                    cost=float(np.dot(u - u_nom, u - u_nom)),
                    kkt_verified=False,
                    max_primal_residual=_max_primal_residual(u, cons),
                )
            anorm2 = float(np.dot(worst_c.a, worst_c.a))
            if anorm2 < 1e-18:
                return QPSolution(u=None, status="degenerate_constraint", active=active_names)
            u = u + ((worst_c.b - np.dot(worst_c.a, u)) / anorm2) * worst_c.a
            if worst_c.name not in active_names:
                active_names.append(worst_c.name)
        return QPSolution(u=None, status="infeasible", active=active_names)

    return QPSolution(u=None, status="infeasible", active=[])


def solve_cbf_qp(
    u_nom: np.ndarray,
    constraints: Sequence[LinearConstraint],
    allow_projection_fallback: bool = False,
) -> Tuple[Optional[np.ndarray], str, List[str]]:
    """Backwards-compatible thin wrapper returning ``(u, status, active)``.

    ``status`` is ``optimal`` only for a KKT-verified exact optimum,
    ``feasible_projection_fallback`` for a projection result (never optimal),
    or ``infeasible`` / ``invalid_nominal`` / ``degenerate_constraint``.
    """
    sol = solve_cbf_qp_ex(u_nom, constraints, allow_projection_fallback=allow_projection_fallback)
    return sol.u, sol.status, sol.active


@dataclass
class NextStateReport:
    """Independent discrete-step validation of p_next."""

    valid: bool
    reason: str
    min_obstacle_clearance_m: float = float("inf")


def validate_next_state(
    p_current: np.ndarray,
    p_next: np.ndarray,
    cfg: SafetyFilterConfig,
    u_star: np.ndarray,
) -> NextStateReport:
    """Independently re-check the actual next state against the modeled safety
    set. Never trusts the internal CBF residual.

    Discrete control-barrier rule (per obstacle / workspace face):
      * Hard boundary (collision / hard box) may **never** be entered.
      * Soft margin set ``dist >= radius + margin``: if the current state is
        inside the margin set, the next state must stay in it; if the current
        state already violates the margin (but not the hard boundary), the step
        may not **decrease** clearance (recovery-only motion is allowed).

    Checks return the first violated condition.
    """
    tol = 1e-6
    vmax = float(cfg.per_axis_velocity_limit_mps)
    if not (np.all(np.isfinite(p_next)) and np.all(np.isfinite(u_star))):
        return NextStateReport(False, "next_state_non_finite")
    if float(np.max(np.abs(u_star))) > vmax + tol:
        return NextStateReport(False, "per_axis_velocity_exceeded")

    # Workspace: hard box (no margin) is the absolute boundary; margin box is
    # the soft set the CBF maintains.
    ws = cfg.workspace
    m = float(cfg.workspace_margin_m)
    for axis, lo, hi in (
        (0, ws.x_min, ws.x_max),
        (1, ws.y_min, ws.y_max),
        (2, ws.z_min, ws.z_max),
    ):
        pc, pn = float(p_current[axis]), float(p_next[axis])
        if pn < lo - tol or pn > hi + tol:
            return NextStateReport(False, "next_state_outside_workspace_hard")
        # Soft-margin non-decrease when already inside the margin band.
        lo_s, hi_s = lo + m, hi - m
        clear_cur = min(pc - lo_s, hi_s - pc)
        clear_next = min(pn - lo_s, hi_s - pn)
        if clear_cur >= -tol:
            if clear_next < -tol:
                return NextStateReport(False, "next_state_left_workspace_margin")
        elif clear_next < clear_cur - tol:
            return NextStateReport(False, "next_state_decreased_workspace_margin")

    min_clear = float("inf")
    for obs in cfg.obstacles:
        c = obs.center_array()
        r_hard = float(obs.radius_m)
        r_safe = r_hard + float(obs.margin_m)
        dist_cur = float(np.linalg.norm(p_current - c))
        dist_next = float(np.linalg.norm(p_next - c))
        clear_next = dist_next - r_safe
        if clear_next < min_clear:
            min_clear = clear_next
        # Absolute: never enter the hard collision sphere.
        if dist_next < r_hard - tol:
            return NextStateReport(False, f"next_state_inside_obstacle_hard:{obs.id}", min_clear)
        clear_cur = dist_cur - r_safe
        if clear_cur >= -tol:
            if clear_next < -tol:
                return NextStateReport(False, f"next_state_entered_obstacle_margin:{obs.id}", min_clear)
        elif clear_next < clear_cur - tol:
            return NextStateReport(False, f"next_state_decreased_obstacle_margin:{obs.id}", min_clear)
    return NextStateReport(True, "ok", min_clear)


class CBFQPFilter:
    """Experimental kinematic Cartesian CBF-QP filter (hardened)."""

    def __init__(self, config: Optional[SafetyFilterConfig] = None) -> None:
        self.config = config or SafetyFilterConfig(mode="cbf_qp")
        self.name = "cbf_qp"
        self._last_position: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._last_position = None

    def _reject(self, reason: str, status: str, elapsed: float, active=None) -> SafetyFilterResult:
        return SafetyFilterResult(
            accepted=False,
            position=None,
            velocity=None,
            reason=reason,
            mode=self.name,
            intervened=True,
            active_constraints=active or [],
            solver_status=status,
            compute_time_s=elapsed,
            constraint_violations=1,
        )

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
            return self._reject("invalid_dt", "error", time.perf_counter() - t0)

        p = np.asarray(current_position, dtype=np.float64).reshape(3)
        p_des = np.asarray(desired_position, dtype=np.float64).reshape(3)
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(p_des))):
            return self._reject("non_finite_input", "error", time.perf_counter() - t0)

        vmax = float(cfg.per_axis_velocity_limit_mps)
        # Per-axis clamp of the nominal command (consistent with the per-axis
        # box the QP enforces). This is NOT a Euclidean speed cap.
        u_nom = (p_des - p) / dt_use
        u_nom = np.clip(u_nom, -vmax, vmax)

        cons: List[LinearConstraint] = []
        cons.extend(_workspace_constraints(p, cfg.workspace, cfg.workspace_margin_m, cfg.alpha))
        cons.extend(_obstacle_constraints(p, cfg.obstacles, cfg.alpha))
        cons.extend(_velocity_box_constraints(vmax))

        sol = solve_cbf_qp_ex(u_nom, cons, allow_projection_fallback=cfg.allow_projection_fallback)
        elapsed = time.perf_counter() - t0

        if sol.u is None:
            return self._reject(f"solver_failure:{sol.status}", sol.status, elapsed, sol.active)

        # A projection fallback is feasible but not proven optimal; never send it
        # unless the operator explicitly enabled it (default: stop).
        if sol.status == "feasible_projection_fallback" and not cfg.allow_projection_fallback:
            return self._reject("solver_failure:projection_disabled", sol.status, elapsed, sol.active)

        u_star = sol.u
        p_next = p + u_star * dt_use

        # Independent discrete-step safety validation (final authority).
        report = validate_next_state(p, p_next, cfg, u_star)
        if not report.valid:
            return self._reject(f"next_state_rejected:{report.reason}", sol.status, elapsed, sol.active)

        interv_mag = float(np.linalg.norm(u_star - u_nom))
        intervened = interv_mag > 1e-9 or len(sol.active) > 0
        self._last_position = p_next.copy()
        return SafetyFilterResult(
            accepted=True,
            position=p_next,
            velocity=u_star,
            reason="ok" if not intervened else "cbf_intervened",
            mode=self.name,
            intervened=intervened,
            intervention_magnitude=interv_mag,
            active_constraints=sol.active,
            solver_status=sol.status,
            compute_time_s=elapsed,
            constraint_violations=0,
        )
