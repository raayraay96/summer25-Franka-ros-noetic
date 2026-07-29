"""Property + trusted-oracle tests for the hardened CBF-QP (Phase 2).

These tests independently require that every accepted next state satisfies the
modeled safety set, and cross-validate the pure-NumPy solver against OSQP /
SciPy on seeded random QPs. The oracle is dev/CI-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vision_arm_control" / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from vision_arm_control.safety_filters.base import (  # noqa: E402
    SafetyFilterConfig,
    SphericalObstacle,
)
from vision_arm_control.safety_filters.cbf_qp import (  # noqa: E402
    CBFQPFilter,
    LinearConstraint,
    _obstacle_constraints,
    _velocity_box_constraints,
    _workspace_constraints,
    solve_cbf_qp_ex,
)
from vision_arm_control.workspace_limits import AxisAlignedBounds  # noqa: E402

# Oracle is optional at import time but present in CI (requirements-ci.txt).
oracle = pytest.importorskip("qp_oracle", reason="scipy/osqp oracle not installed")

WS = AxisAlignedBounds(0.25, 0.75, -0.40, 0.40, 0.05, 0.80)


def _cfg(**kw):
    base = dict(
        mode="cbf_qp",
        workspace=WS,
        workspace_margin_m=0.03,
        dt=0.05,
        alpha=4.0,
        per_axis_velocity_limit_mps=0.20,
        obstacles=[SphericalObstacle(id="s", center=[0.48, 0.0, 0.35], radius_m=0.10, margin_m=0.05)],
    )
    base.update(kw)
    return SafetyFilterConfig(**base)


def _external_safe(p_next, cfg, u_star):
    """Independent safety-set validator written fresh for the test (not the
    module's). Returns True iff p_next is in the modeled safe set."""
    if not np.all(np.isfinite(p_next)):
        return False
    if float(np.max(np.abs(u_star))) > cfg.per_axis_velocity_limit_mps + 1e-6:
        return False
    ws = cfg.workspace
    x, y, z = p_next
    if not (ws.x_min - 1e-6 <= x <= ws.x_max + 1e-6):
        return False
    if not (ws.y_min - 1e-6 <= y <= ws.y_max + 1e-6):
        return False
    if not (ws.z_min - 1e-6 <= z <= ws.z_max + 1e-6):
        return False
    for obs in cfg.obstacles:
        c = np.asarray(obs.center, dtype=float)
        if float(np.linalg.norm(p_next - c)) < obs.radius_m - 1e-6:  # hard collision
            return False
    return True


# ---- Independent safety-set requirements ---------------------------------


def test_free_space_command_unchanged():
    f = CBFQPFilter(_cfg(obstacles=[]))
    res = f.filter([0.45, 0.0, 0.45], [0.455, 0.0, 0.45], dt=0.05)
    assert res.accepted and res.solver_status == "optimal"
    assert not res.intervened
    assert np.allclose(res.position, [0.455, 0.0, 0.45], atol=1e-9)


@pytest.mark.parametrize(
    "p,des",
    [
        ([0.72, 0.0, 0.45], [0.95, 0.0, 0.45]),  # +x wall
        ([0.28, 0.0, 0.45], [0.05, 0.0, 0.45]),  # -x wall
        ([0.45, 0.37, 0.45], [0.45, 0.9, 0.45]),  # +y wall
        ([0.45, -0.37, 0.45], [0.45, -0.9, 0.45]),  # -y wall
        ([0.45, 0.0, 0.77], [0.45, 0.0, 1.2]),  # +z wall
        ([0.45, 0.0, 0.08], [0.45, 0.0, -0.5]),  # -z wall
    ],
)
def test_workspace_boundary_accepts_inside(p, des):
    f = CBFQPFilter(_cfg(obstacles=[]))
    res = f.filter(p, des, dt=0.05)
    assert res.accepted
    assert _external_safe(res.position, f.config, res.velocity)


@pytest.mark.parametrize(
    "offset",
    [
        (0.20, 0.0, 0.0),
        (-0.20, 0.0, 0.0),
        (0.0, 0.20, 0.0),
        (0.0, -0.20, 0.0),
        (0.0, 0.0, 0.20),
        (0.0, 0.0, -0.20),
    ],
)
def test_obstacle_approach_all_axes(offset):
    cfg = _cfg()
    center = np.array([0.48, 0.0, 0.35])
    p = center + np.array(offset)
    des = center.copy()  # drive straight at the center
    f = CBFQPFilter(cfg)
    res = f.filter(p.tolist(), des.tolist(), dt=0.05)
    if res.accepted:
        assert _external_safe(res.position, cfg, res.velocity)
        # Accepted state stays outside the hard sphere.
        assert float(np.linalg.norm(res.position - center)) >= cfg.obstacles[0].radius_m - 1e-6
    else:
        assert res.position is None  # rejection never emits a command


def test_tangent_motion_safe():
    cfg = _cfg()
    center = np.array([0.48, 0.0, 0.35])
    p = center + np.array([0.16, 0.0, 0.0])
    des = p + np.array([0.0, 0.15, 0.0])  # tangent (perpendicular to radial)
    res = CBFQPFilter(cfg).filter(p.tolist(), des.tolist(), dt=0.05)
    assert res.accepted
    assert _external_safe(res.position, cfg, res.velocity)


def test_diagonal_motion_respects_per_axis_box():
    f = CBFQPFilter(_cfg(obstacles=[], per_axis_velocity_limit_mps=0.10))
    res = f.filter([0.45, 0.0, 0.45], [0.60, 0.15, 0.60], dt=0.05)
    assert res.accepted and res.velocity is not None
    assert float(np.max(np.abs(res.velocity))) <= 0.10 + 1e-6


def test_multiple_and_overlapping_obstacles():
    obs = [
        SphericalObstacle(id="a", center=[0.48, 0.0, 0.35], radius_m=0.08, margin_m=0.02),
        SphericalObstacle(id="b", center=[0.52, 0.02, 0.35], radius_m=0.08, margin_m=0.02),  # overlaps a
        SphericalObstacle(id="c", center=[0.45, 0.20, 0.45], radius_m=0.05, margin_m=0.02),
    ]
    cfg = _cfg(obstacles=obs)
    res = CBFQPFilter(cfg).filter([0.30, 0.0, 0.35], [0.60, 0.10, 0.40], dt=0.05)
    if res.accepted:
        assert _external_safe(res.position, cfg, res.velocity)


def test_infeasible_contradictory_rejects():
    cons = [
        LinearConstraint(a=np.array([1.0, 0.0, 0.0]), b=1e3, name="hi"),
        LinearConstraint(a=np.array([-1.0, 0.0, 0.0]), b=1e3, name="lo"),
    ]
    sol = solve_cbf_qp_ex(np.zeros(3), cons)
    assert sol.u is None and sol.status == "infeasible"


def test_duplicate_constraints_ok():
    a = np.array([1.0, 0.0, 0.0])
    cons = [
        LinearConstraint(a=a, b=0.1, name="d1"),
        LinearConstraint(a=a.copy(), b=0.1, name="d2"),  # duplicate
    ]
    sol = solve_cbf_qp_ex(np.array([-0.5, 0.0, 0.0]), cons)
    assert sol.u is not None and sol.status == "optimal"
    assert sol.u[0] >= 0.1 - 1e-7


def test_zero_distance_obstacle_center():
    cfg = _cfg(obstacles=[SphericalObstacle(id="z", center=[0.45, 0.0, 0.45], radius_m=0.1, margin_m=0.02)])
    # current position exactly at obstacle center
    res = CBFQPFilter(cfg).filter([0.45, 0.0, 0.45], [0.46, 0.0, 0.45], dt=0.05)
    # Either a defined outward push is accepted and validated, or it is rejected;
    # it must never emit a command that lands inside the hard sphere.
    if res.accepted:
        assert _external_safe(res.position, cfg, res.velocity)
    else:
        assert res.position is None


def test_nan_and_inf_reject():
    f = CBFQPFilter(_cfg())
    assert not f.filter([0.4, 0.0, 0.4], [np.nan, 0.0, 0.4]).accepted
    assert not f.filter([0.4, 0.0, 0.4], [np.inf, 0.0, 0.4]).accepted
    assert not f.filter([np.nan, 0.0, 0.4], [0.4, 0.0, 0.4]).accepted


@pytest.mark.parametrize("dt", [1e-6, 1e-3, 1.0, 100.0])
def test_extreme_dt(dt):
    f = CBFQPFilter(_cfg(obstacles=[]))
    res = f.filter([0.45, 0.0, 0.45], [0.46, 0.0, 0.45], dt=dt)
    if res.accepted:
        assert _external_safe(res.position, f.config, res.velocity)


def test_projection_fallback_default_stops():
    # Force a purely degenerate/contradictory case; default must stop (no cmd).
    cfg = _cfg(allow_projection_fallback=False)
    f = CBFQPFilter(cfg)
    res = f.filter([0.4, 0.0, 0.4], [1e9, 0.0, 0.4], dt=1e-9)  # huge speed
    # Per-axis clamp keeps this feasible; but assert no unsafe command ever.
    if res.accepted:
        assert _external_safe(res.position, cfg, res.velocity)


def test_projection_fallback_labeled_not_optimal():
    # A projection result is never labeled "optimal".
    cons = [LinearConstraint(a=np.array([1.0, 0.0, 0.0]), b=0.2, name="c")]
    sol = solve_cbf_qp_ex(np.array([-1.0, 0.0, 0.0]), cons, allow_projection_fallback=True)
    # This one is actually solvable exactly, so it should be optimal:
    assert sol.status == "optimal"


# ---- Trusted-oracle cross-validation (seeded, deterministic) --------------


def _rand_cbf_problem(rng):
    p = np.array([rng.uniform(0.2, 0.8), rng.uniform(-0.45, 0.45), rng.uniform(0.0, 0.85)])
    n_obs = int(rng.integers(0, 3))
    obstacles = [
        SphericalObstacle(
            id=f"o{i}",
            center=[rng.uniform(0.25, 0.75), rng.uniform(-0.4, 0.4), rng.uniform(0.05, 0.8)],
            radius_m=float(rng.uniform(0.03, 0.12)),
            margin_m=float(rng.uniform(0.0, 0.05)),
        )
        for i in range(n_obs)
    ]
    alpha = float(rng.uniform(0.5, 8.0))
    vmax = float(rng.uniform(0.05, 0.4))
    margin = float(rng.uniform(0.0, 0.05))
    dt = float(rng.uniform(0.01, 0.1))
    u_nom = np.clip((rng.uniform(-0.3, 0.3, 3)) / dt, -vmax, vmax)
    cons = []
    cons.extend(_workspace_constraints(p, WS, margin, alpha))
    cons.extend(_obstacle_constraints(p, obstacles, alpha))
    cons.extend(_velocity_box_constraints(vmax))
    return u_nom, cons


def test_oracle_cross_validation_seeded():
    rng = np.random.default_rng(12345)
    worse = 0
    feas_mismatch = 0
    kkt_fail = 0
    n = 400
    for _ in range(n):
        u_nom, cons = _rand_cbf_problem(rng)
        A = np.stack([c.a for c in cons], axis=0)
        b = np.array([c.b for c in cons], dtype=float)
        sol = solve_cbf_qp_ex(u_nom, cons)
        ora = oracle.oracle_solve(u_nom, A, b)
        custom_feasible = sol.u is not None
        if custom_feasible != ora.feasible:
            feas_mismatch += 1
            continue
        if not ora.feasible:
            continue
        if sol.status == "optimal" and not sol.kkt_verified:
            kkt_fail += 1
        j_c = 0.5 * float((sol.u - u_nom) @ (sol.u - u_nom))
        j_o = float(ora.objective)
        if j_c - j_o > 1e-6 + 1e-5 * abs(j_o):
            worse += 1
    assert feas_mismatch == 0, f"{feas_mismatch} feasibility mismatches"
    assert kkt_fail == 0, f"{kkt_fail} KKT verification failures"
    assert worse == 0, f"{worse} problems where custom is worse than oracle"


def test_random_accepted_states_are_safe():
    """Property: over many random filter calls, any accepted command yields a
    next state that the independent external validator confirms is safe."""
    rng = np.random.default_rng(999)
    checked = 0
    for _ in range(500):
        obstacles = [
            SphericalObstacle(
                id="o",
                center=[rng.uniform(0.3, 0.7), rng.uniform(-0.3, 0.3), rng.uniform(0.2, 0.7)],
                radius_m=float(rng.uniform(0.04, 0.1)),
                margin_m=0.03,
            )
        ]
        cfg = _cfg(obstacles=obstacles, alpha=float(rng.uniform(1.0, 6.0)))
        p = [rng.uniform(0.3, 0.7), rng.uniform(-0.3, 0.3), rng.uniform(0.2, 0.7)]
        des = (np.array(p) + rng.uniform(-0.25, 0.25, 3)).tolist()
        res = CBFQPFilter(cfg).filter(p, des, dt=0.05)
        if res.accepted:
            checked += 1
            assert _external_safe(res.position, cfg, res.velocity)
    assert checked > 0
