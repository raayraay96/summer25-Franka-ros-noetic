"""Tests for experimental Cartesian CBF-QP safety filter."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vision_arm_control" / "src"))

from vision_arm_control.safety_filters import cbf_qp as cbf_mod  # noqa: E402
from vision_arm_control.safety_filters.base import (  # noqa: E402
    SafetyFilterConfig,
    SphericalObstacle,
    create_safety_filter,
)
from vision_arm_control.safety_filters.cbf_qp import CBFQPFilter, solve_cbf_qp  # noqa: E402
from vision_arm_control.workspace_limits import AxisAlignedBounds  # noqa: E402


WS = AxisAlignedBounds(0.25, 0.75, -0.40, 0.40, 0.05, 0.80)


def _cfg(**kwargs):
    base = dict(
        mode="cbf_qp",
        workspace=WS,
        workspace_margin_m=0.03,
        dt=0.05,
        alpha=4.0,
        max_cartesian_velocity_mps=0.20,
        stop_on_solver_failure=True,
        obstacles=[
            SphericalObstacle(
                id="demo_sphere",
                center=[0.48, 0.0, 0.35],
                radius_m=0.10,
                margin_m=0.05,
            )
        ],
    )
    base.update(kwargs)
    return SafetyFilterConfig(**base)


def test_qp_feasible_free_space():
    f = CBFQPFilter(_cfg())
    # Move slightly inside free space
    res = f.filter([0.40, 0.0, 0.50], [0.42, 0.0, 0.50], dt=0.05)
    assert res.accepted
    assert res.position is not None
    # Truthful status: KKT-verified exact optimum only.
    assert res.solver_status == "optimal"


def test_workspace_barrier_slows_outbound():
    f = CBFQPFilter(_cfg(obstacles=[]))
    # Near x_max, try to go further out
    p = [0.74, 0.0, 0.40]
    des = [0.90, 0.0, 0.40]
    res = f.filter(p, des, dt=0.05)
    assert res.accepted
    assert res.position is not None
    assert res.position[0] <= WS.x_max + 1e-6
    assert res.intervened


def test_obstacle_barrier_intervention():
    # Start in free space (safe), drive toward the obstacle. The accepted next
    # state must INDEPENDENTLY satisfy the modeled safety set, not merely show
    # that an intervention occurred.
    f = CBFQPFilter(_cfg())
    center = np.array([0.48, 0.0, 0.35])
    p = [0.30, 0.0, 0.35]
    des = [0.60, 0.0, 0.35]
    res = f.filter(p, des, dt=0.05)
    assert res.accepted
    assert res.position is not None
    assert res.intervened
    # Hard collision boundary (radius) must never be entered.
    dist = float(np.linalg.norm(res.position - center))
    assert dist >= 0.10 - 1e-6  # radius
    # And the configured margin set must be respected from a safe start.
    assert dist >= 0.10 + 0.05 - 1e-6  # radius + margin


def test_solver_infeasible_rejects():
    # Conflicting constraints: require u_x >= 1e6 and u_x <= -1e6 style
    cons = [
        cbf_mod.LinearConstraint(a=np.array([1.0, 0.0, 0.0]), b=1e3, name="a"),
        cbf_mod.LinearConstraint(a=np.array([-1.0, 0.0, 0.0]), b=1e3, name="b"),
    ]
    u, status, _ = solve_cbf_qp(np.zeros(3), cons)
    assert u is None
    assert "infeasible" in status or status == "infeasible"


def test_solver_failure_never_passes_unsafe():
    f = CBFQPFilter(_cfg(stop_on_solver_failure=True))
    # Non-finite desired
    res = f.filter([0.4, 0.0, 0.4], [float("nan"), 0.0, 0.4])
    assert not res.accepted
    assert res.position is None


def test_reject_and_clamp_still_available():
    rej = create_safety_filter(SafetyFilterConfig(mode="reject", workspace=WS))
    r = rej.filter([0.4, 0, 0.4], [2.0, 0, 0.4])
    assert not r.accepted
    cl = create_safety_filter(SafetyFilterConfig(mode="clamp", workspace=WS))
    c = cl.filter([0.4, 0, 0.4], [2.0, 0, 0.4])
    assert c.accepted
    assert c.position[0] == pytest.approx(WS.x_max)


def test_velocity_limit_enforced():
    f = CBFQPFilter(_cfg(obstacles=[], max_cartesian_velocity_mps=0.1))
    res = f.filter([0.4, 0.0, 0.4], [0.7, 0.0, 0.4], dt=0.05)
    assert res.accepted
    assert res.velocity is not None
    assert float(np.linalg.norm(res.velocity)) <= 0.1 + 1e-6


def test_invalid_dt_rejects():
    f = CBFQPFilter(_cfg())
    res = f.filter([0.4, 0, 0.4], [0.41, 0, 0.4], dt=0.0)
    assert not res.accepted
