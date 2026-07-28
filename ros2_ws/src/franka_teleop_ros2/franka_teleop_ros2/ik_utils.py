"""Geometric IK helpers for demo joint commands (no ROS dependency).

This is an approximation for RViz fake-hardware demos only — not industrial
Franka IK / not MoveIt. Tuned so the visual Panda URDF tip tracks the
commanded Cartesian target closely enough for a recruiter-facing demo.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# Approximate Panda lengths (m) for a planar 2R arm + base height.
# Chosen to match moveit_resources_panda visuals better than a pure toy model.
L_BASE = 0.333  # base → shoulder height contribution
L1 = 0.316  # upper arm
L2 = 0.384  # forearm + wrist reach proxy
L_EE = 0.107  # flange / EE offset along last link


def panda_geometric_ik(x: float, y: float, z: float) -> Optional[List[float]]:
    """Approximate 7-DOF IK for demo motion (not full analytical Franka IK).

    Produces smooth, workspace-aware joint configurations for visualization /
    fake-hardware demos. Not claimed as precision industrial IK.
    """
    j1 = math.atan2(y, x)
    j1 = _clamp(j1, -2.7, 2.7)

    r = math.hypot(x, y)
    r = _clamp(r, 0.15, 0.90)
    z = _clamp(z, 0.05, 0.95)

    # Work in the shoulder plane: subtract base height, aim slightly short of
    # true EE so the visual tip lands near the marker.
    zw = z - L_BASE
    # Reach in plane (account for small EE offset)
    reach = max(0.05, r - 0.02)
    d = math.hypot(reach, zw)
    d = _clamp(d, 0.08, L1 + L2 - 0.02)

    cos_elbow = (L1 * L1 + L2 * L2 - d * d) / (2.0 * L1 * L2)
    cos_elbow = _clamp(cos_elbow, -1.0, 1.0)
    # Elbow-down configuration for a natural Panda look
    elbow = math.acos(cos_elbow)
    j4 = _clamp(-(math.pi - elbow), -3.0, -0.05)

    alpha = math.atan2(zw, reach)
    cos_sh = (L1 * L1 + d * d - L2 * L2) / (2.0 * L1 * d)
    cos_sh = _clamp(cos_sh, -1.0, 1.0)
    beta = math.acos(cos_sh)
    # joint2 sign convention for panda (negative pitches the arm forward)
    j2 = _clamp(-(alpha + beta), -1.75, 1.75)

    j3 = 0.0
    # Wrist follows so the hand faces roughly downward/outward
    j5 = _clamp(0.35 * j1, -2.5, 2.5)
    j6 = _clamp(math.pi / 2 + 0.4 * (z - 0.40) - 0.15 * j2, 0.15, 3.5)
    j7 = _clamp(0.4 * j1, -2.5, 2.5)
    return [j1, j2, j3, j4, j5, j6, j7]


def panda_geometric_fk(joints: Sequence[float]) -> Tuple[float, float, float]:
    """Forward kinematics matching ``panda_geometric_ik`` (for EE markers)."""
    j1, j2, j3, j4, j5, j6, j7 = [float(v) for v in list(joints)[:7]]
    # Planar arm in the j1 plane
    # After j2 and j4 (elbow), tip distance in shoulder plane:
    # Using law of cosines layout consistent with IK
    # Shoulder angle from horizontal
    # Reconstruct r,z from j2 and elbow geometry
    # j2 = -(alpha + beta)  =>  alpha + beta = -j2
    # This is approximate; good enough for marker placement.
    th2 = -j2
    th4 = j4
    # Upper arm direction after shoulder
    ux = math.sin(th2) * L1
    uz = math.cos(th2) * L1
    # Forearm after elbow (relative)
    th_f = th2 + th4
    fx = math.sin(th_f) * L2
    fz = math.cos(th_f) * L2
    r = ux + fx + 0.02
    z = L_BASE + uz + fz
    x = r * math.cos(j1)
    y = r * math.sin(j1)
    return float(x), float(y), float(z)
