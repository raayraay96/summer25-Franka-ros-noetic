"""Geometric IK helpers for demo joint commands (no ROS dependency)."""
from __future__ import annotations

import math
from typing import List, Optional


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def panda_geometric_ik(x: float, y: float, z: float) -> Optional[List[float]]:
    """Approximate 7-DOF IK for demo motion (not full analytical Franka IK).

    Produces smooth, workspace-aware joint configurations for visualization /
    fake-hardware demos. Not claimed as precision industrial IK.
    """
    j1 = math.atan2(y, x)
    j1 = _clamp(j1, -2.7, 2.7)

    r = math.hypot(x, y)
    r = _clamp(r, 0.20, 0.85)
    z = _clamp(z, 0.05, 0.85)

    L1, L2 = 0.333, 0.316
    zw = z - 0.333
    d = math.hypot(r, zw)
    d = _clamp(d, 0.05, L1 + L2 - 0.02)
    cos_elbow = (L1 * L1 + L2 * L2 - d * d) / (2 * L1 * L2)
    cos_elbow = _clamp(cos_elbow, -1.0, 1.0)
    elbow = math.pi - math.acos(cos_elbow)
    j4 = _clamp(-elbow, -3.0, -0.1)

    alpha = math.atan2(zw, r)
    cos_sh = (L1 * L1 + d * d - L2 * L2) / (2 * L1 * d)
    cos_sh = _clamp(cos_sh, -1.0, 1.0)
    beta = math.acos(cos_sh)
    j2 = _clamp(-(alpha + beta) + 0.2, -1.7, 1.7)

    j3 = 0.0
    j5 = _clamp(0.5 * j2, -2.5, 2.5)
    j6 = _clamp(math.pi / 2 + 0.3 * (z - 0.4), 0.2, 3.5)
    j7 = _clamp(0.5 * j1, -2.5, 2.5)
    return [j1, j2, j3, j4, j5, j6, j7]
