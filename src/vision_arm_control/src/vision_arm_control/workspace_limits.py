"""Workspace and joint-limit validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class AxisAlignedBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def contains(self, point: Sequence[float]) -> bool:
        x, y, z = float(point[0]), float(point[1]), float(point[2])
        return (
            self.x_min <= x <= self.x_max
            and self.y_min <= y <= self.y_max
            and self.z_min <= z <= self.z_max
        )

    def clamp(self, point: Sequence[float]) -> np.ndarray:
        p = np.asarray(point, dtype=np.float64).copy()
        p[0] = min(max(p[0], self.x_min), self.x_max)
        p[1] = min(max(p[1], self.y_min), self.y_max)
        p[2] = min(max(p[2], self.z_min), self.z_max)
        return p


@dataclass(frozen=True)
class JointLimits:
    lower: Tuple[float, ...]
    upper: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.lower) != len(self.upper):
            raise ValueError("lower and upper joint limits must match length")

    def contains(self, joints: Sequence[float]) -> bool:
        if len(joints) != len(self.lower):
            return False
        for q, lo, hi in zip(joints, self.lower, self.upper):
            if q < lo or q > hi:
                return False
        return True

    def clamp(self, joints: Sequence[float]) -> np.ndarray:
        q = np.asarray(joints, dtype=np.float64).copy()
        if q.shape[0] != len(self.lower):
            raise ValueError("joint vector length mismatch")
        for i, (lo, hi) in enumerate(zip(self.lower, self.upper)):
            q[i] = min(max(q[i], lo), hi)
        return q


# Approximate Panda joint limits (radians). Source: franka_ros / manufacturer docs.
# These are for software validation only; the robot controller enforces real limits.
PANDA_JOINT_LIMITS = JointLimits(
    lower=(-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973),
    upper=(2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973),
)


# Conservative Cartesian workspace in base frame (meters). EXAMPLE defaults.
DEFAULT_WORKSPACE = AxisAlignedBounds(
    x_min=0.25,
    x_max=0.75,
    y_min=-0.40,
    y_max=0.40,
    z_min=0.05,
    z_max=0.80,
)


def validate_or_clamp_position(
    point: Sequence[float],
    bounds: AxisAlignedBounds,
    mode: str = "reject",
) -> Tuple[Optional[np.ndarray], bool, str]:
    """Validate a Cartesian target.

    mode:
      - reject: return (None, False, reason) if outside bounds
      - clamp: return clamped point with accepted=True and reason if clamped

    Returns (position_or_none, accepted, reason).
    """
    p = np.asarray(point, dtype=np.float64)
    if p.shape != (3,):
        return None, False, "position must be length-3"
    if bounds.contains(p):
        return p.copy(), True, "ok"
    if mode == "clamp":
        return bounds.clamp(p), True, "clamped_to_workspace"
    return None, False, "outside_workspace"
