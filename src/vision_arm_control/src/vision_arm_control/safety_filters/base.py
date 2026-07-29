"""Safety filter protocol and configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from vision_arm_control.workspace_limits import AxisAlignedBounds, DEFAULT_WORKSPACE


@dataclass
class SphericalObstacle:
    id: str
    center: Sequence[float]
    radius_m: float
    margin_m: float = 0.0

    def center_array(self) -> np.ndarray:
        return np.asarray(self.center, dtype=np.float64).reshape(3)


@dataclass
class SafetyFilterConfig:
    mode: str = "reject"  # reject | clamp | cbf_qp
    workspace: AxisAlignedBounds = field(default_factory=lambda: DEFAULT_WORKSPACE)
    workspace_margin_m: float = 0.0
    dt: float = 0.05
    alpha: float = 4.0
    # Per-axis command-velocity limit (m/s). This is enforced by the CBF-QP as a
    # per-axis box |u_axis| <= limit, which is NOT a Euclidean speed cap. The
    # old name ``max_cartesian_velocity_mps`` is accepted as a back-compat alias
    # but denoted the same per-axis box, never a Euclidean ball.
    per_axis_velocity_limit_mps: float = 0.20
    max_cartesian_velocity_mps: Optional[float] = None  # deprecated alias
    stop_on_solver_failure: bool = True
    # When False (default), an exact-solver failure => stop (return None). When
    # True, an unverified sequential-projection fallback may be used and is
    # reported as ``feasible_projection_fallback`` (never ``optimal``).
    allow_projection_fallback: bool = False
    obstacles: List[SphericalObstacle] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Honor the deprecated alias if a caller still sets it.
        if self.max_cartesian_velocity_mps is not None:
            self.per_axis_velocity_limit_mps = float(self.max_cartesian_velocity_mps)
        self.max_cartesian_velocity_mps = float(self.per_axis_velocity_limit_mps)


@dataclass
class SafetyFilterResult:
    accepted: bool
    position: Optional[np.ndarray]
    velocity: Optional[np.ndarray]
    reason: str
    mode: str
    intervened: bool = False
    intervention_magnitude: float = 0.0
    active_constraints: List[str] = field(default_factory=list)
    solver_status: str = "n/a"
    compute_time_s: float = 0.0
    constraint_violations: int = 0

    def as_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "position": None if self.position is None else self.position.tolist(),
            "velocity": None if self.velocity is None else self.velocity.tolist(),
            "reason": self.reason,
            "mode": self.mode,
            "intervened": self.intervened,
            "intervention_magnitude": self.intervention_magnitude,
            "active_constraints": list(self.active_constraints),
            "solver_status": self.solver_status,
            "compute_time_s": self.compute_time_s,
            "constraint_violations": self.constraint_violations,
        }


@runtime_checkable
class SafetyFilter(Protocol):
    name: str

    def filter(
        self,
        current_position: Sequence[float],
        desired_position: Sequence[float],
        dt: Optional[float] = None,
    ) -> SafetyFilterResult:
        ...

    def reset(self) -> None:
        ...


def create_safety_filter(config: Optional[SafetyFilterConfig] = None) -> SafetyFilter:
    cfg = config or SafetyFilterConfig()
    mode = str(cfg.mode).lower().strip()
    if mode == "reject":
        from .hard_limits import RejectFilter

        return RejectFilter(cfg)
    if mode == "clamp":
        from .hard_limits import ClampFilter

        return ClampFilter(cfg)
    if mode in ("cbf_qp", "cbf-qp", "cbf"):
        from .cbf_qp import CBFQPFilter

        return CBFQPFilter(cfg)
    raise ValueError(f"Unknown safety filter mode '{cfg.mode}'. Supported: reject, clamp, cbf_qp")
