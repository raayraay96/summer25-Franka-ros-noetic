"""Retargeting strategy protocol and factory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from vision_arm_control.landmarks.schema import ArmLandmarks, RetargetingTarget


@dataclass
class RetargetingConfig:
    """Shared configuration for retargeting strategies."""

    name: str = "shoulder_relative"
    workspace_origin: Sequence[float] = field(default_factory=lambda: (0.45, 0.0, 0.45))
    workspace_scale: Sequence[float] = field(default_factory=lambda: (0.55, 0.55, 0.35))
    # Human reference lengths used for SEW-inspired scale normalization (meters).
    # These are configuration knobs, not measured body anthropometry claims.
    human_upper_arm_ref_m: float = 0.28
    human_lower_arm_ref_m: float = 0.25
    robot_reach_scale: float = 0.55
    min_segment_norm: float = 1e-4
    collinear_sin_eps: float = 1e-3
    min_confidence: float = 0.5
    fixed_orientation_xyzw: Sequence[float] = field(
        default_factory=lambda: (1.0, 0.0, 0.0, 0.0)
    )  # default: identity-like placeholder; mapper may override
    image_width: int = 640
    image_height: int = 480
    # When landmarks are image-normalized 2.5D, z may be MediaPipe relative.
    treat_z_as_image_relative: bool = True


@runtime_checkable
class RetargetingStrategy(Protocol):
    """Pure-Python retargeting strategy interface."""

    name: str

    def retarget(self, landmarks: ArmLandmarks) -> RetargetingTarget:
        """Map arm landmarks to a robot retargeting target."""
        ...

    def reset(self) -> None:
        """Clear any internal state."""
        ...


def create_retargeter(config: Optional[RetargetingConfig] = None) -> RetargetingStrategy:
    """Factory for configured retargeting strategies."""
    cfg = config or RetargetingConfig()
    name = str(cfg.name).lower().strip()
    if name in ("shoulder_relative", "shoulder-relative"):
        from .shoulder_relative import ShoulderRelativeRetargeter

        return ShoulderRelativeRetargeter(cfg)
    if name in ("sew_orientation", "sew", "sew-inspired", "sew_inspired"):
        from .sew_orientation import SEWOrientationRetargeter

        return SEWOrientationRetargeter(cfg)
    raise ValueError(
        f"Unknown retargeting strategy '{cfg.name}'. " "Supported: shoulder_relative, sew_orientation"
    )


def safe_unit(v: np.ndarray, eps: float) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n < eps or not np.isfinite(n):
        return None
    return v / n
