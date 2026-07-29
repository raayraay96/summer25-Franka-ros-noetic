"""Baseline shoulder-relative end-effector position retargeting.

Preserves the existing repository mapping behavior:
  displacement = wrist - shoulder (image-normalized or Cartesian)
  target = origin + scale * displacement

This is **end-effector position teleoperation**, not full SEW arm orientation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from vision_arm_control.coordinate_mapping import scale_displacement
from vision_arm_control.landmarks.schema import (
    ArmLandmarks,
    RetargetingStatus,
    RetargetingTarget,
)

from .base import RetargetingConfig, safe_unit


class ShoulderRelativeRetargeter:
    """Existing baseline mapper, exposed as a selectable strategy."""

    def __init__(self, config: Optional[RetargetingConfig] = None) -> None:
        self.config = config or RetargetingConfig(name="shoulder_relative")
        self.name = "shoulder_relative"

    def reset(self) -> None:
        return None

    def retarget(self, landmarks: ArmLandmarks) -> RetargetingTarget:
        cfg = self.config
        ts = float(landmarks.timestamp)

        if landmarks.shoulder is None:
            return RetargetingTarget(
                position=None,
                status=RetargetingStatus.MISSING_SHOULDER,
                reason="missing_shoulder",
                timestamp=ts,
            )
        if landmarks.wrist is None:
            return RetargetingTarget(
                position=None,
                status=RetargetingStatus.MISSING_WRIST,
                reason="missing_wrist",
                timestamp=ts,
            )
        if not landmarks.shoulder.is_finite() or not landmarks.wrist.is_finite():
            return RetargetingTarget(
                position=None,
                status=RetargetingStatus.NON_FINITE,
                reason="non_finite",
                timestamp=ts,
            )
        conf = min(landmarks.shoulder.confidence, landmarks.wrist.confidence)
        if conf < cfg.min_confidence:
            return RetargetingTarget(
                position=None,
                confidence=conf,
                status=RetargetingStatus.LOW_CONFIDENCE,
                reason="low_confidence",
                timestamp=ts,
            )

        s = landmarks.shoulder.as_array()
        w = landmarks.wrist.as_array()
        # Image-normalized path: use x,y displacement normalized by image size
        # matching map_wrist_end_effector_position shoulder_relative mode.
        if cfg.treat_z_as_image_relative:
            width = float(cfg.image_width)
            height = float(cfg.image_height)
            # Landmarks may already be normalized [0,1]; convert to pixel-relative
            # displacement normalized by image size (same as baseline).
            disp = np.array(
                [
                    (w[0] - s[0]),  # already normalized if MediaPipe-style
                    (w[1] - s[1]),
                    0.0,
                ],
                dtype=np.float64,
            )
            # If values look like pixels, normalize.
            if abs(disp[0]) > 2.0 or abs(disp[1]) > 2.0:
                disp[0] /= width
                disp[1] /= height
        else:
            disp = w - s

        if not np.all(np.isfinite(disp)):
            return RetargetingTarget(
                position=None,
                confidence=conf,
                status=RetargetingStatus.NON_FINITE,
                reason="non_finite_displacement",
                timestamp=ts,
            )

        target = scale_displacement(disp, cfg.workspace_scale, cfg.workspace_origin)
        direction = safe_unit(disp, cfg.min_segment_norm)

        return RetargetingTarget(
            position=target,
            orientation_xyzw=np.asarray(cfg.fixed_orientation_xyzw, dtype=np.float64),
            direction=direction,
            confidence=float(conf),
            status=RetargetingStatus.OK,
            reason="ok",
            timestamp=ts,
            joint_positions=None,
            ik_status="not_applicable_position_teleop",
            metadata={"strategy": self.name},
        )
