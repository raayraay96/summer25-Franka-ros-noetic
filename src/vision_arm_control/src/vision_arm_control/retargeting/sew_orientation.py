"""SEW-inspired arm-orientation retargeting (paper-inspired baseline).

Inspired by SEW-Mimic (arXiv:2602.01632): use shoulder, elbow, wrist (SEW)
keypoints to recover upper-arm and lower-arm directions and the arm-plane
normal, then map those geometric features into a robot base-frame end-effector
target with an orientation/direction signal.

This is **not** a full reproduction of the closed-form 7-DoF SEW-Mimic joint
solver. No optimality claim is made. Joint IK is intentionally not faked:
``joint_positions`` remains None and ``ik_status`` documents the missing layer.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from vision_arm_control.landmarks.schema import (
    ArmLandmarks,
    RetargetingStatus,
    RetargetingTarget,
)

from .base import RetargetingConfig, safe_unit


def elbow_bend_angle(upper_unit: np.ndarray, lower_unit: np.ndarray) -> float:
    """Interior elbow bend angle (radians) between upper and lower arm units.

    Angle between vectors pointing away from the elbow along each segment
    is computed from the chain directions (shoulder→elbow, elbow→wrist).
    Returns the supplement that grows as the arm folds (0 = fully extended
    collinear continuation, pi = fully folded).
    """
    # upper_unit: shoulder → elbow; lower_unit: elbow → wrist
    # At full extension, lower ≈ upper ⇒ angle between them ≈ 0.
    c = float(np.clip(np.dot(upper_unit, lower_unit), -1.0, 1.0))
    return float(np.arccos(c))


def arm_plane_normal(
    upper_unit: np.ndarray, lower_unit: np.ndarray, sin_eps: float
) -> Tuple[Optional[np.ndarray], RetargetingStatus]:
    """Compute arm-plane normal; detect near-collinear degeneracy."""
    n = np.cross(upper_unit, lower_unit)
    sin_mag = float(np.linalg.norm(n))
    if sin_mag < sin_eps:
        return None, RetargetingStatus.COLLINEAR_ARM_PLANE
    return n / sin_mag, RetargetingStatus.OK


class SEWOrientationRetargeter:
    """SEW-inspired feature-level retargeter for Panda EE teleoperation.

    Pipeline:
      1. Validate SEW landmarks and confidence
      2. upper = normalize(elbow - shoulder)
      3. lower = normalize(wrist - elbow)
      4. elbow bend angle; arm-plane normal when valid
      5. Map wrist-relative chain into robot workspace as EE position
      6. Emit orientation/direction from lower-arm unit and plane normal

    Output is a documented retargeting **target feature** suitable for the
    existing Cartesian simulation backend. Full SEW-Mimic closed-form joint
    retargeting is **not** implemented.
    """

    def __init__(self, config: Optional[RetargetingConfig] = None) -> None:
        self.config = config or RetargetingConfig(name="sew_orientation")
        self.name = "sew_orientation"

    def reset(self) -> None:
        return None

    def retarget(self, landmarks: ArmLandmarks) -> RetargetingTarget:
        cfg = self.config
        ts = float(landmarks.timestamp)

        if landmarks.shoulder is None:
            return self._fail(RetargetingStatus.MISSING_SHOULDER, ts)
        if landmarks.elbow is None:
            return self._fail(RetargetingStatus.MISSING_ELBOW, ts)
        if landmarks.wrist is None:
            return self._fail(RetargetingStatus.MISSING_WRIST, ts)

        for p, code in (
            (landmarks.shoulder, RetargetingStatus.NON_FINITE),
            (landmarks.elbow, RetargetingStatus.NON_FINITE),
            (landmarks.wrist, RetargetingStatus.NON_FINITE),
        ):
            if not p.is_finite():
                return self._fail(code, ts, "non_finite")

        conf = float(
            min(
                landmarks.shoulder.confidence,
                landmarks.elbow.confidence,
                landmarks.wrist.confidence,
            )
        )
        if conf < cfg.min_confidence:
            return self._fail(RetargetingStatus.LOW_CONFIDENCE, ts, "low_confidence", conf)

        s = landmarks.shoulder.as_array()
        e = landmarks.elbow.as_array()
        w = landmarks.wrist.as_array()

        # If image-normalized 2.5D MediaPipe-style, lift z slightly using
        # relative landmark z differences so the chain is 3D-ish without
        # claiming metric depth.
        if cfg.treat_z_as_image_relative:
            # x,y in [0,1]; keep z as relative offset (not meters).
            pass

        upper_vec = e - s
        lower_vec = w - e
        upper = safe_unit(upper_vec, cfg.min_segment_norm)
        if upper is None:
            return self._fail(RetargetingStatus.UPPER_ARM_NORM_NEAR_ZERO, ts, confidence=conf)
        lower = safe_unit(lower_vec, cfg.min_segment_norm)
        if lower is None:
            return self._fail(RetargetingStatus.LOWER_ARM_NORM_NEAR_ZERO, ts, confidence=conf)

        bend = elbow_bend_angle(upper, lower)
        plane_n, plane_status = arm_plane_normal(upper, lower, cfg.collinear_sin_eps)
        # Collinear plane is a soft warning: still produce EE position from chain.
        plane_ok = plane_status == RetargetingStatus.OK

        # Chain displacement shoulder→wrist, scale-normalized for teleop workspace.
        chain = w - s
        if cfg.treat_z_as_image_relative:
            # Map image-normalized SEW chain into workspace axes:
            # image x → robot -y (camera facing robot is application-specific;
            # we keep a simple configurable scale like shoulder_relative).
            disp = np.array([chain[0], chain[1], chain[2] * 0.5], dtype=np.float64)
            if np.max(np.abs(disp[:2])) > 2.0:
                disp[0] /= float(cfg.image_width)
                disp[1] /= float(cfg.image_height)
        else:
            # Metric-ish body frame: normalize by reference arm length.
            ref = max(cfg.human_upper_arm_ref_m + cfg.human_lower_arm_ref_m, 1e-6)
            disp = (chain / ref) * cfg.robot_reach_scale

        origin = np.asarray(cfg.workspace_origin, dtype=np.float64)
        scale = np.asarray(cfg.workspace_scale, dtype=np.float64)
        position = origin + scale * disp

        if not np.all(np.isfinite(position)):
            return self._fail(RetargetingStatus.NON_FINITE, ts, "non_finite_target", conf)

        # Orientation: point EE approach roughly along lower-arm direction.
        # Quaternion from direction is a simple heuristic, not SEW-Mimic joints.
        orientation = _direction_to_quat(lower, plane_n if plane_ok else None)
        if orientation is None:
            orientation = np.asarray(cfg.fixed_orientation_xyzw, dtype=np.float64)

        status = RetargetingStatus.OK
        reason = "ok" if plane_ok else "ok_collinear_plane_soft"
        return RetargetingTarget(
            position=position,
            orientation_xyzw=orientation,
            direction=lower.copy(),
            elbow_plane_normal=None if plane_n is None else plane_n.copy(),
            upper_arm_unit=upper.copy(),
            lower_arm_unit=lower.copy(),
            elbow_bend_rad=bend,
            confidence=conf,
            status=status,
            reason=reason,
            timestamp=ts,
            joint_positions=None,
            ik_status="missing_ik_layer_feature_level_only",
            metadata={
                "strategy": self.name,
                "inspired_by": "SEW-Mimic arXiv:2602.01632",
                "plane_valid": plane_ok,
                "not_full_sew_mimic": True,
            },
        )

    def _fail(
        self,
        status: RetargetingStatus,
        ts: float,
        reason: Optional[str] = None,
        confidence: float = 0.0,
    ) -> RetargetingTarget:
        return RetargetingTarget(
            position=None,
            confidence=confidence,
            status=status,
            reason=reason or status.value,
            timestamp=ts,
            joint_positions=None,
            ik_status="missing_ik_layer_feature_level_only",
            metadata={"strategy": self.name, "not_full_sew_mimic": True},
        )


def _direction_to_quat(direction: np.ndarray, plane_normal: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Build a crude orientation quaternion with z-axis along ``direction``.

    Returns xyzw. Falls back to None on degeneracy.
    """
    z = safe_unit(direction, 1e-9)
    if z is None:
        return None
    if plane_normal is not None:
        x = safe_unit(plane_normal, 1e-9)
        if x is None:
            x = safe_unit(np.cross(np.array([0.0, 0.0, 1.0]), z), 1e-9)
    else:
        x = safe_unit(np.cross(np.array([0.0, 0.0, 1.0]), z), 1e-9)
    if x is None:
        x = safe_unit(np.cross(np.array([0.0, 1.0, 0.0]), z), 1e-9)
    if x is None:
        return None
    y = np.cross(z, x)
    y = safe_unit(y, 1e-9)
    if y is None:
        return None
    # Re-orthogonalize x
    x = np.cross(y, z)
    # Rotation matrix columns = axes
    R = np.column_stack([x, y, z])
    return _rotmat_to_quat_xyzw(R)


def _rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (x, y, z, w)."""
    m = R
    t = float(np.trace(m))
    if t > 0.0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n
