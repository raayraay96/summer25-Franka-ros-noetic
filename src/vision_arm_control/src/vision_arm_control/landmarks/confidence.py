"""Confidence-aware command gating (not probabilistic UQ)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from .schema import ArmLandmarks, RetargetingStatus, RetargetingTarget


class GateStatus(str, Enum):
    PASS = "pass"
    HOLD = "hold"
    STOPPED = "stopped"
    RECOVERING = "recovering"
    REJECT = "reject"


@dataclass
class ConfidenceGateConfig:
    min_keypoint_confidence: float = 0.5
    hold_interval_sec: float = 0.25
    stale_timeout_sec: float = 0.50
    recovery_blend_sec: float = 0.30
    max_timestamp_rewind_sec: float = 0.05
    require_shoulder: bool = True
    require_elbow: bool = False  # shoulder_relative can run without elbow
    require_wrist: bool = True


@dataclass
class ConfidenceGate:
    """Gate observations by confidence, staleness, and recovery blending.

    Behaviors:
    - low confidence → hold last safe target for ``hold_interval_sec``
    - after hold expires → no-command / stopped
    - reacquisition → blend from last safe toward new target (no sudden jump)
    - timestamp reordering beyond tolerance → reject as stale/reorder

    This is **confidence-aware gating**, not probabilistic uncertainty
    quantification.
    """

    config: ConfidenceGateConfig = field(default_factory=ConfidenceGateConfig)
    _last_safe: Optional[RetargetingTarget] = None
    _last_valid_time: Optional[float] = None
    _last_timestamp: Optional[float] = None
    _hold_start: Optional[float] = None
    _recovery_start: Optional[float] = None
    _recovery_from: Optional[np.ndarray] = None
    status: GateStatus = GateStatus.STOPPED
    last_reason: str = "init"

    def reset(self) -> None:
        self._last_safe = None
        self._last_valid_time = None
        self._last_timestamp = None
        self._hold_start = None
        self._recovery_start = None
        self._recovery_from = None
        self.status = GateStatus.STOPPED
        self.last_reason = "reset"

    def validate_landmarks(self, landmarks: ArmLandmarks, now: float) -> Tuple[bool, RetargetingStatus, str]:
        """Return (ok, status_enum, reason)."""
        cfg = self.config

        if self._last_timestamp is not None:
            rewind = self._last_timestamp - float(landmarks.timestamp)
            if rewind > cfg.max_timestamp_rewind_sec:
                return False, RetargetingStatus.STALE_TIMESTAMP, "timestamp_reordered"

        if cfg.require_shoulder and landmarks.shoulder is None:
            return False, RetargetingStatus.MISSING_SHOULDER, "missing_shoulder"
        if cfg.require_elbow and landmarks.elbow is None:
            return False, RetargetingStatus.MISSING_ELBOW, "missing_elbow"
        if cfg.require_wrist and landmarks.wrist is None:
            return False, RetargetingStatus.MISSING_WRIST, "missing_wrist"

        for p in landmarks.points():
            if p is None:
                continue
            if not p.is_finite():
                return False, RetargetingStatus.NON_FINITE, "non_finite_landmark"
            if float(p.confidence) < cfg.min_keypoint_confidence:
                return False, RetargetingStatus.LOW_CONFIDENCE, "low_confidence"

        age = float(now) - float(landmarks.timestamp)
        if age > cfg.stale_timeout_sec:
            return False, RetargetingStatus.STALE_TIMESTAMP, "stale_observation"

        return True, RetargetingStatus.OK, "ok"

    def update(
        self,
        candidate: Optional[RetargetingTarget],
        landmarks: Optional[ArmLandmarks],
        now: float,
    ) -> RetargetingTarget:
        """Apply confidence / hold / recovery policy to a retargeting candidate."""
        cfg = self.config
        now = float(now)

        valid = False
        reason = "no_observation"
        status_code = RetargetingStatus.NO_COMMAND

        if landmarks is not None:
            valid, status_code, reason = self.validate_landmarks(landmarks, now)

        if valid and candidate is not None and candidate.is_commandable():
            self._last_timestamp = float(landmarks.timestamp) if landmarks else now
            return self._accept_or_recover(candidate, now)

        # Invalid observation path: hold then stop
        if self._last_safe is not None and self._last_safe.position is not None:
            if self._hold_start is None:
                self._hold_start = now
            hold_age = now - self._hold_start
            # hold_interval_sec <= 0 disables holding entirely
            if cfg.hold_interval_sec > 0.0 and hold_age < cfg.hold_interval_sec:
                self.status = GateStatus.HOLD
                self.last_reason = f"hold:{reason}"
                held = RetargetingTarget(
                    position=self._last_safe.position.copy(),
                    orientation_xyzw=(
                        None
                        if self._last_safe.orientation_xyzw is None
                        else self._last_safe.orientation_xyzw.copy()
                    ),
                    direction=self._last_safe.direction,
                    elbow_plane_normal=self._last_safe.elbow_plane_normal,
                    upper_arm_unit=self._last_safe.upper_arm_unit,
                    lower_arm_unit=self._last_safe.lower_arm_unit,
                    elbow_bend_rad=self._last_safe.elbow_bend_rad,
                    confidence=self._last_safe.confidence,
                    status=RetargetingStatus.HOLDING_LAST_SAFE,
                    reason=self.last_reason,
                    timestamp=now,
                    ik_status=self._last_safe.ik_status,
                    metadata={"gate": self.status.value, "underlying": reason},
                )
                return held

        self.status = GateStatus.STOPPED
        self.last_reason = f"stopped:{reason}"
        self._hold_start = self._hold_start or now
        return RetargetingTarget(
            position=None,
            confidence=0.0,
            status=(status_code if status_code != RetargetingStatus.OK else RetargetingStatus.NO_COMMAND),
            reason=self.last_reason,
            timestamp=now,
            metadata={"gate": self.status.value},
        )

    def _accept_or_recover(self, candidate: RetargetingTarget, now: float) -> RetargetingTarget:
        cfg = self.config
        was_degraded = self.status in (
            GateStatus.HOLD,
            GateStatus.STOPPED,
            GateStatus.REJECT,
        )
        if was_degraded and self._last_safe is not None and self._last_safe.position is not None:
            if self._recovery_start is None:
                self._recovery_start = now
                self._recovery_from = self._last_safe.position.copy()

        self._hold_start = None

        pos = candidate.position
        if (
            self._recovery_start is not None
            and self._recovery_from is not None
            and pos is not None
            and cfg.recovery_blend_sec > 0.0
        ):
            t = (now - self._recovery_start) / cfg.recovery_blend_sec
            if t < 1.0:
                alpha = float(np.clip(t, 0.0, 1.0))
                blended = (1.0 - alpha) * self._recovery_from + alpha * pos
                self.status = GateStatus.RECOVERING
                self.last_reason = "recovering"
                out = RetargetingTarget(
                    position=blended,
                    orientation_xyzw=candidate.orientation_xyzw,
                    direction=candidate.direction,
                    elbow_plane_normal=candidate.elbow_plane_normal,
                    upper_arm_unit=candidate.upper_arm_unit,
                    lower_arm_unit=candidate.lower_arm_unit,
                    elbow_bend_rad=candidate.elbow_bend_rad,
                    confidence=candidate.confidence,
                    status=RetargetingStatus.OK,
                    reason="recovering",
                    timestamp=candidate.timestamp,
                    ik_status=candidate.ik_status,
                    metadata={"gate": "recovering", "blend_alpha": alpha},
                )
                self._last_safe = out
                self._last_valid_time = now
                return out
            self._recovery_start = None
            self._recovery_from = None

        self.status = GateStatus.PASS
        self.last_reason = "pass"
        self._last_safe = candidate
        self._last_valid_time = now
        return candidate
