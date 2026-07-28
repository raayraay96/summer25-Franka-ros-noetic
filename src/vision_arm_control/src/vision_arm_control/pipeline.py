"""Composable teleoperation pipeline: gate → retarget → safety → backend.

Inspired by modular vision teleoperation stacks (AnyTeleop-style separation
of perception, retargeting, and execution interfaces) without copying code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from vision_arm_control.backends.base import BackendResult, RobotBackend, create_backend
from vision_arm_control.landmarks.confidence import ConfidenceGate, ConfidenceGateConfig
from vision_arm_control.landmarks.schema import (
    ArmLandmarks,
    RetargetingStatus,
    RetargetingTarget,
)
from vision_arm_control.retargeting.base import (
    RetargetingConfig,
    RetargetingStrategy,
    create_retargeter,
)
from vision_arm_control.safety_filters.base import (
    SafetyFilter,
    SafetyFilterConfig,
    SafetyFilterResult,
    create_safety_filter,
)


@dataclass
class PipelineConfig:
    retargeting: RetargetingConfig = field(default_factory=RetargetingConfig)
    safety: SafetyFilterConfig = field(default_factory=SafetyFilterConfig)
    gate: ConfidenceGateConfig = field(default_factory=ConfidenceGateConfig)
    backend: str = "dry_run"
    initial_position: Optional[tuple] = (0.45, 0.0, 0.45)


@dataclass
class PipelineStepResult:
    retargeted: RetargetingTarget
    gated: RetargetingTarget
    safety: Optional[SafetyFilterResult]
    backend: Optional[BackendResult]
    commanded: bool


class TeleopPipeline:
    """Selectable retargeter + safety filter + backend with confidence gate."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.retargeter: RetargetingStrategy = create_retargeter(self.config.retargeting)
        self.safety: SafetyFilter = create_safety_filter(self.config.safety)
        self.gate = ConfidenceGate(self.config.gate)
        self.backend: RobotBackend = create_backend(self.config.backend)
        init = self.config.initial_position or (0.45, 0.0, 0.45)
        self._current = np.asarray(init, dtype=np.float64)

    def reset(self) -> None:
        self.retargeter.reset()
        self.safety.reset()
        self.gate.reset()
        self.backend.reset()
        init = self.config.initial_position or (0.45, 0.0, 0.45)
        self._current = np.asarray(init, dtype=np.float64)

    def step(self, landmarks: Optional[ArmLandmarks], now: float) -> PipelineStepResult:
        if landmarks is None:
            retargeted = RetargetingTarget(
                position=None,
                status=RetargetingStatus.NO_COMMAND,
                reason="no_landmarks",
                timestamp=now,
            )
            candidate = None
        else:
            retargeted = self.retargeter.retarget(landmarks)
            candidate = retargeted if retargeted.is_commandable() else None

        gated = self.gate.update(candidate, landmarks, now)

        safety_res: Optional[SafetyFilterResult] = None
        backend_res: Optional[BackendResult] = None
        commanded = False

        if gated.is_commandable() and gated.position is not None:
            safety_res = self.safety.filter(self._current, gated.position, dt=self.config.safety.dt)
            if safety_res.accepted and safety_res.position is not None:
                backend_res = self.backend.send_cartesian_target(safety_res.position, gated.orientation_xyzw)
                if backend_res.accepted:
                    self._current = safety_res.position.copy()
                    commanded = True
        return PipelineStepResult(
            retargeted=retargeted,
            gated=gated,
            safety=safety_res,
            backend=backend_res,
            commanded=commanded,
        )
