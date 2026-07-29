"""Hard reject / clamp workspace safety filters (existing baseline)."""

from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np

from vision_arm_control.workspace_limits import validate_or_clamp_position

from .base import SafetyFilterConfig, SafetyFilterResult


class RejectFilter:
    """Reject targets outside the workspace (baseline)."""

    def __init__(self, config: Optional[SafetyFilterConfig] = None) -> None:
        self.config = config or SafetyFilterConfig(mode="reject")
        self.name = "reject"

    def reset(self) -> None:
        return None

    def filter(
        self,
        current_position: Sequence[float],
        desired_position: Sequence[float],
        dt: Optional[float] = None,
    ) -> SafetyFilterResult:
        t0 = time.perf_counter()
        pos, ok, reason = validate_or_clamp_position(desired_position, self.config.workspace, mode="reject")
        elapsed = time.perf_counter() - t0
        if not ok or pos is None:
            return SafetyFilterResult(
                accepted=False,
                position=None,
                velocity=None,
                reason=reason,
                mode=self.name,
                intervened=True,
                intervention_magnitude=0.0,
                active_constraints=["workspace"],
                solver_status="n/a",
                compute_time_s=elapsed,
                constraint_violations=1,
            )
        return SafetyFilterResult(
            accepted=True,
            position=pos,
            velocity=None,
            reason=reason,
            mode=self.name,
            intervened=False,
            compute_time_s=elapsed,
        )


class ClampFilter:
    """Clamp targets into the workspace (baseline)."""

    def __init__(self, config: Optional[SafetyFilterConfig] = None) -> None:
        self.config = config or SafetyFilterConfig(mode="clamp")
        self.name = "clamp"

    def reset(self) -> None:
        return None

    def filter(
        self,
        current_position: Sequence[float],
        desired_position: Sequence[float],
        dt: Optional[float] = None,
    ) -> SafetyFilterResult:
        t0 = time.perf_counter()
        desired = np.asarray(desired_position, dtype=np.float64)
        pos, ok, reason = validate_or_clamp_position(desired, self.config.workspace, mode="clamp")
        elapsed = time.perf_counter() - t0
        if not ok or pos is None:
            return SafetyFilterResult(
                accepted=False,
                position=None,
                velocity=None,
                reason=reason,
                mode=self.name,
                intervened=True,
                compute_time_s=elapsed,
                constraint_violations=1,
            )
        mag = float(np.linalg.norm(pos - desired))
        intervened = reason == "clamped_to_workspace" or mag > 1e-12
        return SafetyFilterResult(
            accepted=True,
            position=pos,
            velocity=None,
            reason=reason,
            mode=self.name,
            intervened=intervened,
            intervention_magnitude=mag,
            active_constraints=["workspace"] if intervened else [],
            compute_time_s=elapsed,
            constraint_violations=1 if intervened else 0,
        )
