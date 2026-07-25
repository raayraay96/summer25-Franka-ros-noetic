"""Trajectory smoothing and rate limiting for end-effector targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class LowPassFilter3D:
    """Exponential moving average filter for 3D positions."""

    alpha: float = 0.3
    _state: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._state = None

    def update(self, sample: Sequence[float]) -> np.ndarray:
        x = np.asarray(sample, dtype=np.float64)
        if x.shape != (3,):
            raise ValueError("sample must be length-3")
        a = float(self.alpha)
        if not 0.0 < a <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if self._state is None:
            self._state = x.copy()
        else:
            self._state = a * x + (1.0 - a) * self._state
        return self._state.copy()


@dataclass
class VelocityLimiter3D:
    """Limit Cartesian step size based on max linear velocity and dt."""

    max_linear_velocity: float = 0.25  # m/s
    _last: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._last = None

    def update(self, target: Sequence[float], dt: float) -> np.ndarray:
        x = np.asarray(target, dtype=np.float64)
        if x.shape != (3,):
            raise ValueError("target must be length-3")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if self._last is None:
            self._last = x.copy()
            return x.copy()
        delta = x - self._last
        max_step = self.max_linear_velocity * dt
        dist = float(np.linalg.norm(delta))
        if dist > max_step and dist > 1e-12:
            delta = delta * (max_step / dist)
        out = self._last + delta
        self._last = out.copy()
        return out


@dataclass
class CommandRateLimiter:
    """Ensure commands are not emitted faster than max_hz."""

    max_hz: float = 50.0
    _last_time: Optional[float] = None

    def allow(self, now: float) -> bool:
        if self.max_hz <= 0.0:
            return True
        min_dt = 1.0 / self.max_hz
        if self._last_time is None or (now - self._last_time) >= min_dt:
            self._last_time = now
            return True
        return False
