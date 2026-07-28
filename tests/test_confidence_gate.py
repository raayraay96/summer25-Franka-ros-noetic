"""Deterministic tests for confidence-aware command gating."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vision_arm_control" / "src"))

from vision_arm_control.landmarks.confidence import (  # noqa: E402
    ConfidenceGate,
    ConfidenceGateConfig,
    GateStatus,
)
from vision_arm_control.landmarks.schema import (  # noqa: E402
    ArmLandmarks,
    LandmarkPoint,
    RetargetingStatus,
    RetargetingTarget,
)


def _pt(x=0.5, y=0.5, z=0.0, c=1.0):
    return LandmarkPoint(x=x, y=y, z=z, confidence=c)


def _lm(ts, conf=1.0, shoulder=True, elbow=True, wrist=True):
    return ArmLandmarks(
        shoulder=_pt(0.45, 0.35, 0, conf) if shoulder else None,
        elbow=_pt(0.50, 0.40, 0, conf) if elbow else None,
        wrist=_pt(0.55, 0.50, 0, conf) if wrist else None,
        timestamp=ts,
    )


def _target(pos, ts, conf=1.0):
    return RetargetingTarget(
        position=np.asarray(pos, dtype=np.float64),
        confidence=conf,
        status=RetargetingStatus.OK,
        reason="ok",
        timestamp=ts,
    )


def test_one_frame_dropout_holds():
    cfg = ConfidenceGateConfig(hold_interval_sec=0.25, stale_timeout_sec=1.0)
    g = ConfidenceGate(cfg)
    t0 = _target([0.5, 0.0, 0.4], 1.0)
    out0 = g.update(t0, _lm(1.0), now=1.0)
    assert out0.is_commandable()
    # One frame dropout
    out1 = g.update(None, None, now=1.05)
    assert out1.status == RetargetingStatus.HOLDING_LAST_SAFE
    np.testing.assert_allclose(out1.position, [0.5, 0.0, 0.4])


def test_multi_frame_dropout_then_stop():
    cfg = ConfidenceGateConfig(hold_interval_sec=0.2, stale_timeout_sec=1.0)
    g = ConfidenceGate(cfg)
    g.update(_target([0.5, 0.0, 0.4], 1.0), _lm(1.0), now=1.0)
    # Within hold
    h = g.update(None, None, now=1.15)
    assert h.status == RetargetingStatus.HOLDING_LAST_SAFE
    # Past hold
    s = g.update(None, None, now=1.35)
    assert not s.is_commandable()
    assert g.status == GateStatus.STOPPED


def test_noisy_wrist_low_confidence():
    cfg = ConfidenceGateConfig(min_keypoint_confidence=0.5, hold_interval_sec=0.3)
    g = ConfidenceGate(cfg)
    g.update(_target([0.5, 0.0, 0.4], 1.0), _lm(1.0, conf=0.9), now=1.0)
    bad = _lm(1.05, conf=0.1)
    out = g.update(_target([0.6, 0.0, 0.4], 1.05, conf=0.1), bad, now=1.05)
    assert out.status == RetargetingStatus.HOLDING_LAST_SAFE


def test_noisy_elbow_when_required():
    cfg = ConfidenceGateConfig(min_keypoint_confidence=0.5, require_elbow=True, hold_interval_sec=0.3)
    g = ConfidenceGate(cfg)
    g.update(_target([0.5, 0.0, 0.4], 1.0), _lm(1.0), now=1.0)
    bad = ArmLandmarks(
        shoulder=_pt(c=0.9),
        elbow=_pt(c=0.1),
        wrist=_pt(c=0.9),
        timestamp=1.05,
    )
    out = g.update(_target([0.55, 0, 0.4], 1.05), bad, now=1.05)
    assert out.status == RetargetingStatus.HOLDING_LAST_SAFE


def test_confidence_oscillation_near_threshold():
    cfg = ConfidenceGateConfig(min_keypoint_confidence=0.5, hold_interval_sec=0.15, recovery_blend_sec=0.2)
    g = ConfidenceGate(cfg)
    g.update(_target([0.5, 0.0, 0.4], 1.0), _lm(1.0, conf=0.9), now=1.0)
    # Below
    g.update(_target([0.5, 0.0, 0.4], 1.05), _lm(1.05, conf=0.49), now=1.05)
    assert g.status == GateStatus.HOLD
    # Above again
    out = g.update(_target([0.52, 0.0, 0.4], 1.10), _lm(1.10, conf=0.51), now=1.10)
    assert out.is_commandable()


def test_timestamp_reordering():
    cfg = ConfidenceGateConfig(max_timestamp_rewind_sec=0.05)
    g = ConfidenceGate(cfg)
    g.update(_target([0.5, 0, 0.4], 2.0), _lm(2.0), now=2.0)
    # Large rewind
    out = g.update(_target([0.5, 0, 0.4], 1.0), _lm(1.0), now=2.05)
    assert out.status == RetargetingStatus.HOLDING_LAST_SAFE or not out.is_commandable()
    assert "timestamp" in g.last_reason or out.reason


def test_recovery_no_sudden_jump():
    cfg = ConfidenceGateConfig(hold_interval_sec=0.1, recovery_blend_sec=0.4, stale_timeout_sec=2.0)
    g = ConfidenceGate(cfg)
    g.update(_target([0.40, 0.0, 0.40], 1.0), _lm(1.0), now=1.0)
    # Start hold, then exceed hold interval
    g.update(None, None, now=1.05)
    assert g.status == GateStatus.HOLD
    g.update(None, None, now=1.20)
    assert g.status == GateStatus.STOPPED
    # Reacquire far target
    out = g.update(_target([0.60, 0.0, 0.40], 1.25), _lm(1.25), now=1.25)
    assert out.is_commandable()
    # Should be blended, not full jump to 0.60
    assert out.position[0] < 0.60 - 1e-6
    assert out.position[0] >= 0.40 - 1e-9
    assert out.metadata.get("gate") == "recovering"
    # Later in recovery, moves toward new target
    out2 = g.update(_target([0.60, 0.0, 0.40], 1.40), _lm(1.40), now=1.40)
    assert out2.position[0] > out.position[0]


def test_stale_observation_timeout():
    cfg = ConfidenceGateConfig(stale_timeout_sec=0.2, hold_interval_sec=0.5)
    g = ConfidenceGate(cfg)
    g.update(_target([0.5, 0, 0.4], 1.0), _lm(1.0), now=1.0)
    # Observation timestamp old relative to now
    stale = _lm(1.0)
    out = g.update(_target([0.5, 0, 0.4], 1.0), stale, now=1.5)
    assert out.status == RetargetingStatus.HOLDING_LAST_SAFE or not out.is_commandable()
