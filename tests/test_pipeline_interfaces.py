"""Interface and pipeline integration tests (no ROS)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vision_arm_control" / "src"))

from vision_arm_control.backends.base import create_backend  # noqa: E402
from vision_arm_control.landmarks.schema import ArmLandmarks, LandmarkPoint  # noqa: E402
from vision_arm_control.pipeline import PipelineConfig, TeleopPipeline  # noqa: E402
from vision_arm_control.retargeting.base import RetargetingConfig  # noqa: E402
from vision_arm_control.safety_filters.base import SafetyFilterConfig  # noqa: E402
from vision_arm_control.landmarks.confidence import ConfidenceGateConfig  # noqa: E402


def _lm(ts=1.0):
    return ArmLandmarks(
        shoulder=LandmarkPoint(0.45, 0.35, 0.0, 1.0, "right_shoulder"),
        elbow=LandmarkPoint(0.50, 0.42, 0.0, 1.0, "right_elbow"),
        wrist=LandmarkPoint(0.58, 0.50, 0.0, 1.0, "right_wrist"),
        timestamp=ts,
    )


def test_default_pipeline_is_shoulder_relative_reject_dry_run():
    p = TeleopPipeline()
    assert p.retargeter.name == "shoulder_relative"
    assert p.safety.name == "reject"
    assert p.backend.name == "dry_run"
    res = p.step(_lm(1.0), now=1.0)
    assert res.commanded


def test_sew_plus_cbf_qp_combination():
    cfg = PipelineConfig(
        retargeting=RetargetingConfig(name="sew_orientation"),
        safety=SafetyFilterConfig(mode="cbf_qp"),
        gate=ConfidenceGateConfig(require_elbow=True),
        backend="dry_run",
    )
    p = TeleopPipeline(cfg)
    res = p.step(_lm(1.0), now=1.0)
    assert res.retargeted.metadata.get("not_full_sew_mimic") is True
    assert res.gated.is_commandable() or res.safety is not None


def test_invalid_landmarks_cannot_command_motion():
    p = TeleopPipeline()
    p.step(_lm(1.0), now=1.0)
    # Clear and send invalid
    bad = ArmLandmarks(shoulder=None, elbow=None, wrist=None, timestamp=2.0)
    # Force stop by waiting past hold
    p.gate.config.hold_interval_sec = 0.0
    res = p.step(bad, now=2.0)
    assert not res.commanded


def test_moveit_backend_disabled_by_default():
    b = create_backend("optional_moveit")
    r = b.send_cartesian_target([0.4, 0.0, 0.4])
    assert not r.accepted


def test_config_yaml_validates():
    for name in ("mapping.yaml", "safety.yaml"):
        path = ROOT / "src" / "vision_arm_control" / "config" / name
        data = yaml.safe_load(path.read_text())
        assert isinstance(data, dict)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        TeleopPipeline(PipelineConfig(retargeting=RetargetingConfig(name="not_a_real_strategy")))
