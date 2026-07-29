"""Tests for the versioned teleop config schema (Phase 8)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vision_arm_control" / "src"))

from vision_arm_control.config_schema import (  # noqa: E402
    ConfigError,
    validate_teleop_config,
)

FIXTURES = ROOT / "src" / "vision_arm_control" / "config" / "fixtures"


def _load(name):
    return yaml.safe_load((FIXTURES / name).read_text())


@pytest.mark.parametrize(
    "name",
    [
        "default_safe.yaml",
        "shoulder_relative_demo.yaml",
        "sew_feature_demo.yaml",
        "cbf_test.yaml",
        "ros2_fake_hardware.yaml",
    ],
)
def test_fixtures_are_valid(name):
    validate_teleop_config(_load(name))


def _base():
    return _load("cbf_test.yaml")


def test_unknown_top_level_key_rejected():
    cfg = _base()
    cfg["bogus"] = 1
    with pytest.raises(ConfigError, match="unknown key"):
        validate_teleop_config(cfg)


def test_wrong_schema_version_rejected():
    cfg = _base()
    cfg["schema_version"] = "0.9"
    with pytest.raises(ConfigError, match="schema_version"):
        validate_teleop_config(cfg)


def test_missing_workspace_rejected():
    cfg = _base()
    del cfg["workspace"]
    with pytest.raises(ConfigError, match="workspace"):
        validate_teleop_config(cfg)


def test_bad_workspace_bounds_rejected():
    cfg = _base()
    cfg["workspace"]["x_min"] = 0.9  # >= x_max
    with pytest.raises(ConfigError, match="x_min"):
        validate_teleop_config(cfg)


def test_negative_dt_rejected():
    cfg = _base()
    cfg["dt"] = 0.0
    with pytest.raises(ConfigError, match="dt"):
        validate_teleop_config(cfg)


def test_bad_velocity_type_rejected():
    cfg = _base()
    cfg["safety"]["per_axis_velocity_limit_mps"] = "fast"
    with pytest.raises(ConfigError, match="per_axis_velocity_limit_mps"):
        validate_teleop_config(cfg)


def test_malformed_center_vector_rejected():
    cfg = _base()
    cfg["safety"]["obstacles"][0]["center"] = [0.1, 0.2]  # not length-3
    with pytest.raises(ConfigError, match="center"):
        validate_teleop_config(cfg)


def test_duplicate_obstacle_id_rejected():
    cfg = _base()
    dup = copy.deepcopy(cfg["safety"]["obstacles"][0])
    cfg["safety"]["obstacles"].append(dup)
    with pytest.raises(ConfigError, match="duplicate obstacle id"):
        validate_teleop_config(cfg)


def test_obstacles_with_non_cbf_mode_conflict():
    cfg = _base()
    cfg["safety"]["mode"] = "reject"  # obstacles present but ignored
    with pytest.raises(ConfigError, match="ignores obstacles"):
        validate_teleop_config(cfg)


def test_margin_larger_than_workspace_rejected():
    cfg = _base()
    cfg["safety"]["workspace_margin_m"] = 1.0  # bigger than half-extent
    with pytest.raises(ConfigError, match="workspace_margin_m"):
        validate_teleop_config(cfg)


def test_bad_backend_rejected():
    cfg = _base()
    cfg["backend"] = "real_robot"
    with pytest.raises(ConfigError, match="backend"):
        validate_teleop_config(cfg)


def test_bad_retargeter_rejected():
    cfg = _base()
    cfg["retargeter"]["name"] = "magic"
    with pytest.raises(ConfigError, match="retargeter.name"):
        validate_teleop_config(cfg)


def test_bad_frame_name_rejected():
    cfg = _load("shoulder_relative_demo.yaml")
    cfg["frame_mapping"]["source_frame"] = "camera_left"
    with pytest.raises(ConfigError, match="source_frame"):
        validate_teleop_config(cfg)
