"""Verify teleop.yaml structure matches node parameter names."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config" / "teleop.yaml"


def test_param_file_has_node_namespaces():
    data = yaml.safe_load(CFG.read_text())
    for node in (
        "mock_landmark_publisher",
        "human_to_robot_mapper",
        "safety_monitor",
        "robot_controller",
        "target_visualizer",
        "diagnostics",
    ):
        assert node in data, node
        assert "ros__parameters" in data[node], node


def test_safety_defaults_disable_hardware():
    data = yaml.safe_load(CFG.read_text())
    p = data["safety_monitor"]["ros__parameters"]
    assert p["allow_real_robot"] is False
    assert p["control_mode"] == "simulation"


def test_mapper_does_not_use_metric_depth_by_default():
    data = yaml.safe_load(CFG.read_text())
    p = data["human_to_robot_mapper"]["ros__parameters"]
    assert p["use_metric_depth"] is False
    assert p["mapping_mode"] == "shoulder_relative"


def test_controller_not_real():
    data = yaml.safe_load(CFG.read_text())
    p = data["robot_controller"]["ros__parameters"]
    assert p["allow_real_robot"] is False
