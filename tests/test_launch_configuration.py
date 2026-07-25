"""Static validation of launch and YAML configuration files (no ROS required)."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "vision_arm_control"
LAUNCH = PKG / "launch"
CONFIG = PKG / "config"

NODE_CONFIGS = {
    "camera_node": "camera_node.yaml",
    "pose_estimator_node": "pose_estimator_node.yaml",
    "depth_estimator_node": "depth_estimator_node.yaml",
    "human_to_robot_mapper_node": "human_to_robot_mapper_node.yaml",
    "safety_monitor_node": "safety_monitor_node.yaml",
    "robot_controller_node": "robot_controller_node.yaml",
}


def test_launch_files_parse_as_xml():
    files = list(LAUNCH.glob("*.launch"))
    assert files, "expected launch files"
    for path in files:
        tree = ET.parse(path)
        assert tree.getroot().tag == "launch", path


def test_mimicry_defaults_disable_robot():
    text = (LAUNCH / "mimicry.launch").read_text()
    assert 'name="use_robot" default="false"' in text
    assert 'name="control_mode" default="dry_run"' in text


def test_yaml_configs_parse():
    names = (
        "camera.yaml",
        "perception.yaml",
        "mapping.yaml",
        "safety.yaml",
        "controller.yaml",
        "models.yaml",
        "camera_intrinsics.yaml",
        *NODE_CONFIGS.values(),
    )
    for name in names:
        path = CONFIG / name
        data = yaml.safe_load(path.read_text())
        assert isinstance(data, dict), name


def test_node_scoped_configs_match_private_parameter_names():
    camera = yaml.safe_load((CONFIG / "camera_node.yaml").read_text())
    pose = yaml.safe_load((CONFIG / "pose_estimator_node.yaml").read_text())
    depth = yaml.safe_load((CONFIG / "depth_estimator_node.yaml").read_text())
    mapper = yaml.safe_load((CONFIG / "human_to_robot_mapper_node.yaml").read_text())
    safety = yaml.safe_load((CONFIG / "safety_monitor_node.yaml").read_text())
    controller = yaml.safe_load((CONFIG / "robot_controller_node.yaml").read_text())

    assert camera["image_topic"].endswith("/compressed")
    assert pose["min_detection_confidence"] == 0.5
    assert depth["encoder_filename"] == "encoder.pth"
    assert mapper["mapping_mode"] == "shoulder_relative"
    assert mapper["use_metric_depth"] is False
    assert safety["allow_real_robot"] is False
    assert safety["control_mode"] == "dry_run"
    assert safety["command_stale_sec"] > 0
    assert controller["mode"] == "dry_run"
    assert controller["use_robot"] is False


def test_mimicry_loads_configs_inside_consuming_nodes():
    root = ET.parse(LAUNCH / "mimicry.launch").getroot()
    nodes = {node.attrib.get("name"): node for node in root.iter("node")}
    for node_name, config_name in NODE_CONFIGS.items():
        assert node_name in nodes, node_name
        rosparams = nodes[node_name].findall("rosparam")
        assert any(
            config_name in rosparam.attrib.get("file", "") for rosparam in rosparams
        ), f"{node_name} does not load {config_name} in its private namespace"


def test_safety_defaults_reject_real_robot():
    data = yaml.safe_load((CONFIG / "safety.yaml").read_text())
    assert data["safety"]["allow_real_robot"] is False


def test_controller_default_dry_run():
    data = yaml.safe_load((CONFIG / "controller.yaml").read_text())
    assert data["controller"]["mode"] == "dry_run"
    assert data["controller"]["use_robot"] is False


def test_intrinsics_labeled_example():
    data = yaml.safe_load((CONFIG / "camera_intrinsics.yaml").read_text())
    assert data["camera_intrinsics"]["status"] == "example"


def test_no_personal_absolute_paths_in_configs_and_launch():
    needle = "/" + "home" + "/edr"
    for path in list(LAUNCH.glob("*")) + list(CONFIG.glob("*")):
        text = path.read_text()
        assert needle not in text, path
