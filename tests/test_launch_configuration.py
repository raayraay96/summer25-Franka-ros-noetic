"""Static validation of launch and YAML config files (no ROS required)."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "vision_arm_control"
LAUNCH = PKG / "launch"
CONFIG = PKG / "config"


def test_launch_files_parse_as_xml():
    files = list(LAUNCH.glob("*.launch"))
    assert files, "expected launch files"
    for f in files:
        tree = ET.parse(f)
        assert tree.getroot().tag == "launch", f


def test_mimicry_defaults_disable_robot():
    text = (LAUNCH / "mimicry.launch").read_text()
    assert 'name="use_robot" default="false"' in text


def test_yaml_configs_parse():
    for name in (
        "camera.yaml",
        "perception.yaml",
        "mapping.yaml",
        "safety.yaml",
        "controller.yaml",
        "models.yaml",
        "camera_intrinsics.yaml",
    ):
        path = CONFIG / name
        data = yaml.safe_load(path.read_text())
        assert isinstance(data, dict), name


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
