"""Repository layout and hygiene guards (no ROS required)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_src_contains_only_vision_arm_control():
    entries = sorted(p.name for p in (ROOT / "src").iterdir() if not p.name.startswith("."))
    assert entries == ["vision_arm_control"], entries


def test_no_gitmodules():
    assert not (ROOT / ".gitmodules").exists()


def test_no_weights_directory_tracked():
    weights = ROOT / "src" / "vision_arm_control" / "weights"
    if weights.exists():
        pths = list(weights.rglob("*.pth")) + list(weights.rglob("*.pt"))
        assert not pths, f"binary weights present: {pths}"


def test_no_large_pth_in_working_tree():
    bad = []
    for p in ROOT.rglob("*"):
        if ".git" in p.parts:
            continue
        if p.suffix.lower() in {".pth", ".pt", ".onnx", ".bag"} and p.is_file():
            bad.append(p)
    assert not bad, f"forbidden binaries in tree: {bad}"


def test_package_version_aligned():
    import re

    pkg_xml = (ROOT / "src" / "vision_arm_control" / "package.xml").read_text()
    init = (ROOT / "src" / "vision_arm_control" / "src" / "vision_arm_control" / "__init__.py").read_text()
    xml_ver = re.search(r"<version>([^<]+)</version>", pkg_xml)
    py_ver = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', init)
    assert xml_ver and py_ver
    assert xml_ver.group(1) == py_ver.group(1)


def test_intrinsics_labeled_example():
    import yaml

    data = yaml.safe_load(
        (ROOT / "src" / "vision_arm_control" / "config" / "camera_intrinsics.yaml").read_text()
    )
    assert data["camera_intrinsics"]["status"] == "example"


def test_controller_defaults_safe():
    import yaml

    data = yaml.safe_load(
        (ROOT / "src" / "vision_arm_control" / "config" / "controller.yaml").read_text()
    )
    assert data["controller"]["mode"] == "dry_run"
    assert data["controller"]["use_robot"] is False
