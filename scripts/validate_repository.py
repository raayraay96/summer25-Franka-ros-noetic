#!/usr/bin/env python3
"""Validate configs, launch XML, README links, and portfolio hygiene without ROS."""
from __future__ import annotations
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "vision_arm_control"
for path in sorted((PKG / "config").glob("*.yaml")):
    assert isinstance(yaml.safe_load(path.read_text()), dict), path
for path in sorted((PKG / "launch").glob("*.launch")):
    assert ET.parse(path).getroot().tag == "launch", path
readme = (ROOT / "README.md").read_text()
links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", readme)
links += re.findall(r'<img[^>]+src="([^"]+)"', readme)
for target in links:
    clean = target.split("#", 1)[0]
    if not clean or clean.startswith(("http://", "https://", "mailto:")):
        continue
    assert (ROOT / clean).exists(), f"README link does not resolve: {target}"
for required in [
    "docs/architecture.svg",
    "docs/topic-graph.svg",
    "docs/frame-tree.svg",
    "docs/repository-audit.md",
    "docs/safety.md",
    "docs/calibration.md",
    "docs/model-setup.md",
    "CITATION.cff",
    "LICENSE",
]:
    assert (ROOT / required).exists(), required
for path in ROOT.rglob("*"):
    if not path.is_file() or ".git" in path.parts:
        continue
    assert path.stat().st_size <= 10 * 1024 * 1024, f"file exceeds 10 MiB: {path}"
    if path.suffix.lower() in {".pt", ".pth", ".bag", ".onnx"}:
        raise AssertionError(f"forbidden binary in working tree: {path}")
placeholder = "your" + "username"
for path in ROOT.rglob("*"):
    if path.is_file() and path.suffix.lower() not in {".gif", ".mp4", ".png", ".jpg", ".jpeg"}:
        assert placeholder not in path.read_text(errors="ignore"), path
assert len(readme.splitlines()) < 400, "README must remain under 400 lines"
print(f"validated {len(links)} README links")
