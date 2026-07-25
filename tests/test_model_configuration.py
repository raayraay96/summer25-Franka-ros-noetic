"""Model configuration path resolution tests."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vision_arm_control" / "src"))

from vision_arm_control.model_config import (  # noqa: E402
    expand_path,
    load_model_paths,
    resolve_model_root,
)


def test_expand_path_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_MODELS", str(tmp_path))
    p = expand_path("$MY_MODELS/encoder.pth")
    assert p == (tmp_path / "encoder.pth").resolve()


def test_expand_path_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    p = expand_path("~/models/encoder.pth")
    assert p == (tmp_path / "models" / "encoder.pth").resolve()


def test_resolve_prefers_explicit(tmp_path):
    root = resolve_model_root(str(tmp_path))
    assert root == tmp_path.resolve()


def test_resolve_prefers_franka_model_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("FRANKA_MODEL_DIR", str(tmp_path))
    root = resolve_model_root(None)
    assert root == tmp_path.resolve()


def test_resolve_prefers_rcac_scratch_layout(tmp_path, monkeypatch):
    monkeypatch.delenv("FRANKA_MODEL_DIR", raising=False)
    models = tmp_path / "franka-teleop-data" / "models"
    models.mkdir(parents=True)
    monkeypatch.setenv("RCAC_SCRATCH", str(tmp_path))
    root = resolve_model_root(None)
    assert root == models.resolve()


def test_load_model_paths_missing(tmp_path):
    paths = load_model_paths(model_root=str(tmp_path))
    missing = paths.missing()
    assert "encoder" in missing
    assert "depth" in missing


def test_load_model_paths_present(tmp_path):
    (tmp_path / "encoder.pth").write_bytes(b"x")
    (tmp_path / "depth.pth").write_bytes(b"y")
    paths = load_model_paths(model_root=str(tmp_path))
    assert paths.missing() == {}
    assert paths.missing(required_only=False)  # pose files still missing


def test_load_model_paths_optional_pose_present(tmp_path):
    for name in ("encoder.pth", "depth.pth", "pose_encoder.pth", "pose.pth"):
        (tmp_path / name).write_bytes(b"x")
    paths = load_model_paths(model_root=str(tmp_path))
    assert paths.missing(required_only=False) == {}
