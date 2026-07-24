"""Model configuration path resolution tests."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vision_arm_control" / "src"))

from vision_arm_control.model_config import expand_path, load_model_paths, resolve_model_root  # noqa: E402


def test_expand_path_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_MODELS", str(tmp_path))
    p = expand_path("$MY_MODELS/encoder.pth")
    assert p == (tmp_path / "encoder.pth").resolve()


def test_resolve_prefers_explicit(tmp_path):
    root = resolve_model_root(str(tmp_path))
    assert root == tmp_path.resolve()


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
