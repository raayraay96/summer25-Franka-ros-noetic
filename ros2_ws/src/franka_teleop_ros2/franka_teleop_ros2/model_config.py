"""Model path and configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class ModelPaths:
    encoder: Path
    depth: Path
    pose_encoder: Optional[Path] = None
    pose: Optional[Path] = None

    def missing(self, required_only: bool = True) -> Dict[str, str]:
        """Return missing model files.

        By default only encoder/depth are required. Pose weights are optional
        research artifacts.
        """
        out: Dict[str, str] = {}
        items = [("encoder", self.encoder), ("depth", self.depth)]
        if not required_only:
            items.extend(
                [
                    ("pose_encoder", self.pose_encoder),
                    ("pose", self.pose),
                ]
            )
        for name, path in items:
            if path is None:
                continue
            if not path.is_file():
                out[name] = str(path)
        return out


def expand_path(path: str) -> Path:
    """Expand ~ and environment variables in a path string."""
    return Path(os.path.expanduser(os.path.expandvars(path))).resolve()


def resolve_model_root(explicit: Optional[str] = None) -> Path:
    """Resolve the model directory without hardcoding personal home paths.

    Priority:
      1. explicit argument
      2. FRANKA_MODEL_DIR env
      3. $RCAC_SCRATCH/franka-teleop-data/models
      4. package-relative weights/ (local dev)
    """
    if explicit:
        return expand_path(explicit)
    env = os.environ.get("FRANKA_MODEL_DIR")
    if env:
        return expand_path(env)
    scratch = os.environ.get("RCAC_SCRATCH")
    if scratch:
        candidate = Path(scratch) / "franka-teleop-data" / "models"
        if candidate.is_dir():
            return candidate.resolve()
    # package-relative: .../vision_arm_control/weights
    here = Path(__file__).resolve()
    pkg_weights = here.parents[2] / "weights"
    return pkg_weights


def load_model_paths(
    model_root: Optional[str] = None,
    encoder_name: str = "encoder.pth",
    depth_name: str = "depth.pth",
    pose_encoder_name: str = "pose_encoder.pth",
    pose_name: str = "pose.pth",
) -> ModelPaths:
    root = resolve_model_root(model_root)
    return ModelPaths(
        encoder=root / encoder_name,
        depth=root / depth_name,
        pose_encoder=root / pose_encoder_name,
        pose=root / pose_name,
    )
