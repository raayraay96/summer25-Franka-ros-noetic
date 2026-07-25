"""Depth preprocessing helpers for MonoDepth2-style models."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def preprocess_rgb(
    image_bgr: np.ndarray,
    width: int = 640,
    height: int = 192,
) -> np.ndarray:
    """Convert BGR uint8 image to NCHW float32 tensor batch [1,3,H,W] in [0,1].

    Matches the preprocessing used in the original research scripts.
    """
    import cv2

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("empty image")
    img = cv2.resize(image_bgr, (width, height))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    return np.expand_dims(img, axis=0)


def disparity_to_relative_depth(disp: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convert disparity map to relative depth (1/disp). Not metric."""
    d = np.asarray(disp, dtype=np.float32)
    return 1.0 / np.maximum(d, eps)


def normalize_depth_for_display(depth: np.ndarray) -> np.ndarray:
    """Normalize depth to uint8 for visualization only."""
    import cv2

    d = np.asarray(depth, dtype=np.float32)
    norm = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def sample_depth_at_pixel(
    depth_map: np.ndarray,
    u_px: float,
    v_px: float,
) -> float:
    """Nearest-neighbor sample depth map at pixel coordinates."""
    h, w = depth_map.shape[:2]
    u = int(round(u_px))
    v = int(round(v_px))
    u = min(max(u, 0), w - 1)
    v = min(max(v, 0), h - 1)
    return float(depth_map[v, u])


def expected_input_size() -> Tuple[int, int]:
    """Default MonoDepth2 inference size (width, height)."""
    return 640, 192
