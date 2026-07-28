"""Shared mock-landmark motion profile for simulation demos.

The same analytic trajectory drives:
  - mock_landmark_publisher (ROS input)
  - optional mock-landmark panel rendering in the recording pipeline

Period is chosen so a 10–12 s GIF loop returns near the start pose.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


# Demo loop period (seconds). Keep in sync with recording length.
DEMO_PERIOD_SEC = 12.0


def wrist_xy(
    t: float,
    scale: float = 0.18,
    period: float = DEMO_PERIOD_SEC,
) -> Tuple[float, float]:
    """Normalized image-plane wrist position that closes after ``period`` seconds.

    Uses a 1:2 Lissajous path so the 2-D loop is seamless and produces
    obvious Y/Z workspace motion after shoulder-relative mapping.
    """
    phase = 2.0 * math.pi * ((t % period) / period)
    # Center of motion in normalized image coords
    cx, cy = 0.52, 0.44
    wx = cx + scale * math.sin(phase)
    wy = cy + scale * 0.75 * math.sin(2.0 * phase + 0.35)
    # Keep inside (0,1) with a small margin
    wx = min(0.92, max(0.08, wx))
    wy = min(0.88, max(0.12, wy))
    return float(wx), float(wy)


def elbow_xy(wrist: Tuple[float, float], shoulder: Tuple[float, float]) -> Tuple[float, float]:
    """Synthetic elbow halfway between shoulder and wrist with slight bend."""
    mx = 0.55 * shoulder[0] + 0.45 * wrist[0]
    my = 0.55 * shoulder[1] + 0.45 * wrist[1]
    # Offset perpendicular for a more arm-like look
    dx = wrist[0] - shoulder[0]
    dy = wrist[1] - shoulder[1]
    mx += -0.08 * dy
    my += 0.08 * dx
    return float(mx), float(my)


def shoulder_xy() -> Tuple[float, float]:
    return 0.42, 0.32


def landmark_payload(
    t: float,
    *,
    scale: float = 0.18,
    period: float = DEMO_PERIOD_SEC,
    image_width: int = 640,
    image_height: int = 480,
    stamp: float = 0.0,
) -> Dict:
    """Build the mock MediaPipe-like JSON dict used by the mapper."""
    sh = shoulder_xy()
    wr = wrist_xy(t, scale=scale, period=period)
    el = elbow_xy(wr, sh)
    return {
        "header": {
            "stamp": stamp if stamp else t,
            "frame_id": "camera_optical_frame",
            "image_width": image_width,
            "image_height": image_height,
        },
        "pose_detected": True,
        "landmarks": [
            {
                "name": "right_shoulder",
                "x": sh[0],
                "y": sh[1],
                "z": 0.0,
                "visibility": 1.0,
                "confidence": 1.0,
            },
            {
                "name": "right_elbow",
                "x": el[0],
                "y": el[1],
                "z": 0.0,
                "visibility": 1.0,
                "confidence": 1.0,
            },
            {
                "name": "right_wrist",
                "x": wr[0],
                "y": wr[1],
                "z": 0.0,
                "visibility": 1.0,
                "confidence": 1.0,
            },
        ],
        "coordinate_convention": "normalized_image_xy_in_0_1_mediapipe_z_relative",
        "source": "mock",
        "depth_used_for_control": False,
        "label": "Mock landmark input",
    }


def draw_mock_landmark_panel_rgb(
    t: float,
    *,
    width: int = 320,
    height: int = 240,
    scale: float = 0.18,
    period: float = DEMO_PERIOD_SEC,
    trail: Optional[List[Tuple[float, float]]] = None,
) -> bytes:
    """Render an RGB panel (row-major, 8-bit) showing the mock arm.

    Pure Python (no OpenCV) so it runs in minimal environments.
    Returns ``width * height * 3`` bytes.
    """
    # Dark navy background
    buf = bytearray([18, 24, 38] * (width * height))

    def set_px(x: int, y: int, rgb: Tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            i = (y * width + x) * 3
            buf[i : i + 3] = bytes(rgb)

    def draw_circle(cx: int, cy: int, r: int, rgb: Tuple[int, int, int]) -> None:
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    set_px(cx + dx, cy + dy, rgb)

    def draw_line(x0: int, y0: int, x1: int, y1: int, rgb: Tuple[int, int, int], thick: int = 2) -> None:
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for s in range(steps + 1):
            x = int(x0 + (x1 - x0) * s / steps)
            y = int(y0 + (y1 - y0) * s / steps)
            for tdy in range(-thick, thick + 1):
                for tdx in range(-thick, thick + 1):
                    set_px(x + tdx, y + tdy, rgb)

    sh = shoulder_xy()
    wr = wrist_xy(t, scale=scale, period=period)
    el = elbow_xy(wr, sh)

    def to_px(nx: float, ny: float) -> Tuple[int, int]:
        # Leave margin for title bar
        mx0, my0 = 16, 36
        mw, mh = width - 32, height - 52
        return int(mx0 + nx * mw), int(my0 + ny * mh)

    # Trail of wrist (caller can pass recent points)
    if trail:
        for i, (nx, ny) in enumerate(trail):
            px, py = to_px(nx, ny)
            fade = int(80 + 150 * (i + 1) / max(len(trail), 1))
            draw_circle(px, py, 2, (min(255, fade), 180, 60))

    sx, sy = to_px(*sh)
    ex, ey = to_px(*el)
    wx, wy = to_px(*wr)

    # Arm segments
    draw_line(sx, sy, ex, ey, (120, 200, 255), thick=3)
    draw_line(ex, ey, wx, wy, (120, 200, 255), thick=3)
    draw_circle(sx, sy, 7, (90, 170, 255))  # shoulder
    draw_circle(ex, ey, 6, (180, 220, 255))  # elbow
    draw_circle(wx, wy, 8, (255, 180, 40))  # wrist (tracked)

    # Title bar
    for y in range(0, 28):
        for x in range(width):
            set_px(x, y, (28, 36, 52))
    # Simple bitmap-ish title using block pixels is overkill; solid bar is fine.
    # Recording pipeline burns the "Mock landmark input" text with drawtext.

    # Border
    for x in range(width):
        set_px(x, 0, (80, 100, 140))
        set_px(x, height - 1, (80, 100, 140))
    for y in range(height):
        set_px(0, y, (80, 100, 140))
        set_px(width - 1, y, (80, 100, 140))

    return bytes(buf)
