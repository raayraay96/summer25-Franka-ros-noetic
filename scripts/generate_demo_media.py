#!/usr/bin/env python3
"""Generate honest no-hardware portfolio media from deterministic synthetic inputs."""
from __future__ import annotations

import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vision_arm_control" / "src"))
from vision_arm_control.workspace_limits import DEFAULT_WORKSPACE, validate_or_clamp_position  # noqa: E402

WIDTH, HEIGHT, FPS, FRAMES = 720, 405, 8, 72
OUT = ROOT / "docs" / "media"
FONT = ImageFont.load_default()


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, fill=(235, 240, 245)) -> None:
    draw.text(xy, value, font=FONT, fill=fill)


def encode(stem: str, frames: list[Image.Image]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gif = OUT / f"{stem}.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=1000 // FPS, loop=0, optimize=True)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to generate MP4 media")
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for index, frame in enumerate(frames):
            frame.save(tmpdir / f"frame-{index:04d}.png")
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(FPS),
                "-i",
                str(tmpdir / "frame-%04d.png"),
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(OUT / f"{stem}.mp4"),
            ],
            check=True,
        )


def safety_frames() -> list[Image.Image]:
    frames: list[Image.Image] = []
    bounds = DEFAULT_WORKSPACE
    for index in range(FRAMES):
        phase = 2.0 * math.pi * index / FRAMES
        target = np.array([0.50 + 0.34 * math.sin(phase), 0.28 * math.sin(2.0 * phase), 0.42])
        accepted, ok, reason = validate_or_clamp_position(target, bounds, mode="reject")
        image = Image.new("RGB", (WIDTH, HEIGHT), (17, 24, 34))
        draw = ImageDraw.Draw(image)
        text(draw, (24, 18), "SAFETY MONITOR • WORKSPACE REJECTION", (255, 255, 255))
        text(
            draw,
            (24, 38),
            "Synthetic target • production workspace utility • no ROS / no hardware",
            (176, 190, 205),
        )
        left, top, right, bottom = 90, 88, 630, 330
        draw.rectangle((left, top, right, bottom), outline=(80, 170, 130), width=3)
        text(draw, (left, top - 18), "allowed x/y workspace projection", (130, 220, 180))

        def project(point: np.ndarray) -> tuple[int, int]:
            x = left + int((point[0] - bounds.x_min) / (bounds.x_max - bounds.x_min) * (right - left))
            y = bottom - int((point[1] - bounds.y_min) / (bounds.y_max - bounds.y_min) * (bottom - top))
            return x, y

        px, py = project(target)
        draw.line((WIDTH // 2, HEIGHT // 2, px, py), fill=(110, 125, 145), width=2)
        draw.ellipse((px - 9, py - 9, px + 9, py + 9), fill=(240, 185, 70), outline=(255, 235, 170))
        text(draw, (px + 12, py - 8), "target", (255, 220, 140))
        if ok and accepted is not None:
            ax, ay = project(accepted)
            draw.ellipse((ax - 5, ay - 5, ax + 5, ay + 5), fill=(70, 220, 145))
            status, status_color = "ACCEPTED", (70, 220, 145)
        else:
            status, status_color = "REJECTED: outside_workspace", (245, 95, 95)
        draw.rounded_rectangle((185, 350, 535, 389), radius=8, outline=status_color, width=2)
        text(draw, (210, 363), status, status_color)
        frames.append(image)
    return frames


def perception_frames() -> list[Image.Image]:
    frames: list[Image.Image] = []
    for index in range(FRAMES):
        phase = 2.0 * math.pi * index / FRAMES
        image = Image.new("RGB", (WIDTH, HEIGHT), (16, 22, 31))
        draw = ImageDraw.Draw(image)
        text(draw, (24, 18), "PERCEPTION-ONLY • HEADLESS OUTPUT ARTIFACT", (255, 255, 255))
        text(
            draw,
            (24, 38),
            "Synthetic landmarks + relative 1/disparity • not camera or metric-depth proof",
            (176, 190, 205),
        )
        # Landmark panel
        panel = (30, 78, 345, 350)
        draw.rounded_rectangle(panel, radius=10, fill=(26, 36, 49), outline=(79, 104, 130), width=2)
        shoulder = (182, 155)
        elbow = (220 + int(28 * math.sin(phase)), 220)
        wrist = (260 + int(55 * math.sin(phase)), 268 + int(18 * math.cos(phase)))
        head = (182, 112)
        hip = (182, 272)
        for a, b in [(head, shoulder), (shoulder, elbow), (elbow, wrist), (shoulder, hip)]:
            draw.line((*a, *b), fill=(115, 215, 255), width=5)
        for point, label in [(shoulder, "shoulder"), (elbow, "elbow"), (wrist, "wrist")]:
            x, y = point
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(255, 205, 85))
            text(draw, (x + 9, y - 6), label, (230, 235, 240))
        # Relative depth panel
        x0, y0, x1, y1 = 375, 78, 690, 242
        for x in range(x0, x1, 4):
            for y in range(y0, y1, 4):
                dx = (x - (520 + 40 * math.sin(phase))) / 130.0
                dy = (y - 160) / 90.0
                disparity = 0.25 + 0.75 * math.exp(-(dx * dx + dy * dy))
                relative_depth = 1.0 / max(disparity, 1e-6)
                t = min(1.0, (relative_depth - 1.0) / 3.0)
                color = (int(55 + 180 * t), int(80 + 100 * (1 - t)), int(220 - 150 * t))
                draw.rectangle((x, y, x + 4, y + 4), fill=color)
        draw.rectangle((x0, y0, x1, y1), outline=(79, 104, 130), width=2)
        text(draw, (382, 250), "relative_depth = 1 / disparity (unitless)", (210, 220, 230))
        nx = (wrist[0] - panel[0]) / (panel[2] - panel[0])
        ny = (wrist[1] - panel[1]) / (panel[3] - panel[1])
        payload = [
            '{"stamp": "synthetic",',
            ' "frame_id": "camera_optical_frame",',
            f' "wrist": {{"x": {nx:.3f}, "y": {ny:.3f}, "confidence": 0.99}}}}',
        ]
        draw.rounded_rectangle((375, 282, 690, 350), radius=8, fill=(23, 31, 42), outline=(79, 104, 130))
        for row, value in enumerate(payload):
            text(draw, (388, 294 + 16 * row), value, (145, 220, 180))
        frames.append(image)
    return frames


def main() -> None:
    encode("safety-workspace-violation", safety_frames())
    encode("perception-headless-relative-depth", perception_frames())
    for path in sorted(OUT.glob("*.gif")) + sorted(OUT.glob("*.mp4")):
        print(path.relative_to(ROOT), path.stat().st_size)


if __name__ == "__main__":
    main()
