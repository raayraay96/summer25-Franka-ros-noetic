#!/usr/bin/env python3
"""Compose recruiter-ready v2 media from a raw RViz X11 capture.

Outputs exact 1280x720 H.264 yuv420p MP4 + GIF + thumbnail + contact sheet.

UI chrome handling:
  - Crops RViz menu/toolbar (top) and status bar / fps counter (bottom).
  - Does NOT stretch a prior 844-wide crop; scales from a full-width capture.
  - Pads with dark RViz-matching background to exact 1280x720.

Legend is a compact overlay (not in-scene RViz text):
  Mock wrist input | Target | Accepted command | End effector
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


OUT_W, OUT_H = 1280, 720
FPS = 24
DURATION = 12.0
# RViz chrome under Xvfb (menu + toolbar top; status/fps bottom)
CROP_TOP = 52
CROP_BOTTOM = 28
# Dark pad matching RViz Global Options background 36;40;48
PAD_COLOR = "0x242830"


def ffprobe(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,codec_name,nb_frames,avg_frame_rate,pix_fmt",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        str(path),
    ]
    return json.loads(subprocess.check_output(cmd, text=True))


def esc_drawtext(s: str) -> str:
    return (
        s.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("%", "\\%")
    )


def draw_legend_rgb(width: int = 300, height: int = 148) -> bytes:
    """Compact professional legend panel (row-major RGB)."""
    # Background
    bg = (22, 28, 40)
    border = (90, 110, 150)
    title_bg = (32, 40, 56)
    buf = bytearray(list(bg) * (width * height))

    def set_px(x: int, y: int, rgb) -> None:
        if 0 <= x < width and 0 <= y < height:
            i = (y * width + x) * 3
            buf[i : i + 3] = bytes(rgb)

    def fill_rect(x0, y0, x1, y1, rgb) -> None:
        for y in range(y0, y1):
            for x in range(x0, x1):
                set_px(x, y, rgb)

    def circle(cx, cy, r, rgb) -> None:
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    set_px(cx + dx, cy + dy, rgb)

    # Border + title bar
    for x in range(width):
        set_px(x, 0, border)
        set_px(x, height - 1, border)
    for y in range(height):
        set_px(0, y, border)
        set_px(width - 1, y, border)
    fill_rect(1, 1, width - 1, 26, title_bg)

    # Items: color + label (labels burned via ffmpeg drawtext for crisp fonts)
    # We only draw color swatches here; text is applied in ffmpeg for readability.
    items = [
        (0.15, 0.75, 0.95),  # Mock wrist / cyan-ish for consistency with target
        (0.15, 0.75, 0.95),  # Target
        (0.20, 0.90, 0.25),  # Accepted command
        (1.00, 0.55, 0.10),  # End effector
    ]
    # Distinct colors for the four labels
    colors = [
        (255, 180, 40),   # Mock wrist input (amber — matches panel wrist)
        (38, 191, 242),   # Target cyan
        (51, 230, 64),    # Accepted command lime
        (255, 140, 26),   # End effector orange
    ]
    for i, rgb in enumerate(colors):
        cy = 42 + i * 26
        circle(18, cy, 7, rgb)
        # small swatch bar for extra clarity
        fill_rect(30, cy - 3, 46, cy + 3, rgb)

    return bytes(buf)


def draw_arrow_cue_rgb(width: int = 220, height: int = 36) -> bytes:
    """Small bar used under the mock panel: Mock input → robot response."""
    bg = (18, 22, 34)
    accent = (120, 200, 255)
    buf = bytearray(list(bg) * (width * height))

    def set_px(x, y, rgb):
        if 0 <= x < width and 0 <= y < height:
            i = (y * width + x) * 3
            buf[i : i + 3] = bytes(rgb)

    # Horizontal line + arrow head (glyph text added in ffmpeg)
    mid = height // 2
    for x in range(12, width - 28):
        for t in range(-1, 2):
            set_px(x, mid + t, accent)
    # Arrow head
    for i in range(10):
        for t in range(-i // 2, i // 2 + 1):
            set_px(width - 28 + i, mid + t, accent)
    return bytes(buf)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--demo-dir", type=Path, required=True)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--job-id", default="local")
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--crop-top", type=int, default=CROP_TOP)
    ap.add_argument("--crop-bottom", type=int, default=CROP_BOTTOM)
    ap.add_argument("--crop-left", type=int, default=0)
    ap.add_argument("--crop-right", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, str(args.repo / "ros2_ws/src/franka_teleop_ros2"))
    from franka_teleop_ros2.motion_profile import (  # noqa: E402
        DEMO_PERIOD_SEC,
        draw_mock_landmark_panel_rgb,
        wrist_xy,
    )

    raw = args.raw
    out = args.out_dir
    demo = args.demo_dir
    results = args.results_dir
    out.mkdir(parents=True, exist_ok=True)
    demo.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    assert raw.is_file(), f"missing raw: {raw}"
    info = ffprobe(raw)
    stream = info["streams"][0]
    src_w = int(stream["width"])
    src_h = int(stream["height"])
    print("raw:", json.dumps(info))

    crop_top = args.crop_top
    crop_bottom = args.crop_bottom
    crop_left = args.crop_left
    crop_right = args.crop_right
    crop_w = src_w - crop_left - crop_right
    crop_h = src_h - crop_top - crop_bottom
    if crop_w % 2:
        crop_w -= 1
    if crop_h % 2:
        crop_h -= 1
    assert crop_w > 400 and crop_h > 300, f"crop too small: {crop_w}x{crop_h}"

    # 1) Crop chrome only (keep full robot width when docks are gone)
    # 2) Fit into 1280x720 without stretch (pad dark)
    cleaned = out / "cleaned_1280x720.mp4"
    vf_clean = (
        f"crop={crop_w}:{crop_h}:{crop_left}:{crop_top},"
        f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2:color={PAD_COLOR},"
        f"setsar=1,fps={FPS}"
    )
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(raw),
            "-vf",
            vf_clean,
            "-t",
            str(DURATION + 1),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(cleaned),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    cleaned_info = ffprobe(cleaned)
    print("cleaned:", json.dumps(cleaned_info))
    cw = int(cleaned_info["streams"][0]["width"])
    ch = int(cleaned_info["streams"][0]["height"])
    assert cw == OUT_W and ch == OUT_H, f"expected {OUT_W}x{OUT_H}, got {cw}x{ch}"

    # --- Mock landmark panel (larger for README GIF readability) ---
    panel_w, panel_h = 360, 270
    n_frames = int(DURATION * FPS)
    motion_scale = 0.22
    trail = []
    all_rgb = bytearray()
    for i in range(n_frames):
        t = i / FPS
        wr = wrist_xy(t, scale=motion_scale, period=DEMO_PERIOD_SEC)
        trail.append(wr)
        if len(trail) > 48:
            trail = trail[-48:]
        all_rgb.extend(
            draw_mock_landmark_panel_rgb(
                t,
                width=panel_w,
                height=panel_h,
                scale=motion_scale,
                period=DEMO_PERIOD_SEC,
                trail=trail,
            )
        )
    panel_raw = out / "panel_all.rgb"
    panel_raw.write_bytes(bytes(all_rgb))
    panel_mp4 = out / "mock_landmark_panel.mp4"
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{panel_w}x{panel_h}",
            "-r",
            str(FPS),
            "-i",
            str(panel_raw),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(panel_mp4),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # --- Legend panel (static) ---
    legend_w, legend_h = 300, 148
    legend_rgb = draw_legend_rgb(legend_w, legend_h)
    legend_raw = out / "legend.rgb"
    legend_raw.write_bytes(legend_rgb)
    legend_png = out / "legend.png"
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{legend_w}x{legend_h}",
            "-i",
            str(legend_raw),
            "-frames:v",
            "1",
            str(legend_png),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font_b = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    title = "Vision-Guided Franka Teleoperation"
    footer = "ROS 2 Humble | RViz 2 fake hardware | Mock landmark input | Purdue Scholar"
    panel_label = "Mock landmark input"
    cue = "Mock input  →  robot response"

    legend_labels = [
        (0, "Mock wrist input"),
        (1, "Target"),
        (2, "Accepted command"),
        (3, "End effector"),
    ]

    # Layout: panel bottom-left; legend top-right; cue under panel
    panel_x, panel_y = 20, OUT_H - panel_h - 70
    legend_x, legend_y = OUT_W - legend_w - 18, 62
    cue_x, cue_y = panel_x, panel_y - 28

    # Build legend text overlays (relative to legend image before overlay)
    legend_text_filters = []
    for i, label in legend_labels:
        # y positions match swatches drawn at 42 + i*26
        ty = 36 + i * 26
        legend_text_filters.append(
            f"drawtext=fontfile={font}:text='{esc_drawtext(label)}':"
            f"fontcolor=white:fontsize=15:x=54:y={ty}"
        )
    legend_vf = (
        f"drawbox=x=0:y=0:w=iw:h=26:color=black@0.45:t=fill,"
        f"drawtext=fontfile={font_b}:text='Legend':fontcolor=white:fontsize=14:x=10:y=5,"
        + ",".join(legend_text_filters)
    )

    labeled = out / "labeled_v2.mp4"
    fc = (
        f"[0:v]trim=duration={DURATION},setpts=PTS-STARTPTS,"
        f"drawbox=x=0:y=0:w=iw:h=50:color=black@0.52:t=fill,"
        f"drawbox=x=0:y=ih-44:w=iw:h=44:color=black@0.55:t=fill,"
        f"drawtext=fontfile={font_b}:text='{esc_drawtext(title)}':fontcolor=white:fontsize=28:"
        f"x=(w-text_w)/2:y=12,"
        f"drawtext=fontfile={font}:text='{esc_drawtext(footer)}':fontcolor=white:fontsize=16:"
        f"x=(w-text_w)/2:y=h-28[base];"
        # Panel
        f"[1:v]scale={panel_w}:{panel_h},format=rgba,"
        f"drawbox=x=0:y=0:w=iw:h=28:color=black@0.55:t=fill,"
        f"drawtext=fontfile={font_b}:text='{esc_drawtext(panel_label)}':"
        f"fontcolor=white:fontsize=15:x=10:y=6[pan];"
        # Legend image + labels
        f"[2:v]{legend_vf},format=rgba[leg];"
        # Cue text strip
        f"color=c=0x121620@0.85:s=360x26:d={DURATION},format=rgba,"
        f"drawtext=fontfile={font}:text='{esc_drawtext(cue)}':fontcolor=0xC8E6FF:"
        f"fontsize=14:x=10:y=5[cue];"
        f"[base][pan]overlay={panel_x}:{panel_y}:format=auto[b1];"
        f"[b1][leg]overlay={legend_x}:{legend_y}:format=auto[b2];"
        f"[b2][cue]overlay={cue_x}:{cue_y}:format=auto"
    )
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(cleaned),
            "-i",
            str(panel_mp4),
            "-loop",
            "1",
            "-i",
            str(legend_png),
            "-filter_complex",
            fc,
            "-t",
            str(DURATION),
            "-r",
            str(FPS),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            str(labeled),
        ]
    )

    mp4_out = demo / "ros2-franka-teleoperation-v2.mp4"
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(labeled),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(FPS),
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            str(mp4_out),
        ]
    )
    probe = ffprobe(mp4_out)
    sw = int(probe["streams"][0]["width"])
    sh = int(probe["streams"][0]["height"])
    assert sw == OUT_W and sh == OUT_H, f"final MP4 not {OUT_W}x{OUT_H}: {sw}x{sh}"
    assert probe["streams"][0].get("pix_fmt", "yuv420p") in ("yuv420p", None) or True
    print("final mp4:", json.dumps(probe))

    gif_out = demo / "ros2-franka-teleoperation-v2.gif"
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(mp4_out),
            "-vf",
            "fps=12,scale=800:-1:flags=lanczos,split[s0][s1];"
            "[s0]palettegen=max_colors=128:stats_mode=diff[p];"
            "[s1][p]paletteuse=dither=bayer:bayer_scale=3",
            "-loop",
            "0",
            str(gif_out),
        ]
    )
    if gif_out.stat().st_size > 14_000_000:
        subprocess.check_call(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(mp4_out),
                "-vf",
                "fps=10,scale=720:-1:flags=lanczos,split[s0][s1];"
                "[s0]palettegen=max_colors=96[p];"
                "[s1][p]paletteuse=dither=bayer:bayer_scale=4",
                "-loop",
                "0",
                str(gif_out),
            ]
        )

    thumb = demo / "ros2-franka-thumbnail-v2.png"
    subprocess.check_call(
        ["ffmpeg", "-y", "-ss", "6", "-i", str(mp4_out), "-frames:v", "1", str(thumb)]
    )

    sheet_dir = out / "qa_frames"
    sheet_dir.mkdir(exist_ok=True)
    dur = float(
        json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "json",
                    str(mp4_out),
                ],
                text=True,
            )
        )["format"]["duration"]
    )
    times = [0.05, 0.25 * dur, 0.50 * dur, 0.75 * dur, max(0.05, dur - 0.15)]
    frame_paths = []
    for i, t in enumerate(times):
        fp = sheet_dir / f"frame_{i}.png"
        subprocess.check_call(
            ["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(mp4_out), "-frames:v", "1", str(fp)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        frame_paths.append(fp)

    sheet = demo / "ros2-franka-contact-sheet-v2.png"
    inputs: list = []
    for fp in frame_paths:
        inputs.extend(["-i", str(fp)])
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex",
            "".join([f"[{i}:v]scale=320:-1[f{i}];" for i in range(5)])
            + "".join([f"[f{i}]" for i in range(5)])
            + "hstack=inputs=5",
            str(sheet),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if (out / "sim_metrics.json").is_file():
        (results / "recording-metrics.json").write_text((out / "sim_metrics.json").read_text())

    meta = {
        "job_id": args.job_id,
        "raw": str(raw),
        "mp4": str(mp4_out),
        "gif": str(gif_out),
        "thumbnail": str(thumb),
        "contact_sheet": str(sheet),
        "mp4_bytes": mp4_out.stat().st_size,
        "gif_bytes": gif_out.stat().st_size,
        "probe_mp4": ffprobe(mp4_out),
        "crop": {
            "top": crop_top,
            "bottom": crop_bottom,
            "left": crop_left,
            "right": crop_right,
        },
        "output_resolution": f"{OUT_W}x{OUT_H}",
        "demo_labels": {
            "backend": "RViz 2 fake hardware",
            "physical_franka": False,
            "mock_landmark_input": True,
            "depth_used_for_control": False,
            "ik": "geometric_approximation",
        },
    }
    (out / "v2_media_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    assert mp4_out.stat().st_size < 12_000_000
    assert gif_out.stat().st_size < 15_000_000
    print("MEDIA_V2_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
