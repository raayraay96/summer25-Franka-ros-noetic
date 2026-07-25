#!/usr/bin/env python3
"""Compose recruiter-ready v2 media from a raw RViz X11 capture.

Usage (inside container or host with ffmpeg + repo on PYTHONPATH):
  python3 scholar/postprocess_demo_v2.py \
    --raw /path/raw_demo.mp4 \
    --demo-dir docs/demo \
    --results-dir results/ros2 \
    --job-id 459321
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def ffprobe(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,codec_name,nb_frames,avg_frame_rate",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        str(path),
    ]
    return json.loads(subprocess.check_output(cmd, text=True))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True, help="Working directory for intermediates")
    ap.add_argument("--demo-dir", type=Path, required=True)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--job-id", default="local")
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
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
    width = int(stream["width"])
    height = int(stream["height"])
    print("raw:", json.dumps(info))

    # Crop RViz Displays/Views docks out of the viewport (left ~30%).
    # Keeps the full robot 3D view for recruiter-facing media.
    crop_left = int(round(width * 0.34))
    # Ensure even dimensions for yuv420p
    crop_w = width - crop_left
    if crop_w % 2:
        crop_w -= 1
    crop_h = height if height % 2 == 0 else height - 1
    cropped = out / "cropped_viewport.mp4"
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(raw),
            "-vf",
            f"crop={crop_w}:{crop_h}:{crop_left}:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(cropped),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    width, height = crop_w, crop_h
    print(f"cropped viewport: {width}x{height} (removed left {crop_left}px docks)")

    panel_w, panel_h = 260, 196
    fps = 24
    duration = 12.0
    n_frames = int(duration * fps)
    motion_scale = 0.22  # match teleop.yaml mock_landmark_publisher

    trail = []
    all_rgb = bytearray()
    for i in range(n_frames):
        t = i / fps
        wr = wrist_xy(t, scale=motion_scale, period=DEMO_PERIOD_SEC)
        trail.append(wr)
        if len(trail) > 40:
            trail = trail[-40:]
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
            str(fps),
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

    title = "Vision-Guided Franka Teleoperation"
    footer = "ROS 2 Humble | RViz 2 fake hardware | Mock landmark input | Purdue Scholar"
    panel_label = "Mock landmark input"
    panel_x = 16
    panel_y = max(70, height - panel_h - 58)
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font_b = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    # Escape for ffmpeg drawtext (spaces + specials)
    def esc(s: str) -> str:
        return (
            s.replace("\\", "\\\\")
            .replace(":", "\\:")
            .replace("'", "\\'")
            .replace("%", "\\%")
        )

    labeled = out / "labeled_v2.mp4"
    # Title/footer with safe margins that do not cover the robot workspace.
    fc = (
        f"[0:v]trim=duration={duration},setpts=PTS-STARTPTS,"
        f"drawbox=x=0:y=0:w=iw:h=52:color=black@0.50:t=fill,"
        f"drawbox=x=0:y=ih-46:w=iw:h=46:color=black@0.55:t=fill,"
        f"drawtext=fontfile={font_b}:text='{esc(title)}':fontcolor=white:fontsize=26:"
        f"x=(w-text_w)/2:y=12,"
        f"drawtext=fontfile={font}:text='{esc(footer)}':fontcolor=white:fontsize=15:"
        f"x=(w-text_w)/2:y=h-30[base];"
        f"[1:v]scale={panel_w}:{panel_h},format=rgba,"
        f"drawbox=x=0:y=0:w=iw:h=24:color=black@0.55:t=fill,"
        f"drawtext=fontfile={font}:text='{esc(panel_label)}':fontcolor=white:fontsize=13:x=6:y=4[pan];"
        f"[base][pan]overlay={panel_x}:{panel_y}:format=auto"
    )
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(cropped),
            "-i",
            str(panel_mp4),
            "-filter_complex",
            fc,
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "21",
            "-movflags",
            "+faststart",
            str(labeled),
        ]
    )

    mp4_out = demo / "ros2-franka-teleoperation-v2.mp4"
    crf = 21 if labeled.stat().st_size <= 9_500_000 else 23
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
            "-crf",
            str(crf),
            "-movflags",
            "+faststart",
            str(mp4_out),
        ]
    )

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
            "".join([f"[{i}:v]scale=384:-1[f{i}];" for i in range(5)])
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
    assert mp4_out.stat().st_size < 12_000_000, "MP4 too large"
    assert gif_out.stat().st_size < 15_000_000, "GIF too large"
    print("MEDIA_V2_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
