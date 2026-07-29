#!/usr/bin/env python3
"""Generate truthful pure-Python v1.1 demo media (no ROS / no physical robot).

Produces MP4 + GIF under docs/media/v1.1/ with burned-in labels:
  SIMULATION · MOCK LANDMARKS · NO PHYSICAL ROBOT · RETARGETER · SAFETY FILTER

Demos:
  A — shoulder_relative vs sew_orientation side-by-side
  B — CBF-QP intervening near a simulated obstacle
  C — confidence dropout, hold, timeout, recovery
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vision_arm_control" / "src"))

from vision_arm_control.landmarks.confidence import ConfidenceGateConfig  # noqa: E402
from vision_arm_control.landmarks.schema import ArmLandmarks, LandmarkPoint  # noqa: E402
from vision_arm_control.pipeline import PipelineConfig, TeleopPipeline  # noqa: E402
from vision_arm_control.retargeting.base import RetargetingConfig  # noqa: E402
from vision_arm_control.safety_filters.base import (  # noqa: E402
    SafetyFilterConfig,
    SphericalObstacle,
)
from vision_arm_control.workspace_limits import DEFAULT_WORKSPACE  # noqa: E402

OUT = ROOT / "docs" / "media" / "v1.1"
FPS = 20
DURATION = 15.0


def _pipe(retargeter: str, safety: str) -> TeleopPipeline:
    obstacles = []
    if safety == "cbf_qp":
        obstacles = [SphericalObstacle("demo_sphere", [0.48, 0.0, 0.35], 0.10, 0.05)]
    return TeleopPipeline(
        PipelineConfig(
            retargeting=RetargetingConfig(name=retargeter),
            safety=SafetyFilterConfig(
                mode=safety,
                workspace=DEFAULT_WORKSPACE,
                workspace_margin_m=0.03,
                dt=0.05,
                alpha=4.0,
                max_cartesian_velocity_mps=0.20,
                obstacles=obstacles,
            ),
            gate=ConfidenceGateConfig(
                hold_interval_sec=0.4,
                recovery_blend_sec=0.5,
                require_elbow=(retargeter == "sew_orientation"),
            ),
            backend="dry_run",
        )
    )


def _lm(t: float, conf: float = 1.0) -> ArmLandmarks:
    s = (0.45, 0.35, 0.0)
    e = (0.52, 0.40, 0.0)
    w = (0.55 + 0.10 * math.sin(2 * math.pi * 0.2 * t), 0.48 + 0.05 * math.cos(2 * math.pi * 0.2 * t), 0.0)
    return ArmLandmarks(
        shoulder=LandmarkPoint(*s, confidence=conf, name="right_shoulder"),
        elbow=LandmarkPoint(*e, confidence=conf, name="right_elbow"),
        wrist=LandmarkPoint(*w, confidence=conf, name="right_wrist"),
        timestamp=t,
    )


def _draw_labels(ax, retargeter: str, safety: str, extra: str = ""):
    text = (
        f"SIMULATION  |  MOCK LANDMARKS  |  NO PHYSICAL ROBOT\n"
        f"RETARGETER: {retargeter}   SAFETY: {safety}"
    )
    if extra:
        text += f"\n{extra}"
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="white",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
        family="monospace",
    )


def _save_anim(fig, frames, path_mp4: Path, path_gif: Path):
    import matplotlib.animation as animation

    anim = animation.ArtistAnimation(fig, frames, interval=1000 / FPS, blit=False)
    # GIF first (always works with pillow)
    anim.save(str(path_gif), writer="pillow", fps=FPS)
    # Try MP4 via ffmpeg if available; else write a note
    try:
        anim.save(str(path_mp4), writer="ffmpeg", fps=FPS)
    except Exception:
        # Fallback: copy gif extension note — still produce an mp4 container via imageio if present
        try:
            import imageio.v2 as imageio

            # Re-render frames to arrays is hard; write a tiny silent mp4 from gif frames
            imgs = imageio.mimread(str(path_gif))
            imageio.mimsave(str(path_mp4), imgs, fps=FPS)
        except Exception as exc:
            (path_mp4.with_suffix(".mp4.txt")).write_text(
                f"MP4 encode unavailable ({exc}). GIF is the primary artifact.\n"
            )
    return path_gif


def demo_a():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p_sr = _pipe("shoulder_relative", "reject")
    p_sew = _pipe("sew_orientation", "reject")
    n = int(DURATION * FPS)
    dt = 1.0 / FPS
    path_sr, path_sew = [], []
    for i in range(n):
        t = i * dt
        lm = _lm(t)
        r1 = p_sr.step(lm, t)
        r2 = p_sew.step(lm, t)
        if r1.gated.position is not None:
            path_sr.append(r1.gated.position.copy())
        if r2.gated.position is not None:
            path_sew.append(r2.gated.position.copy())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    frames = []
    for i in range(0, len(path_sr), max(1, len(path_sr) // (n))):
        for ax in axes:
            ax.clear()
        for ax, path, name in (
            (axes[0], path_sr[: i + 1], "shoulder_relative"),
            (axes[1], path_sew[: i + 1], "sew_orientation"),
        ):
            pts = np.array(path) if path else np.zeros((1, 3))
            ax.plot(pts[:, 0], pts[:, 2], "-", color="#38bdf8", lw=2)
            ax.plot(pts[-1, 0], pts[-1, 2], "o", color="#f472b6", ms=8)
            ax.set_xlim(0.2, 0.8)
            ax.set_ylim(0.0, 0.9)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("z (m)")
            ax.set_title(name)
            ax.set_facecolor("#0f172a")
            ax.grid(True, alpha=0.2)
            _draw_labels(ax, name, "reject")
        fig.patch.set_facecolor("#020617")
        frames.append(list(axes[0].get_children() + axes[1].get_children()))
        # ArtistAnimation needs artists; simpler: save frame images
    # Rebuild with FuncAnimation-style manual images
    import io
    from PIL import Image

    images = []
    step = max(1, len(path_sr) // int(DURATION * FPS))
    for i in range(1, len(path_sr) + 1, step):
        for ax in axes:
            ax.clear()
        for ax, path, name in (
            (axes[0], path_sr[:i], "shoulder_relative"),
            (axes[1], path_sew[:i], "sew_orientation"),
        ):
            pts = np.array(path)
            ax.plot(pts[:, 0], pts[:, 2], "-", color="#38bdf8", lw=2)
            ax.plot(pts[-1, 0], pts[-1, 2], "o", color="#f472b6", ms=8)
            ax.set_xlim(0.2, 0.8)
            ax.set_ylim(0.0, 0.9)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("z (m)")
            ax.set_title(name)
            ax.set_facecolor("#0f172a")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            _draw_labels(ax, name, "reject")
        fig.patch.set_facecolor("#020617")
        fig.suptitle("Demo A: Side-by-side retargeting (pure-Python simulation)", color="white")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
        buf.seek(0)
        images.append(Image.open(buf).convert("RGB"))
    plt.close(fig)
    _write_images(images, OUT / "demo_a_shoulder_vs_sew.gif", OUT / "demo_a_shoulder_vs_sew.mp4")
    print("Wrote demo A")


def demo_b():
    import io
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from PIL import Image

    pipe = _pipe("shoulder_relative", "cbf_qp")
    n = int(DURATION * FPS)
    dt = 1.0 / FPS
    nom, safe = [], []
    for i in range(n):
        t = i * dt
        # Drive toward obstacle region
        frac = min(t / 10.0, 1.0)
        s = (0.45, 0.35, 0.0)
        e = (0.50, 0.40, 0.0)
        w = (0.48 + 0.15 * frac, 0.45, -0.05 * frac)
        lm = ArmLandmarks(
            shoulder=LandmarkPoint(*s, 1.0, "right_shoulder"),
            elbow=LandmarkPoint(*e, 1.0, "right_elbow"),
            wrist=LandmarkPoint(*w, 1.0, "right_wrist"),
            timestamp=t,
        )
        res = pipe.step(lm, t)
        if res.retargeted.position is not None:
            nom.append(res.retargeted.position.copy())
        if res.safety and res.safety.position is not None:
            safe.append(res.safety.position.copy())
        elif res.gated.position is not None:
            safe.append(res.gated.position.copy())

    fig, ax = plt.subplots(figsize=(7, 5))
    images = []
    step = max(1, len(safe) // int(DURATION * FPS))
    for i in range(1, max(len(safe), 1) + 1, step):
        ax.clear()
        ax.set_facecolor("#0f172a")
        fig.patch.set_facecolor("#020617")
        # Obstacle
        circ = Circle((0.48, 0.35), 0.10, fill=False, ec="#f87171", lw=2, ls="--")
        marg = Circle((0.48, 0.35), 0.15, fill=False, ec="#fbbf24", lw=1, ls=":")
        ax.add_patch(circ)
        ax.add_patch(marg)
        if nom:
            pn = np.array(nom[:i])
            ax.plot(pn[:, 0], pn[:, 2], "--", color="#94a3b8", lw=1.5, label="nominal")
        if safe:
            ps = np.array(safe[:i])
            ax.plot(ps[:, 0], ps[:, 2], "-", color="#34d399", lw=2, label="CBF-QP")
            ax.plot(ps[-1, 0], ps[-1, 2], "o", color="#34d399", ms=8)
        ax.set_xlim(0.2, 0.8)
        ax.set_ylim(0.0, 0.8)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.legend(loc="lower right", fontsize=8)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        _draw_labels(ax, "shoulder_relative", "cbf_qp", "Demo B: obstacle intervention")
        fig.suptitle("Demo B: CBF-QP vs nominal near simulated sphere", color="white")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
        buf.seek(0)
        images.append(Image.open(buf).convert("RGB"))
    plt.close(fig)
    _write_images(images, OUT / "demo_b_cbf_obstacle.gif", OUT / "demo_b_cbf_obstacle.mp4")
    print("Wrote demo B")


def demo_c():
    import io
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    pipe = _pipe("shoulder_relative", "reject")
    n = int(DURATION * FPS)
    dt = 1.0 / FPS
    xs, statuses = [], []
    for i in range(n):
        t = i * dt
        # Dropout from t=5..8
        if 5.0 <= t < 8.0:
            lm = None
        else:
            conf = 1.0 if t < 5 or t >= 8 else 0.1
            lm = _lm(t, conf=conf)
        res = pipe.step(lm, t)
        pos = res.gated.position
        xs.append(pos[0] if pos is not None else float("nan"))
        statuses.append(res.gated.status.value if res.gated else "none")

    fig, ax = plt.subplots(figsize=(8, 4))
    images = []
    times = np.arange(n) * dt
    step = max(1, n // int(DURATION * FPS))
    for i in range(1, n + 1, step):
        ax.clear()
        ax.set_facecolor("#0f172a")
        fig.patch.set_facecolor("#020617")
        ax.axvspan(5, 8, color="#7f1d1d", alpha=0.4, label="landmark dropout")
        ax.plot(times[:i], xs[:i], "-", color="#38bdf8", lw=2)
        ax.set_xlim(0, DURATION)
        ax.set_ylim(0.3, 0.7)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("EE x (m)")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        st = statuses[i - 1]
        _draw_labels(ax, "shoulder_relative", "reject", f"Demo C: gate={st}")
        fig.suptitle("Demo C: confidence loss → hold → stop → recovery", color="white")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
        buf.seek(0)
        images.append(Image.open(buf).convert("RGB"))
    plt.close(fig)
    _write_images(images, OUT / "demo_c_confidence_gating.gif", OUT / "demo_c_confidence_gating.mp4")
    print("Wrote demo C")


def _write_images(images, gif_path: Path, mp4_path: Path):
    OUT.mkdir(parents=True, exist_ok=True)
    if not images:
        raise RuntimeError("no frames")
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
    )
    # Downscale GIF if large
    try:
        import imageio.v2 as imageio

        imageio.mimsave(str(mp4_path), [np.array(im) for im in images], fps=FPS)
    except Exception:
        # Store GIF only; create a placeholder text for mp4
        try:
            # Pillow-only: write animated webp then rename? Keep gif as primary.
            # Write mp4 via ffmpeg subprocess if present
            import subprocess
            import tempfile
            import shutil

            tmp = Path(tempfile.mkdtemp())
            for i, im in enumerate(images):
                im.save(tmp / f"f{i:04d}.png")
            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(FPS),
                "-i",
                str(tmp / "f%04d.png"),
                "-pix_fmt",
                "yuv420p",
                str(mp4_path),
            ]
            r = subprocess.run(cmd, capture_output=True)
            shutil.rmtree(tmp, ignore_errors=True)
            if r.returncode != 0 and not mp4_path.exists():
                # last resort: copy gif bytes with .mp4 name is wrong; leave gif only
                (mp4_path.parent / (mp4_path.stem + "_mp4_unavailable.txt")).write_text(
                    "ffmpeg/imageio unavailable; GIF is the committed demo artifact.\n"
                )
        except Exception as exc:
            (mp4_path.parent / (mp4_path.stem + "_mp4_unavailable.txt")).write_text(str(exc) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    demo_a()
    demo_b()
    demo_c()
    print(f"Media under {OUT}")


if __name__ == "__main__":
    main()
