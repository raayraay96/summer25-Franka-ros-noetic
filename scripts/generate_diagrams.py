#!/usr/bin/env python3
"""Generate architecture / topic / frame SVG diagrams programmatically."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def svg_box(x, y, w, h, label, fill="#e8f1ff", stroke="#1f4e79") -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        f'<text x="{x + w/2}" y="{y + h/2 + 5}" text-anchor="middle" '
        f'font-family="Helvetica,Arial,sans-serif" font-size="14" fill="#123">{label}</text>'
    )


def svg_arrow(x1, y1, x2, y2) -> str:
    return (
        f'<defs><marker id="arrow" markerWidth="10" markerHeight="10" '
        f'refX="8" refY="3" orient="auto" markerUnits="strokeWidth">'
        f'<path d="M0,0 L0,6 L9,3 z" fill="#333"/></marker></defs>'
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>'
    )


def write_architecture() -> None:
    boxes = [
        (40, 40, 140, 50, "Camera / rosbag"),
        (240, 40, 160, 50, "Pose estimator"),
        (460, 40, 160, 50, "Depth (relative)"),
        (240, 140, 200, 50, "Human→robot mapper"),
        (500, 140, 160, 50, "Safety monitor"),
        (280, 240, 200, 50, "Controller (dry_run)"),
        (280, 340, 200, 50, "Sim / Franka (opt)"),
    ]
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="420" viewBox="0 0 720 420">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="360" y="24" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" '
        'font-size="16" font-weight="bold" fill="#111">Vision-Guided Franka Teleoperation</text>',
    ]
    # arrows between stages (simple vertical/horizontal)
    parts.append(svg_arrow(180, 65, 240, 65))
    parts.append(svg_arrow(400, 65, 460, 65))
    parts.append(svg_arrow(320, 90, 320, 140))
    parts.append(svg_arrow(440, 165, 500, 165))
    parts.append(svg_arrow(580, 190, 380, 240))
    parts.append(svg_arrow(380, 290, 380, 340))
    for b in boxes:
        parts.append(svg_box(*b))
    parts.append(
        '<text x="40" y="400" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#444">'
        "Default control mode: dry_run. Real robot disabled unless use_robot:=true.</text>"
    )
    parts.append("</svg>")
    (DOCS / "architecture.svg").write_text("\n".join(parts) + "\n")


def write_topic_graph() -> None:
    content = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="780" height="360" viewBox="0 0 780 360">
  <rect width="100%" height="100%" fill="white"/>
  <text x="390" y="28" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="16" font-weight="bold">ROS topic graph (logical)</text>
  <text x="40" y="70" font-family="monospace" font-size="13">/camera/.../compressed  →  pose_estimator_node  →  ~/landmarks (JSON String)</text>
  <text x="40" y="100" font-family="monospace" font-size="13">landmarks  →  human_to_robot_mapper_node  →  ~/target_pose (PoseStamped)</text>
  <text x="40" y="130" font-family="monospace" font-size="13">target_pose  →  safety_monitor_node  →  /teleop/command_pose</text>
  <text x="40" y="160" font-family="monospace" font-size="13">command_pose  →  robot_controller_node  →  ~/status, ~/executed_pose</text>
  <text x="40" y="190" font-family="monospace" font-size="13">image  →  depth_estimator_node  →  ~/depth (32FC1 relative), ~/depth_colormap</text>
  <text x="40" y="240" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#444">Mock path: mock_landmark_publisher → mapper (no camera required).</text>
  <text x="40" y="270" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#444">Safety: ~/emergency_stop (Bool), ~/deadman (Bool) into safety_monitor_node.</text>
</svg>
"""
    (DOCS / "topic-graph.svg").write_text(content)


def write_frame_tree() -> None:
    content = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="640" height="300" viewBox="0 0 640 300">
  <rect width="100%" height="100%" fill="white"/>
  <text x="320" y="28" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="16" font-weight="bold">Frame tree (conceptual)</text>
  <text x="80" y="90" font-family="monospace" font-size="14">panda_link0 (base)</text>
  <text x="100" y="120" font-family="monospace" font-size="14">└─ camera_optical_frame   [EXAMPLE static TF — calibrate]</text>
  <text x="100" y="150" font-family="monospace" font-size="14">└─ panda_link1 … panda_link8 (robot model / MoveIt)</text>
  <text x="80" y="210" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#444">
    Normalized image coords are NOT base_link coordinates. Convert via intrinsics + extrinsics.
  </text>
</svg>
"""
    (DOCS / "frame-tree.svg").write_text(content)


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    write_architecture()
    write_topic_graph()
    write_frame_tree()
    print("Wrote docs/architecture.svg docs/topic-graph.svg docs/frame-tree.svg")


if __name__ == "__main__":
    main()
