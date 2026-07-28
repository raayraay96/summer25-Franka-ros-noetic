#!/usr/bin/env python3
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "ros2-architecture.svg"
OUT.write_text(
    """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="760" height="420" viewBox="0 0 760 420">
  <rect width="100%" height="100%" fill="white"/>
  <text x="380" y="28" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="16" font-weight="bold">
    ROS 2 Franka Teleop Simulation (Humble)
  </text>
  <rect x="40" y="60" width="180" height="50" rx="8" fill="#e8f1ff" stroke="#1f4e79" stroke-width="2"/>
  <text x="130" y="90" text-anchor="middle" font-family="Helvetica" font-size="13">mock landmarks</text>
  <rect x="280" y="60" width="180" height="50" rx="8" fill="#e8f1ff" stroke="#1f4e79" stroke-width="2"/>
  <text x="370" y="90" text-anchor="middle" font-family="Helvetica" font-size="13">mapper (2D EE)</text>
  <rect x="520" y="60" width="180" height="50" rx="8" fill="#fff3cd" stroke="#856404" stroke-width="2"/>
  <text x="610" y="90" text-anchor="middle" font-family="Helvetica" font-size="13">safety monitor</text>
  <rect x="280" y="160" width="200" height="50" rx="8" fill="#d4edda" stroke="#155724" stroke-width="2"/>
  <text x="380" y="190" text-anchor="middle" font-family="Helvetica" font-size="13">robot_controller (IK)</text>
  <rect x="280" y="260" width="200" height="50" rx="8" fill="#e8f1ff" stroke="#1f4e79" stroke-width="2"/>
  <text x="380" y="290" text-anchor="middle" font-family="Helvetica" font-size="13">joint_states + RSP</text>
  <rect x="280" y="340" width="200" height="50" rx="8" fill="#cce5ff" stroke="#004085" stroke-width="2"/>
  <text x="380" y="370" text-anchor="middle" font-family="Helvetica" font-size="13">RViz 2 Panda model</text>
  <line x1="220" y1="85" x2="280" y2="85" stroke="#333" stroke-width="2"/>
  <line x1="460" y1="85" x2="520" y2="85" stroke="#333" stroke-width="2"/>
  <line x1="610" y1="110" x2="380" y2="160" stroke="#333" stroke-width="2"/>
  <line x1="380" y1="210" x2="380" y2="260" stroke="#333" stroke-width="2"/>
  <line x1="380" y1="310" x2="380" y2="340" stroke="#333" stroke-width="2"/>
  <text x="40" y="410" font-family="Helvetica" font-size="12" fill="#444">
    Demo type: RViz 2 fake-hardware simulation · No physical Franka · Depth not used for control
  </text>
</svg>
"""
)
print("wrote", OUT)
