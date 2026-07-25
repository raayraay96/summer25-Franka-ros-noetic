#!/usr/bin/env python3
"""Lightweight diagnostics: topic rates and joint-state movement detection."""
from __future__ import annotations

import json
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String


class Diagnostics(Node):
    def __init__(self) -> None:
        super().__init__("diagnostics")
        self.declare_parameter("output_json", "")
        self.declare_parameter("window_sec", 8.0)
        self.window = float(self.get_parameter("window_sec").value)
        self.counts = {
            "target_pose": 0,
            "command_pose": 0,
            "joint_states": 0,
            "safety_status": 0,
        }
        self.joint_samples = []
        self.t0 = time.time()
        self.create_subscription(PoseStamped, "/teleop/target_pose", self._t, 10)
        self.create_subscription(PoseStamped, "/teleop/command_pose", self._c, 10)
        self.create_subscription(JointState, "/joint_states", self._j, 10)
        self.create_subscription(String, "/teleop/safety_status", self._s, 10)
        self.create_timer(1.0, self._tick)
        self.get_logger().info("diagnostics listening")

    def _t(self, _):
        self.counts["target_pose"] += 1

    def _c(self, _):
        self.counts["command_pose"] += 1

    def _s(self, _):
        self.counts["safety_status"] += 1

    def _j(self, msg: JointState):
        self.counts["joint_states"] += 1
        if msg.position:
            self.joint_samples.append(list(msg.position[:7]))

    def _tick(self) -> None:
        elapsed = time.time() - self.t0
        if elapsed < self.window:
            return
        rates = {k: v / elapsed for k, v in self.counts.items()}
        moved = False
        travel = 0.0
        if len(self.joint_samples) >= 2:
            a = self.joint_samples[0]
            b = self.joint_samples[-1]
            travel = sum(abs(bi - ai) for ai, bi in zip(a, b))
            moved = travel > 0.05
        report = {
            "elapsed_sec": elapsed,
            "topic_rates_hz": rates,
            "joint_state_travel_rad": travel,
            "joints_moved": moved,
            "n_joint_samples": len(self.joint_samples),
            "simulation_label": "RViz 2 fake-hardware / joint_state simulation",
            "physical_hardware": False,
        }
        self.get_logger().info(json.dumps(report))
        out = str(self.get_parameter("output_json").value)
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(json.dumps(report, indent=2) + "\n")
            self.get_logger().info(f"wrote {out}")
        # keep running for longer demos; reset window optional
        # do not reset so report is cumulative after first window


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Diagnostics()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
