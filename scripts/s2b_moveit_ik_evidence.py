#!/usr/bin/env python3
"""S2-B evidence: call MoveIt /compute_ik and record limit-checked JointStates.

Does not fake joints. Requires a live move_group advertising /compute_ik
(e.g. panda_moveit_ik.launch.py use_rviz:=false). dry_run only — no hardware.

Usage (inside ROS 2 Humble + sourced workspace):
  python3 scripts/s2b_moveit_ik_evidence.py \\
    --output results/v1.1-hardening/ros2/s2b_joint_evidence.json \\
    --jsonl results/v1.1-hardening/ros2/scenario_s2b_moveit_ik.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import PositionIKRequest
from moveit_msgs.srv import GetPositionIK
from rclpy.node import Node
from sensor_msgs.msg import JointState

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vision_arm_control" / "src"))
from vision_arm_control.workspace_limits import PANDA_JOINT_LIMITS  # noqa: E402

PANDA_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
PANDA_READY = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

# Reachable Cartesian targets in panda_link0 (m) with pointing-down orientation.
TARGETS = [
    (0.40, 0.00, 0.45),
    (0.45, 0.10, 0.40),
    (0.45, -0.10, 0.40),
    (0.50, 0.00, 0.35),
    (0.35, 0.15, 0.50),
    (0.42, -0.12, 0.48),
    (0.48, 0.05, 0.42),
    (0.38, 0.00, 0.55),
]


class IkEvidenceNode(Node):
    def __init__(self, service: str, timeout: float) -> None:
        super().__init__("s2b_moveit_ik_evidence")
        self.cli = self.create_client(GetPositionIK, service)
        self.pub = self.create_publisher(JointState, "/teleop/ik_joint_states", 10)
        self.timeout = timeout
        self.seed = list(PANDA_READY)

    def wait_service(self, max_wait: float) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < max_wait:
            if self.cli.wait_for_service(timeout_sec=1.0):
                return True
        return False

    def compute(self, xyz) -> dict:
        req = GetPositionIK.Request()
        req.ik_request = PositionIKRequest()
        req.ik_request.group_name = "panda_arm"
        req.ik_request.ik_link_name = "panda_link8"
        ps = PoseStamped()
        ps.header.frame_id = "panda_link0"
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(xyz[0])
        ps.pose.position.y = float(xyz[1])
        ps.pose.position.z = float(xyz[2])
        # Pointing-down EE (common Panda demo convention)
        ps.pose.orientation.x = 1.0
        ps.pose.orientation.y = 0.0
        ps.pose.orientation.z = 0.0
        ps.pose.orientation.w = 0.0
        req.ik_request.pose_stamped = ps
        req.ik_request.avoid_collisions = False
        req.ik_request.timeout.sec = 0
        req.ik_request.timeout.nanosec = int(self.timeout * 1e9)
        req.ik_request.robot_state.joint_state.name = list(PANDA_JOINTS)
        req.ik_request.robot_state.joint_state.position = list(self.seed)

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.timeout + 0.5)
        rec = {
            "target_xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
            "ok": False,
            "error_code": None,
            "joints": None,
            "within_limits": False,
            "frame_id": None,
        }
        if not future.done():
            rec["error"] = "timeout"
            return rec
        try:
            resp = future.result()
        except Exception as exc:  # noqa: BLE001
            rec["error"] = str(exc)
            return rec
        rec["error_code"] = int(resp.error_code.val)
        if resp.error_code.val != 1:
            return rec
        name_to_pos = {
            n: float(v)
            for n, v in zip(resp.solution.joint_state.name, resp.solution.joint_state.position)
        }
        if not all(j in name_to_pos for j in PANDA_JOINTS):
            rec["error"] = "missing_joint_names"
            return rec
        joints = [name_to_pos[j] for j in PANDA_JOINTS]
        within = bool(PANDA_JOINT_LIMITS.contains(joints))
        rec["joints"] = joints
        rec["within_limits"] = within
        if not within:
            rec["error"] = "joint_limits_rejected"
            return rec
        self.seed = joints
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.header.frame_id = "verified_moveit"
        js.name = list(PANDA_JOINTS)
        js.position = joints
        self.pub.publish(js)
        # give subscribers a tick
        rclpy.spin_once(self, timeout_sec=0.05)
        rec["ok"] = True
        rec["frame_id"] = "verified_moveit"
        return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--service", default="/compute_ik")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--jsonl", type=Path, default=None)
    ap.add_argument("--timeout", type=float, default=0.15)
    ap.add_argument("--wait-service", type=float, default=60.0)
    ap.add_argument("--repeat", type=int, default=2, help="cycles over TARGETS")
    args = ap.parse_args(argv)

    rclpy.init()
    node = IkEvidenceNode(args.service, args.timeout)
    evidence = {
        "gate": "S2-B",
        "service": args.service,
        "ik_backend": "moveit",
        "verified_topic": "/teleop/ik_joint_states",
        "frame_id_expected": "verified_moveit",
        "samples": [],
        "attempts": [],
        "n_samples": 0,
        "n_ok": 0,
        "n_fail": 0,
        "status": "no_samples",
        "physical_hardware": False,
        "dry_run": True,
    }

    if not node.wait_service(args.wait_service):
        evidence["status"] = "blocked_compute_ik_missing"
        evidence["blocker"] = f"{args.service} not available within {args.wait_service}s"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2) + "\n")
        print(json.dumps(evidence, indent=2))
        node.destroy_node()
        rclpy.shutdown()
        return 2

    jsonl_f = None
    if args.jsonl:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl_f = args.jsonl.open("w")

    for _cycle in range(max(1, args.repeat)):
        for xyz in TARGETS:
            rec = node.compute(xyz)
            evidence["attempts"].append(rec)
            if jsonl_f is not None:
                jsonl_f.write(json.dumps(rec, sort_keys=True) + "\n")
            if rec.get("ok"):
                evidence["n_ok"] += 1
                sample = {
                    "target_xyz": rec["target_xyz"],
                    "frame_id": rec["frame_id"],
                    "name": list(PANDA_JOINTS),
                    "position": rec["joints"],
                    "within_limits": True,
                }
                evidence["samples"].append(sample)
            else:
                evidence["n_fail"] += 1
            time.sleep(0.05)

    if jsonl_f is not None:
        jsonl_f.close()

    evidence["n_samples"] = len(evidence["samples"])
    if evidence["n_samples"] > 0 and all(
        s.get("frame_id") == "verified_moveit" and s.get("within_limits") for s in evidence["samples"]
    ):
        evidence["status"] = "verified_moveit_jointstate_evidence"
    elif evidence["n_samples"] > 0:
        evidence["status"] = "partial_samples"
    else:
        evidence["status"] = "no_verified_joints_recorded"
        # keep a few error codes for debugging
        codes = [a.get("error_code") for a in evidence["attempts"] if a.get("error_code") is not None]
        evidence["error_code_histogram"] = {str(c): codes.count(c) for c in sorted(set(codes))}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({k: evidence[k] for k in ("status", "n_samples", "n_ok", "n_fail")}, indent=2))
    node.destroy_node()
    rclpy.shutdown()
    return 0 if evidence["status"] == "verified_moveit_jointstate_evidence" else 1


if __name__ == "__main__":
    raise SystemExit(main())
