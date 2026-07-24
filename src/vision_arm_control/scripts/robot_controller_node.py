#!/usr/bin/env python3
"""Simulation-safe / dry-run robot controller interface.

Modes:
  dry_run     — log and re-publish targets; never command hardware
  simulation  — attempt MoveIt if available; otherwise dry_run behavior
  real_robot  — MoveIt/hardware path only if explicitly enabled

Default is dry_run. Real robot requires use_robot:=true and allow_real_robot.
"""
from __future__ import annotations

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String


class RobotControllerNode:
    def __init__(self) -> None:
        rospy.init_node("robot_controller_node")
        self.mode = rospy.get_param("~mode", "dry_run")
        self.use_robot = bool(rospy.get_param("~use_robot", False))
        self.allow_real = bool(rospy.get_param("~allow_real_robot", False))
        self.use_moveit = bool(rospy.get_param("~use_moveit", True))

        if self.mode == "real_robot" and not (self.use_robot and self.allow_real):
            rospy.logerr(
                "real_robot mode requested but use_robot/allow_real_robot not both true. "
                "Forcing dry_run."
            )
            self.mode = "dry_run"

        if self.mode == "real_robot":
            rospy.logwarn("=" * 60)
            rospy.logwarn("REAL ROBOT CONTROLLER ACTIVE")
            rospy.logwarn("Ensure physical E-stop is reachable.")
            rospy.logwarn("=" * 60)

        self.move_group = None
        if self.use_moveit and self.mode in ("simulation", "real_robot"):
            self._try_init_moveit()

        in_topic = rospy.get_param("~input_topic", "/teleop/command_pose")
        self.sub = rospy.Subscriber(in_topic, PoseStamped, self._cb, queue_size=1)
        self.status_pub = rospy.Publisher("~status", String, queue_size=1)
        self.echo_pub = rospy.Publisher("~executed_pose", PoseStamped, queue_size=1)
        rospy.loginfo("robot_controller_node mode=%s moveit=%s", self.mode, self.move_group is not None)

    def _try_init_moveit(self) -> None:
        try:
            import moveit_commander

            moveit_commander.roscpp_initialize([])
            group_name = rospy.get_param("~move_group", "panda_arm")
            self.move_group = moveit_commander.MoveGroupCommander(group_name)
            self.move_group.set_planning_time(float(rospy.get_param("~planning_time", 1.0)))
            self.move_group.set_max_velocity_scaling_factor(
                float(rospy.get_param("~max_velocity_scaling", 0.2))
            )
            self.move_group.set_max_acceleration_scaling_factor(
                float(rospy.get_param("~max_acceleration_scaling", 0.2))
            )
            rospy.loginfo("MoveIt MoveGroupCommander initialized: %s", group_name)
        except Exception as exc:  # noqa: BLE001
            rospy.logwarn("MoveIt unavailable (%s). Controller will dry-run targets.", exc)
            self.move_group = None

    def _cb(self, msg: PoseStamped) -> None:
        if self.mode == "dry_run" or self.move_group is None:
            self.status_pub.publish(String(data=f"dry_run:{msg.header.frame_id}"))
            self.echo_pub.publish(msg)
            rospy.logdebug(
                "dry_run target [%.3f, %.3f, %.3f]",
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            )
            return

        try:
            self.move_group.set_pose_target(msg)
            ok = self.move_group.go(wait=True)
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            self.status_pub.publish(String(data="executed" if ok else "plan_or_exec_failed"))
            if ok:
                self.echo_pub.publish(msg)
        except Exception as exc:  # noqa: BLE001
            self.status_pub.publish(String(data=f"error:{exc}"))
            rospy.logerr_throttle(2.0, "controller error: %s", exc)


if __name__ == "__main__":
    try:
        RobotControllerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
