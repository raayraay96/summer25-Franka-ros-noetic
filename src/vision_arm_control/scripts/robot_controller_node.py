#!/usr/bin/env python3
"""Simulation-safe / dry-run robot controller interface.

Modes:
  dry_run     - log and re-publish targets; never command hardware
  simulation  - execute through MoveIt when available
  real_robot  - MoveIt path only after explicit double opt-in

Default is dry_run. Real robot requires use_robot:=true and allow_real_robot.
"""
from __future__ import annotations

import queue
import threading

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String


class RobotControllerNode:
    def __init__(self) -> None:
        rospy.init_node("robot_controller_node")
        self.mode = rospy.get_param("~mode", "dry_run")
        self.use_robot = bool(rospy.get_param("~use_robot", False))
        self.allow_real = bool(rospy.get_param("~allow_real_robot", False))
        self.use_moveit = bool(rospy.get_param("~use_moveit", True))
        self.cancelled = False
        self._shutdown = threading.Event()
        self._targets: queue.Queue = queue.Queue(maxsize=1)
        self._worker = None

        if self.mode == "real_robot" and not (self.use_robot and self.allow_real):
            rospy.logerr(
                "real_robot mode requested but use_robot/allow_real_robot are not both true; "
                "forcing dry_run"
            )
            self.mode = "dry_run"

        if self.mode == "real_robot":
            rospy.logwarn("=" * 60)
            rospy.logwarn("REAL ROBOT CONTROLLER ACTIVE")
            rospy.logwarn("Ensure physical E-stop is reachable and Franka Desk is configured.")
            rospy.logwarn("=" * 60)

        self.move_group = None
        if self.use_moveit and self.mode in ("simulation", "real_robot"):
            self._try_init_moveit()

        in_topic = rospy.get_param("~input_topic", "/teleop/command_pose")
        self.sub = rospy.Subscriber(in_topic, PoseStamped, self._cb, queue_size=1)
        self.cancel_sub = rospy.Subscriber(
            "/teleop/cancel_trajectory", Bool, self._cancel_cb, queue_size=1
        )
        self.status_pub = rospy.Publisher("~status", String, queue_size=1)
        self.echo_pub = rospy.Publisher("~executed_pose", PoseStamped, queue_size=1)

        if self.move_group is not None:
            self._worker = threading.Thread(target=self._execution_worker, daemon=True)
            self._worker.start()

        rospy.on_shutdown(self._on_shutdown)
        rospy.loginfo(
            "robot_controller_node mode=%s moveit=%s",
            self.mode,
            self.move_group is not None,
        )

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
            rospy.logwarn("MoveIt unavailable (%s); controller will dry-run targets", exc)
            self.move_group = None

    def _clear_pending_target(self) -> None:
        try:
            while True:
                self._targets.get_nowait()
                self._targets.task_done()
        except queue.Empty:
            return

    def _cancel_cb(self, msg: Bool) -> None:
        self.cancelled = bool(msg.data)
        if self.cancelled:
            self._clear_pending_target()
            if self.move_group is not None:
                try:
                    self.move_group.stop()
                    self.move_group.clear_pose_targets()
                except Exception as exc:  # noqa: BLE001
                    rospy.logwarn("MoveIt stop during cancel failed: %s", exc)
            self.status_pub.publish(String(data="cancelled"))
            rospy.logwarn("Trajectory cancellation asserted")
        else:
            self.status_pub.publish(String(data="resumed"))
            rospy.loginfo("Trajectory cancellation released")

    def _enqueue_latest(self, msg: PoseStamped) -> None:
        self._clear_pending_target()
        try:
            self._targets.put_nowait(msg)
        except queue.Full:
            self.status_pub.publish(String(data="target_queue_full"))

    def _cb(self, msg: PoseStamped) -> None:
        if self.cancelled:
            self.status_pub.publish(String(data="blocked_cancelled"))
            return

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

        self._enqueue_latest(msg)

    def _execution_worker(self) -> None:
        while not self._shutdown.is_set() and not rospy.is_shutdown():
            try:
                msg = self._targets.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if self.cancelled or self.move_group is None:
                    continue
                self.move_group.set_pose_target(msg)
                ok = self.move_group.go(wait=True)
                self.move_group.stop()
                self.move_group.clear_pose_targets()
                if self.cancelled:
                    self.status_pub.publish(String(data="cancelled_during_execution"))
                else:
                    self.status_pub.publish(
                        String(data="executed" if ok else "plan_or_exec_failed")
                    )
                    if ok:
                        self.echo_pub.publish(msg)
            except Exception as exc:  # noqa: BLE001
                self.status_pub.publish(String(data=f"error:{exc}"))
                rospy.logerr_throttle(2.0, "controller error: %s", exc)
            finally:
                self._targets.task_done()

    def _on_shutdown(self) -> None:
        self._shutdown.set()
        if self.move_group is not None:
            try:
                self.move_group.stop()
                self.move_group.clear_pose_targets()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    try:
        RobotControllerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
