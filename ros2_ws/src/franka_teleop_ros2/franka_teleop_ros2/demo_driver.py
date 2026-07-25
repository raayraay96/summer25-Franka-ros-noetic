#!/usr/bin/env python3
"""Optional scripted e-stop pulse mid-demo for safety verification."""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


class DemoDriver(Node):
    def __init__(self) -> None:
        super().__init__("demo_driver")
        self.declare_parameter("estop_at_sec", -1.0)  # negative = disabled
        self.declare_parameter("estop_hold_sec", 1.5)
        self.pub = self.create_publisher(Bool, "/teleop/emergency_stop", 10)
        self.t0 = self.get_clock().now()
        self.fired = False
        self.cleared = False
        self.create_timer(0.1, self._tick)

    def _tick(self) -> None:
        at = float(self.get_parameter("estop_at_sec").value)
        if at < 0:
            return
        t = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        hold = float(self.get_parameter("estop_hold_sec").value)
        if not self.fired and t >= at:
            m = Bool()
            m.data = True
            self.pub.publish(m)
            self.fired = True
            self.get_logger().warn("demo_driver: e-stop ON")
        if self.fired and not self.cleared and t >= at + hold:
            m = Bool()
            m.data = False
            self.pub.publish(m)
            self.cleared = True
            self.get_logger().warn("demo_driver: e-stop OFF")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DemoDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
