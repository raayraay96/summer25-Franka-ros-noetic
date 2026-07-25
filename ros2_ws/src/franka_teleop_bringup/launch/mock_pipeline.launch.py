"""Mock ROS 2 pipeline only (no robot model). For topic-flow tests."""
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    share = get_package_share_directory("franka_teleop_ros2")
    params = os.path.join(share, "config", "teleop.yaml")
    return LaunchDescription(
        [
            Node(
                package="franka_teleop_ros2",
                executable="mock_landmark_publisher",
                name="mock_landmark_publisher",
                parameters=[params],
                output="screen",
            ),
            Node(
                package="franka_teleop_ros2",
                executable="human_to_robot_mapper",
                name="human_to_robot_mapper",
                parameters=[params],
                output="screen",
            ),
            Node(
                package="franka_teleop_ros2",
                executable="safety_monitor",
                name="safety_monitor",
                parameters=[params],
                output="screen",
            ),
            Node(
                package="franka_teleop_ros2",
                executable="robot_controller",
                name="robot_controller",
                parameters=[params],
                output="screen",
            ),
            Node(
                package="franka_teleop_ros2",
                executable="diagnostics",
                name="diagnostics",
                parameters=[params],
                output="screen",
            ),
        ]
    )
