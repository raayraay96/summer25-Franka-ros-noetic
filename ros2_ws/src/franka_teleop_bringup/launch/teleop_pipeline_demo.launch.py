"""Hardened teleop pipeline demo (mock landmarks -> selectable pipeline).

Wires the CI-cross-validated ``vision_arm_control.TeleopPipeline`` into the ROS 2
runtime via ``teleop_pipeline_node``. Combine with ``panda_rviz_sim.launch.py``
(or a MoveIt Panda fake-hardware bringup) to visualize / record.

Scenario arguments (see docs/research/scholar-completion-runbook.md):
  ros2 launch franka_teleop_bringup teleop_pipeline_demo.launch.py \
      strategy:=shoulder_relative safety:=cbf_qp ik_backend:=none \
      diagnostics_jsonl:=/tmp/run_a.jsonl
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    strategy = LaunchConfiguration("strategy")
    safety = LaunchConfiguration("safety")
    ik_backend = LaunchConfiguration("ik_backend")
    obstacles = LaunchConfiguration("obstacles")
    diagnostics_jsonl = LaunchConfiguration("diagnostics_jsonl")
    landmark_topic = LaunchConfiguration("landmark_topic")

    default_obstacles = '[{"id": "demo_sphere", "center": [0.48, 0.0, 0.35], "radius_m": 0.10, "margin_m": 0.05}]'

    return LaunchDescription(
        [
            DeclareLaunchArgument("strategy", default_value="shoulder_relative"),
            DeclareLaunchArgument("safety", default_value="cbf_qp"),
            DeclareLaunchArgument("ik_backend", default_value="none"),
            DeclareLaunchArgument("obstacles", default_value=default_obstacles),
            DeclareLaunchArgument("diagnostics_jsonl", default_value="/tmp/teleop_run.jsonl"),
            DeclareLaunchArgument("landmark_topic", default_value="/perception/landmarks"),
            Node(
                package="franka_teleop_ros2",
                executable="mock_landmark_publisher",
                name="mock_landmark_publisher",
                output="screen",
                parameters=[{"topic": landmark_topic}],
            ),
            Node(
                package="franka_teleop_ros2",
                executable="teleop_pipeline_node",
                name="teleop_pipeline_node",
                output="screen",
                parameters=[
                    {
                        "retargeting_strategy": strategy,
                        "safety_filter": safety,
                        "ik_backend": ik_backend,
                        "obstacles": obstacles,
                        "landmark_topic": landmark_topic,
                        "diagnostics_jsonl": diagnostics_jsonl,
                    }
                ],
            ),
            Node(
                package="franka_teleop_ros2",
                executable="diagnostics",
                name="diagnostics",
                output="screen",
            ),
        ]
    )
