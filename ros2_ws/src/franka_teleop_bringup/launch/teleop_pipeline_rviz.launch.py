"""End-to-end hardened teleop pipeline + RViz 2 fake-hardware visualization.

Combines:
  mock landmarks → teleop_pipeline_node (CI-cross-validated TeleopPipeline)
  → joint_states (optional IK) → robot_state_publisher → RViz 2

Label: ROS 2 SIMULATION | MOCK LANDMARKS | PANDA FAKE HARDWARE | NO PHYSICAL ROBOT
"""
from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue  # also used for robot_description


def generate_launch_description():
    bringup_share = get_package_share_directory("franka_teleop_bringup")
    rviz_cfg = os.path.join(bringup_share, "rviz", "panda_teleop.rviz")

    try:
        panda_share = get_package_share_directory("moveit_resources_panda_description")
        urdf_xacro = os.path.join(panda_share, "urdf", "panda.urdf")
        if not os.path.exists(urdf_xacro):
            cand = os.path.join(panda_share, "urdf", "panda.urdf.xacro")
            urdf_xacro = cand if os.path.exists(cand) else urdf_xacro
    except Exception:
        urdf_xacro = ""

    strategy = LaunchConfiguration("strategy")
    safety = LaunchConfiguration("safety")
    ik_backend = LaunchConfiguration("ik_backend")
    obstacles = LaunchConfiguration("obstacles")
    diagnostics_jsonl = LaunchConfiguration("diagnostics_jsonl")
    landmark_topic = LaunchConfiguration("landmark_topic")
    use_rviz = LaunchConfiguration("use_rviz")
    publish_joint_states = LaunchConfiguration("publish_joint_states")

    default_obstacles = (
        '[{"id": "demo_sphere", "center": [0.48, 0.0, 0.35], '
        '"radius_m": 0.10, "margin_m": 0.05}]'
    )

    robot_description = (
        ParameterValue(
            Command(["cat ", urdf_xacro])
            if urdf_xacro.endswith(".urdf")
            else Command(["xacro ", urdf_xacro]),
            value_type=str,
        )
        if urdf_xacro
        else ParameterValue("", value_type=str)
    )

    nodes = [
        DeclareLaunchArgument("strategy", default_value="shoulder_relative"),
        DeclareLaunchArgument("safety", default_value="cbf_qp"),
        DeclareLaunchArgument("ik_backend", default_value="geometric_unverified"),
        DeclareLaunchArgument("obstacles", default_value=default_obstacles),
        DeclareLaunchArgument("diagnostics_jsonl", default_value="/tmp/teleop_run.jsonl"),
        DeclareLaunchArgument("landmark_topic", default_value="/perception/landmarks"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument("publish_joint_states", default_value="true"),
        Node(
            package="franka_teleop_ros2",
            executable="mock_landmark_publisher",
            name="mock_landmark_publisher",
            output="screen",
            parameters=[{"topic": ParameterValue(landmark_topic, value_type=str)}],
        ),
        Node(
            package="franka_teleop_ros2",
            executable="teleop_pipeline_node",
            name="teleop_pipeline_node",
            output="screen",
            parameters=[
                {
                    "retargeting_strategy": ParameterValue(strategy, value_type=str),
                    "safety_filter": ParameterValue(safety, value_type=str),
                    "ik_backend": ParameterValue(ik_backend, value_type=str),
                    "obstacles": ParameterValue(obstacles, value_type=str),
                    "landmark_topic": ParameterValue(landmark_topic, value_type=str),
                    "diagnostics_jsonl": ParameterValue(diagnostics_jsonl, value_type=str),
                    "publish_joint_states": ParameterValue(publish_joint_states, value_type=bool),
                }
            ],
        ),
        Node(
            package="franka_teleop_ros2",
            executable="diagnostics",
            name="diagnostics",
            output="screen",
        ),
        Node(
            package="franka_teleop_ros2",
            executable="target_visualizer",
            name="target_visualizer",
            output="screen",
        ),
    ]

    if urdf_xacro:
        nodes.append(
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                parameters=[
                    {"robot_description": robot_description, "publish_frequency": 50.0}
                ],
                output="screen",
            )
        )

    nodes.append(
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_cfg] if os.path.exists(rviz_cfg) else [],
            condition=IfCondition(use_rviz),
            output="screen",
            additional_env={
                "QT_AUTO_SCREEN_SCALE_FACTOR": "0",
                "QT_SCALE_FACTOR": "1",
            },
        )
    )

    nodes.append(
        TimerAction(
            period=1.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        "bash",
                        "-lc",
                        "echo '=== ROS 2 SIMULATION | MOCK LANDMARKS | "
                        "PANDA FAKE HARDWARE | NO PHYSICAL ROBOT ==='",
                    ],
                    output="screen",
                )
            ],
        )
    )
    return LaunchDescription(nodes)
