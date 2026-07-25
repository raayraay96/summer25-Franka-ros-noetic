"""RViz 2 fake-hardware simulation of Franka Panda teleoperation.

Label: RViz 2 fake-hardware simulation (NOT Gazebo, NOT physical Franka).
Pipeline: mock landmarks → mapper → safety → joint IK controller →
          robot_state_publisher + joint_states → RViz 2.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    teleop_share = get_package_share_directory("franka_teleop_ros2")
    bringup_share = get_package_share_directory("franka_teleop_bringup")
    params = os.path.join(teleop_share, "config", "teleop.yaml")
    rviz_cfg = os.path.join(bringup_share, "rviz", "panda_teleop.rviz")

    # Prefer moveit_resources_panda_description URDF/xacro
    try:
        panda_share = get_package_share_directory("moveit_resources_panda_description")
        urdf_xacro = os.path.join(panda_share, "urdf", "panda.urdf")
        if not os.path.exists(urdf_xacro):
            # some distros use .urdf.xacro
            cand = os.path.join(panda_share, "urdf", "panda.urdf.xacro")
            urdf_xacro = cand if os.path.exists(cand) else urdf_xacro
    except Exception:
        urdf_xacro = ""

    use_rviz = LaunchConfiguration("use_rviz")
    diagnostics_out = LaunchConfiguration("diagnostics_out")

    robot_description = ParameterValue(
        Command(["cat ", urdf_xacro]) if urdf_xacro.endswith(".urdf")
        else Command(["xacro ", urdf_xacro]),
        value_type=str,
    ) if urdf_xacro else ParameterValue("", value_type=str)

    nodes = [
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument("diagnostics_out", default_value=""),
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
            executable="target_visualizer",
            name="target_visualizer",
            parameters=[params],
            output="screen",
        ),
        Node(
            package="franka_teleop_ros2",
            executable="diagnostics",
            name="diagnostics",
            parameters=[params, {"output_json": diagnostics_out}],
            output="screen",
        ),
    ]

    if urdf_xacro:
        nodes.append(
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                parameters=[{"robot_description": robot_description, "publish_frequency": 50.0}],
                output="screen",
            )
        )
    else:
        nodes.append(
            ExecuteProcess(
                cmd=["echo", "WARNING: moveit_resources_panda_description not found"],
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
            # Single full-window viewport under Xvfb (no side docks in config).
            additional_env={
                "QT_AUTO_SCREEN_SCALE_FACTOR": "0",
                "QT_SCALE_FACTOR": "1",
            },
        )
    )

    # Label banner in logs
    nodes.append(
        TimerAction(
            period=1.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        "bash",
                        "-lc",
                        "echo '=== Demo type: RViz 2 fake-hardware simulation | "
                        "No physical Franka | Depth not used for control ==='",
                    ],
                    output="screen",
                )
            ],
        )
    )
    return LaunchDescription(nodes)
