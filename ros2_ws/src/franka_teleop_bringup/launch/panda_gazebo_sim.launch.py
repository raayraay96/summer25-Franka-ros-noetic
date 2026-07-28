"""Placeholder Gazebo launch — only runs if Gazebo Classic/Fortress packages exist.

This file will fail clearly if Gazebo packages are missing. Prefer
panda_rviz_sim.launch.py for the verified Scholar demo.
"""
from launch import LaunchDescription
from launch.actions import LogInfo, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
import os


def generate_launch_description():
    # Always include the RViz teleop pipeline; Gazebo is optional extra.
    bringup = get_package_share_directory("franka_teleop_bringup")
    rviz_launch = os.path.join(bringup, "launch", "panda_rviz_sim.launch.py")

    actions = [
        LogInfo(
            msg=(
                "Gazebo simulation is optional. "
                "Verified Scholar demo uses: RViz 2 fake-hardware simulation. "
                "Do not label this launch as Gazebo unless Gazebo packages are installed "
                "and a panda gazebo world is included."
            )
        ),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(rviz_launch),
            launch_arguments={"use_rviz": LaunchConfiguration("use_rviz")}.items(),
        ),
    ]

    # Try to detect gazebo ros packages
    try:
        get_package_share_directory("gazebo_ros")
        actions.append(
            LogInfo(
                msg=(
                    "gazebo_ros is installed, but a dedicated Franka Gazebo world is not "
                    "bundled in this repository. Use RViz fake-hardware path for the demo."
                )
            )
        )
    except PackageNotFoundError:
        actions.append(LogInfo(msg="gazebo_ros not found — using RViz 2 fake-hardware only."))

    return LaunchDescription(actions)
