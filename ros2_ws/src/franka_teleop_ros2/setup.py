from setuptools import find_packages, setup
from glob import glob

package_name = "franka_teleop_ros2"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/rviz", glob("rviz/*")),
    ],
    install_requires=["setuptools", "numpy", "PyYAML"],
    zip_safe=True,
    maintainer="Eric Raymond",
    maintainer_email="135032187+raayraay96@users.noreply.github.com",
    description="ROS 2 Franka teleoperation nodes",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mock_landmark_publisher = franka_teleop_ros2.mock_landmark_publisher:main",
            "human_to_robot_mapper = franka_teleop_ros2.human_to_robot_mapper:main",
            "safety_monitor = franka_teleop_ros2.safety_monitor:main",
            "robot_controller = franka_teleop_ros2.robot_controller:main",
            "target_visualizer = franka_teleop_ros2.target_visualizer:main",
            "diagnostics = franka_teleop_ros2.diagnostics:main",
            "demo_driver = franka_teleop_ros2.demo_driver:main",
        ],
    },
)
