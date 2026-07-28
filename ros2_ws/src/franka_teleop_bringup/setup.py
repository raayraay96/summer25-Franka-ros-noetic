from setuptools import setup
from glob import glob
import os

package_name = "franka_teleop_bringup"

setup(
    name=package_name,
    version="0.2.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.py")),
        (f"share/{package_name}/rviz", glob("rviz/*")),
        (f"share/{package_name}/config", glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Eric Raymond",
    maintainer_email="135032187+raayraay96@users.noreply.github.com",
    description="Launch files for Franka teleop ROS 2 simulation",
    license="MIT",
    entry_points={},
)
