#!/usr/bin/env python
# Package version is declared in package.xml and
# src/vision_arm_control/__init__.py (__version__ = "0.3.0").
# Keep those two sources aligned (enforced by tests/test_tree_layout.py).

from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=[
        "vision_arm_control",
        "vision_arm_control.retargeting",
        "vision_arm_control.safety_filters",
        "vision_arm_control.landmarks",
        "vision_arm_control.backends",
    ],
    package_dir={"": "src"},
)

setup(**d)
