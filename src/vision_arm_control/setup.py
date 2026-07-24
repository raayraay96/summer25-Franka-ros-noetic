#!/usr/bin/env python

from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=["vision_arm_control"],
    package_dir={"": "src"},
)

setup(**d)
