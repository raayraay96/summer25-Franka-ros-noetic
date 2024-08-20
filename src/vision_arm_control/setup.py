#!/usr/bin/env python

from setuptools import setup, find_packages
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'rospy',
        'std_msgs',
        'sensor_msgs',
        'geometry_msgs',
        'cv_bridge',
        'numpy',
        'opencv-python',
        'mediapipe',
    ],
)

setup(**d)