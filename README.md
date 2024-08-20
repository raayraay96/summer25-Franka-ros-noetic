# summer25-Franka-ros-noetic
Human-Robot Mimicry: ROS1-based project enabling a Panda Franka Emika robot to mimic human arm movements using computer vision and monocular depth estimation. Features real-time pose tracking, 2D-to-3D mapping, and advanced AI techniques for intuitive robotic control. Developed during HUMANS MOVE Program at University of Wyoming.


# Vision Arm Control

## Overview

This project implements a vision-based control system for a robotic arm using ROS (Robot Operating System) and various computer vision techniques. The system uses human pose estimation and depth estimation to control a Panda robotic arm, allowing it to mimic human arm movements.

## Features

- Human pose estimation using MediaPipe
- Depth estimation using MonoDepth2
- ROS integration for robotic arm control
- Real-time video processing and visualization

## Prerequisites

- ROS Noetic
- Python 3.8+
- OpenCV
- PyTorch
- MediaPipe
- MonoDepth2

## Installation

1. Clone this repository into your catkin workspace:
   ```
   cd ~/catkin_ws/src
   git clone https://github.com/yourusername/vision_arm_control.git
   ```

2. Install the required Python packages:
   ```
   pip install opencv-python torch torchvision mediapipe
   ```

3. Clone and install MonoDepth2:
   ```
   git clone https://github.com/nianticlabs/monodepth2.git
   cd monodepth2
   pip install -r requirements.txt
   ```

4. Build your catkin workspace:
   ```
   cd ~/catkin_ws
   catkin_make
   ```

## Usage

1. Source your ROS workspace:
   ```
   source ~/catkin_ws/devel/setup.bash
   ```

2. Launch the main node:
   ```
   rosrun vision_arm_control main.py
   ```

3. In a separate terminal, run the camera node:
   ```
   rosrun vision_arm_control ros_camera_view.py
   ```

## Configuration

You can adjust various parameters in the `main.py` file, such as:
- Camera topic
- Pose estimation confidence thresholds
- Depth estimation model weights

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MonoDepth2](https://github.com/nianticlabs/monodepth2) for depth estimation
- [MediaPipe](https://github.com/google/mediapipe) for pose estimation
- [ROS](https://www.ros.org/) for robotics framework
