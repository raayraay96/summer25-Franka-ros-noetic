# Vision Arm Control Project

## Overview

This project implements a vision-based control system for a Franka robotic arm using ROS Noetic. It utilizes computer vision techniques, including human pose estimation and depth estimation, to control the Franka arm in real-time, allowing it to mimic human arm movements.

Key features:
- Human pose estimation using MediaPipe
- Depth estimation using MonoDepth2
- ROS Noetic integration for Franka arm control
- Real-time video processing and visualization

## Prerequisites

- Ubuntu 20.04
- ROS Noetic
- Python 3.8+
- CUDA 11.8 or higher
- cuDNN 8.0 or higher
- OpenCV 4.2 or higher
- PyTorch 1.8 or higher
- TensorFlow 2.4 or higher (for MediaPipe)

## Installation

### 1. ROS Noetic

Follow the [official ROS Noetic installation guide](http://wiki.ros.org/noetic/Installation/Ubuntu) for Ubuntu 20.04.

### 2. CUDA and cuDNN

[Instructions for CUDA and cuDNN installation remain the same]

### 3. Python Environment

[Instructions for Python environment setup remain the same]

### 4. Project Setup

1. Clone the repository:
   ```
   mkdir -p ~/catkin_ws/src
   cd ~/catkin_ws/src
   git clone https://github.com/raayraay96/summer25-Franka-ros-noetic.git
   ```

2. Install additional dependencies:
   ```
   sudo apt-get install ros-noetic-franka-ros
   sudo apt-get install ros-noetic-libfranka
   ```

3. Build the workspace using catkin_make_isolated:
   ```
   cd ~/catkin_ws
   catkin_make_isolated
   ```

4. Source the workspace:
   ```
   source devel_isolated/setup.bash
   ```

## Usage

1. Launch the main node:
   ```
   roslaunch vision_arm_control vision_arm_control.launch
   ```

2. In a separate terminal, run the camera node:
   ```
   rosrun vision_arm_control ros_camera_view.py
   ```

## Project Structure

- `src/vision_arm_control/`: Main project package
  - `scripts/`: Python scripts for vision processing and arm control
  - `launch/`: ROS launch files
  - `config/`: Configuration files
- `src/monodepth2/`: MonoDepth2 submodule for depth estimation
- `src/trac_ik/`: TRAC-IK submodule for inverse kinematics

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MonoDepth2](https://github.com/nianticlabs/monodepth2) for depth estimation
- [MediaPipe](https://github.com/google/mediapipe) for pose estimation
- [ROS](https://www.ros.org/) for robotics framework
- [Franka Emika](https://www.franka.de/) for the Franka robot

## Troubleshooting

If you encounter any issues during setup or execution, please check the following:

1. Ensure all dependencies are correctly installed.
2. Verify that your CUDA and cuDNN versions are compatible with your PyTorch and TensorFlow installations.
3. Make sure your ROS environment is properly sourced.
4. If you encounter build issues, try cleaning your workspace and rebuilding:
   ```
   cd ~/catkin_ws
   rm -rf build_isolated devel_isolated
   catkin_make_isolated
   ```

For additional help, please open an issue in the GitHub repository.

Below is the images of my Depth Map. I was able to figure this out on my last day working on it but didnt really get to much due to the lighting in the room.

![Screenshot from 2024-08-18 19-59-52](https://github.com/user-attachments/assets/b5c32aea-6236-45f5-b62f-182bf95e7df9)

![Screenshot from 2024-08-18 19-57-36](https://github.com/user-attachments/assets/ddf8d767-7f2e-402e-ac0e-d0cc13ebf851)


