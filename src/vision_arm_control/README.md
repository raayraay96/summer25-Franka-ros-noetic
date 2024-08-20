# Vision Arm Control Project

## Overview

This project implements a vision-based control system for a Franka robotic arm using ROS Noetic. It utilizes computer vision techniques, including human pose estimation and depth estimation, to control the Franka arm in real-time, allowing it to mimic human arm movements.

Key features:
- Human pose estimation using MediaPipe
- Depth estimation using MonoDepth2
- ROS Noetic integration for Franka arm control
- Real-time video processing and visualization
- Inverse kinematics using TRAC-IK

## Current State of the Project

As of August 2024, the project has reached the following milestones:

1. Basic ROS package structure set up
2. Integration of MediaPipe for human pose estimation
3. Integration of MonoDepth2 for depth estimation
4. Initial implementation of inverse kinematics using TRAC-IK
5. Basic mapping of human arm movements to robot arm

The project is functional but requires further refinement and testing.

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

1. Install CUDA:
   ```
   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
   sudo sh cuda_11.8.0_520.61.05_linux.run
   ```

2. Install cuDNN:
   Download cuDNN v8.0 or higher from the [NVIDIA website](https://developer.nvidia.com/cudnn) and follow their installation instructions.

3. Add the following to your `~/.bashrc`:
   ```
   export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
   export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
   ```

### 3. Python Environment

We use Conda for managing the Python environment:

1. Install Miniconda:
   ```
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

2. Create and activate the environment:
   ```
   conda create -n vision_arm_control python=3.8
   conda activate vision_arm_control
   ```

3. Install required Python packages:
   ```
   pip install torch torchvision torchaudio
   pip install tensorflow
   pip install opencv-python
   pip install mediapipe
   pip install rospy
   ```

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

## System Architecture

### Inverse Kinematics

We use the TRAC-IK library for inverse kinematics calculations.

1. **Integration**: 
   The TRAC-IK solver is integrated into our ROS node.

2. **Usage**:
   - Input: desired end-effector position and orientation
   - Output: joint angles required to achieve that pose

3. **Launch File Integration**:
   The TRAC-IK node is included in our main launch file `vision_arm_control.launch`.

### Human-Robot Movement Mapping

The system maps the movements of the human's right arm to the Franka robot arm.

1. **Pose Estimation**: Using MediaPipe
2. **Coordinate Transformation**: 2D to robot's 3D space
3. **End-Effector Targeting**: Robot targets human hand position
4. **Joint Angle Calculation**: Using TRAC-IK solver
5. **Motion Smoothing**: Low-pass filter applied
6. **Robot Control**: Smoothed joint angles sent to robot controller

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

## Current Limitations and Known Issues

1. Depth estimation accuracy needs improvement in certain lighting conditions
2. Occasional jitter in robot movements when mimicking rapid human motions
3. Limited testing with physical Franka robot (mostly simulated environment)

## Next Steps and Future Work

1. Improve depth estimation accuracy:
   - Fine-tune MonoDepth2 model on project-specific dataset
   - Experiment with alternative depth estimation techniques

2. Enhance movement smoothing:
   - Implement advanced filtering techniques (e.g., Kalman filter)
   - Tune motion parameters for more natural movement

3. Expand robot control capabilities:
   - Implement grasping and object manipulation
   - Add collision avoidance algorithms

4. Improve human-robot interaction:
   - Develop a user interface for easy calibration and control
   - Implement gesture recognition for additional commands

5. Extensive testing with physical Franka robot:
   - Conduct thorough tests in various real-world scenarios
   - Fine-tune parameters based on physical robot performance

6. Integration with additional sensors:
   - Explore the use of RGB-D cameras for improved depth perception
   - Investigate the potential of tactile sensors for precise manipulation

7. Documentation and code refactoring:
   - Improve inline code documentation
   - Refactor code for better modularity and reusability

## How to Contribute

1. Fork the repository
2. Create a new branch for your feature: `git checkout -b feature-name`
3. Implement your feature or bug fix
4. Commit your changes: `git commit -m 'Add some feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## Troubleshooting

1. Ensure all dependencies are correctly installed.
2. Verify CUDA and cuDNN compatibility with PyTorch and TensorFlow.
3. Ensure ROS environment is properly sourced.
4. For build issues:
   ```
   cd ~/catkin_ws
   rm -rf build_isolated devel_isolated
   catkin_make_isolated
   ```
5. For inverse kinematics issues:
   - Check if TRAC-IK package is properly installed and built
   - Verify TRAC-IK node is running: `rosnode list`
   - Check robot model and joint limits in URDF file

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MonoDepth2](https://github.com/nianticlabs/monodepth2) for depth estimation
- [MediaPipe](https://github.com/google/mediapipe) for pose estimation
- [ROS](https://www.ros.org/) for robotics framework
- [Franka Emika](https://www.franka.de/) for the Franka robot
- [TRAC-IK](https://traclabs.com/projects/trac-ik/) for inverse kinematics solver

## Contact

For any queries or further information, please open an issue in this GitHub repository.

Below is the images of my Depth Map. I was able to figure this out on my last day working on it but didnt really get to much due to the lighting in the room.

![Screenshot from 2024-08-18 19-59-52](https://github.com/user-attachments/assets/b5c32aea-6236-45f5-b62f-182bf95e7df9)

![Screenshot from 2024-08-18 19-57-36](https://github.com/user-attachments/assets/ddf8d767-7f2e-402e-ac0e-d0cc13ebf851)


