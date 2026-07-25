# ROS 2 Franka Simulation on Purdue Scholar

## Environment discovered

| Item | Value |
|------|-------|
| Host OS | Rocky Linux 9.6 |
| System ROS | **None** |
| Apptainer | 1.4.1 |
| Approach | Isolated Apptainer container (no sudo) |
| ROS 2 | Humble (inside container) |
| Simulation backend (verified demo) | **RViz 2 fake-hardware / joint_state simulation** |
| Gazebo | Optional / not the verified demo path |
| Physical Franka | **Disabled** |

## Container

```text
$RCAC_SCRATCH/franka-teleop-data/containers/ros2_humble_franka.sif
```

Definition: `scholar/ros2_humble.def`  
Build: `sbatch scholar/build_ros2_container.slurm`

Do **not** commit the `.sif` file.

## Packages

```text
ros2_ws/src/franka_teleop_ros2/      # nodes + mapping/safety library
ros2_ws/src/franka_teleop_bringup/   # launch + rviz config
```

## Pipeline

```text
mock_landmark_publisher
        ↓
human_to_robot_mapper   (shoulder-relative EE position; depth NOT used for control)
        ↓
safety_monitor          (workspace, rate, timeout, e-stop)
        ↓
robot_controller        (geometric IK → joint_states / trajectory)
        ↓
robot_state_publisher + RViz 2 (Panda URDF from moveit_resources_panda_description)
```

## Labels (use exactly)

| Mode | Label |
|------|--------|
| Topic-only test | Mock ROS 2 pipeline |
| Verified Scholar demo | RViz 2 fake-hardware simulation |
| If Gazebo packages + world used | Gazebo simulation |
| Never claimed here | Physical Franka |

## Build & run (on a compute node via SLURM)

```bash
export REPO_DIR=/scratch/scholar/edraymon/franka-teleoperation
export RCAC_SCRATCH=/scratch/scholar/edraymon

sbatch scholar/build_ros2_container.slurm
# after SIF exists:
sbatch scholar/run_ros2_simulation.slurm
sbatch scholar/record_ros2_demo.slurm
```

### Interactive (inside container)

```bash
SIF=$RCAC_SCRATCH/franka-teleop-data/containers/ros2_humble_franka.sif
apptainer shell -B $REPO_DIR:/repo $SIF
source /opt/ros/humble/setup.bash
cd /repo/ros2_ws && colcon build
source install/setup.bash
ros2 launch franka_teleop_bringup panda_rviz_sim.launch.py
```

## Evidence location

```text
results/ros2/build-summary.txt
results/ros2/test-summary.txt
results/ros2/environment.json
results/ros2/joint-verification.json
results/ros2/simulation-metrics.json
docs/demo/ros2-franka-teleoperation.gif
docs/demo/ros2-franka-teleoperation.mp4
```

Raw video (not in Git): `$RCAC_SCRATCH/franka-teleop-data/raw_video/`
