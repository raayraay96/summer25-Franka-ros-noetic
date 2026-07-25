# ROS 2 Simulation Documentation

See also: [`scholar/ROS2_SIMULATION.md`](../scholar/ROS2_SIMULATION.md)

## What is demonstrated

A ROS 2 Humble teleoperation pipeline driving a **Franka Emika Panda** model in **RViz 2** using **fake joint states** (no physical robot, not Gazebo unless separately enabled).

**Demo type:** RViz 2 fake-hardware simulation on Purdue Scholar  
**Input:** Smooth mock human-landmark trajectory  
**Depth:** Not used for control (shoulder-relative 2D mapping)  
**Hardware:** No physical Franka connected  

## Architecture

![ROS 2 architecture](ros2-architecture.svg)

## Reproduction

```bash
# 1) Build container (once)
sbatch scholar/build_ros2_container.slurm

# 2) Build workspace + verify topics/joints
sbatch scholar/run_ros2_simulation.slurm

# 3) Record demo video
sbatch scholar/record_ros2_demo.slurm
```

## Safety

- `allow_real_robot: false` in config  
- E-stop topic `/teleop/emergency_stop` cancels motion  
- Workspace clamping and command rate limits active  

## Known limitations

- Geometric IK is a demo approximation, not industrial-grade Franka IK  
- MoveIt 2 planning stack is installed in the container; the verified Scholar recording path uses joint_state fake-hardware for reliability under Xvfb  
- Gazebo panda world is not bundled; `panda_gazebo_sim.launch.py` falls back to the RViz path with clear logging  
