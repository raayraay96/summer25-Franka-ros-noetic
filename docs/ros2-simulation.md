# ROS 2 Simulation Documentation

See also: [`scholar/ROS2_SIMULATION.md`](../scholar/ROS2_SIMULATION.md)

## What is demonstrated

A ROS 2 Humble teleoperation pipeline driving a **Franka Emika Panda** model in **RViz 2** using **fake joint states** (no physical robot, not Gazebo unless separately enabled).

**Demo type:** RViz 2 fake-hardware simulation on Purdue Scholar  
**Input:** Smooth mock human-landmark trajectory (cyclic, 12 s period)  
**Depth:** Not used for control (shoulder-relative 2D mapping)  
**Hardware:** No physical Franka connected  
**IK:** Geometric approximation (not industrial-grade Franka IK)

## Architecture

![ROS 2 architecture](ros2-architecture.svg)

## Pipeline

```text
mock_landmark_publisher  →  human_to_robot_mapper  →  safety_monitor
        │                         (target_pose)            │
        │                                                  ▼
        │                                           command_pose
        │                                                  │
        ▼                                                  ▼
 mock landmark image                          robot_controller (geometric IK)
 (panel labeled “Mock landmark input”)                 │
                                                       ▼
                                    /joint_states + robot_state_publisher
                                                       │
                                                       ▼
                                    RViz 2 + target_visualizer markers
```

## Reproduction

```bash
# 1) Build container (once)
sbatch scholar/build_ros2_container.slurm

# 2) Build workspace + verify topics/joints
sbatch scholar/run_ros2_simulation.slurm

# 3) Record demo video (v2 media + contact sheet)
sbatch scholar/record_ros2_demo.slurm
```

Post-process only (from an existing raw capture):

```bash
# inside container with ffmpeg
python3 scholar/postprocess_demo_v2.py \
  --raw $RAW/raw_demo.mp4 \
  --out-dir $RAW \
  --demo-dir docs/demo \
  --results-dir results/ros2 \
  --job-id $SLURM_JOB_ID \
  --repo .
```

## Verified Scholar evidence (v2)

| Item | Value |
|------|--------|
| Simulation job | **459319** (pipeline + rates + joint verification) |
| Recording job | **459324** (COMPLETED; RViz X11 capture + v2 media) |
| `colcon build` | 2 packages OK |
| Unit tests (`pytest`) | 14 passed (ROS 2 package) / 51 total with host pure-Python |
| Topic rates | ~20 Hz (`target_pose`, `command_pose`, `joint_states`, `safety_status`) |
| Joint path travel (12 s window) | **4.37 rad** |
| Max single-joint excursion | **0.84 rad** (sim) / **1.50 rad** (record window) |
| Unexpected e-stop / timeout / joint-limit flags | **none** |
| Physical hardware | **false** |
| Depth used for control | **false** |

Evidence files: [`results/ros2/`](../results/ros2/)  
Demo assets: [`docs/demo/ros2-franka-teleoperation-v2.*`](demo/)

## Safety hardening exercised by this demo path

The verified Scholar run uses the current ROS 2 nodes with:

- `allow_real_robot: false`
- Workspace clamp mode
- Command rate limiting
- Pose-timeout monitor
- E-stop → `/teleop/cancel_trajectory` cancellation path
- Approximate Panda joint-limit configuration

The committed v2 recording was produced **after** these controls landed on the demo branch; it is not a pre-hardening artifact.

## Safety

- `allow_real_robot: false` in config  
- E-stop topic `/teleop/emergency_stop` cancels motion  
- Workspace clamping and command rate limits active  

## Known limitations

- Geometric IK is a demo approximation, not industrial-grade Franka IK  
- MoveIt 2 planning stack is installed in the container; the verified Scholar recording path uses joint_state fake-hardware for reliability under Xvfb  
- Gazebo panda world is not bundled; `panda_gazebo_sim.launch.py` falls back to the RViz path with clear logging  
- Mock landmarks are synthetic (not a live camera feed)
