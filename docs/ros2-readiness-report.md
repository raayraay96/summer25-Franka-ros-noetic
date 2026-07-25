# ROS 2 Readiness Report

**Branch:** `ros2-sim-demo`  
**Target:** `portfolio-v2`  
**Simulation label:** RViz 2 fake-hardware simulation  
**Physical hardware:** Not connected

## Acceptance matrix

| Lane | Scope | Status | Evidence |
|---:|---|---|---|
| 1 | Repository and PR state | Verified | PR #2 is open and mergeable |
| 2 | ROS 1 parameter wiring | Implemented and CI-validated | Node-scoped YAML and static wiring tests |
| 3 | ROS 1 topic integration | Implemented but not runtime-rerun | Launch topic names and static tests |
| 4 | Safety enforcement | Implemented and unit-tested | Staleness, rate, timeout, E-stop, joint-limit tests |
| 5 | Controller reliability | Implemented but not runtime-rerun | ROS 1 worker/cancel path and ROS 2 hold path |
| 6 | Depth truthfulness | Verified | Depth explicitly documented as visualization-only |
| 7 | ROS 2 package architecture | Verified on Scholar | Two packages built with `colcon` |
| 8 | ROS 2 launch and parameters | Verified on Scholar | Mock and RViz launch paths executed |
| 9 | Panda simulation | Verified for RViz 2 fake hardware | Joint states moved; not Gazebo |
| 10 | Scholar container | Verified | Apptainer ROS 2 Humble job completed |
| 11 | Video pipeline | Verified | GIF, MP4, and thumbnail committed |
| 12 | Tests | Verified | Scholar tests plus GitHub pure tests |
| 13 | CI | Verified on latest completed head | Lint and test workflows passed |
| 14 | Portfolio documentation | Verified | README and ROS 2 documentation accurately label scope |
| 15 | Integration gate | Partially complete | Post-hardening Scholar rerun remains |

## Verified claims

- ROS 2 Humble runs in an Apptainer environment on Purdue Scholar.
- The repository contains ROS 2 packages for teleoperation logic and bringup.
- Mock landmarks are mapped to Cartesian targets.
- Safety gating publishes accepted command poses.
- The geometric IK controller publishes Panda joint states for RViz 2.
- Joint states changed during the recorded Scholar run.
- Demo media was captured from the actual RViz 2 run.
- No physical Franka was connected.
- The verified demo is not Gazebo.
- Monocular depth is not used for robot control.

## Implemented after the recorded demo

- Stale-command rejection based on message timestamps.
- Pose-loss watchdog cancellation.
- Correct cancellation release after recovery.
- Joint hold publication on cancellation.
- Panda joint-limit checks before trajectory publication.
- ROS 1 private parameter namespace correction.
- ROS 1 cancellable MoveIt worker.
- Expanded ROS 1 and ROS 2 test coverage.
- Expanded CI coverage and artifact-size guards.

## Runtime validation still required

Run the latest branch on Scholar again to ensure the new safety watchdog and hold behavior do not change the expected demonstration path:

```bash
sbatch scholar/run_ros2_simulation.slurm
sbatch scholar/record_ros2_demo.slurm
```

After completion, update the job IDs and evidence files under `results/ros2/` only if the commands actually succeed.

## Out of scope

- Physical Franka execution
- Certified robot safety
- Metric monocular depth
- Full upper-body retargeting
- Gazebo Franka world validation
- Production-grade model-predictive or torque control
