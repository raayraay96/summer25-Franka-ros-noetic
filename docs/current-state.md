# Current Repository State

**Review date:** 2026-07-24  
**Branch:** `ros2-sim-demo`  
**Base:** `portfolio-v2`  
**Pull request:** #2

## Verified before this hardening pass

- ROS 2 Humble Apptainer environment built on Purdue Scholar.
- Both ROS 2 packages completed `colcon build`.
- Mock landmarks, mapping, safety, controller, joint states, and RViz 2 ran on Scholar.
- Recorded topic rates were approximately 20 Hz.
- Joint-state travel was measured and `joints_moved` was true.
- RViz 2 fake-hardware simulation media was recorded and committed.
- No physical Franka was connected.
- Depth was not used for control.

Evidence is stored under `results/ros2/` and `docs/demo/`.

## Hardening completed after the recorded run

### ROS 2

- Added timestamp-based command staleness rejection.
- Added a pose-loss watchdog that asserts trajectory cancellation.
- Fixed cancellation release after E-stop or pose recovery.
- Added a hold-at-current-joints command when cancellation is asserted.
- Added Panda joint-limit rejection before publishing trajectories.
- Added configurable hold time, blend factor, watchdog rate, and stale-command age.
- Added pure tests for staleness, command rate, and joint limits.
- Extended GitHub Actions to execute ROS 2 pure tests and validate safe defaults.

### ROS 1

- Added node-scoped YAML files whose keys match the nodes' private parameters.
- Updated launch files to load configuration inside the consuming node namespace.
- Added command-rate limiting and stale-command rejection.
- Added pose-loss cancellation and E-stop cancellation release.
- Moved blocking MoveIt execution into a cancellable worker thread.
- Added static tests that verify the node/config wiring.

## Current validation boundary

| Item | Status |
|---|---|
| Existing Scholar ROS 2 simulation | Verified |
| Existing demo video | Verified |
| Latest GitHub lint workflow | Verified after hardening |
| Latest GitHub unit workflow | Verified after hardening |
| Post-hardening Scholar runtime rerun | Not yet rerun |
| Gazebo Franka world | Not implemented or claimed |
| Physical Franka validation | Blocked by unavailable hardware |
| Metric depth control | Not implemented or claimed |

## Merge recommendation

Do not describe this project as a Gazebo or physical-hardware demonstration. The public claim should remain:

> ROS 2 Humble RViz 2 fake-hardware simulation using mock human landmarks, shoulder-relative end-effector mapping, safety gating, geometric IK, and simulated joint-state publication.

A final Scholar rerun of `scholar/run_ros2_simulation.slurm` and `scholar/record_ros2_demo.slurm` is recommended after the safety hardening commits. The existing video remains valid evidence of the original ROS 2 pipeline, while the latest CI validates the new pure logic and configuration changes.
