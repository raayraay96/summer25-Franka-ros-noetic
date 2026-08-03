# Vision-Guided Franka Panda Teleoperation

[![ROS Noetic](https://img.shields.io/badge/ROS-Noetic-22314E?logo=ros)](https://wiki.ros.org/noetic)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](requirements.txt)
[![CI](https://github.com/raayraay96/summer25-Franka-ros-noetic/actions/workflows/ci.yml/badge.svg)](https://github.com/raayraay96/summer25-Franka-ros-noetic/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A research teleoperation prototype hardened into a reproducible, safety-gated, testable Linux/ROS system with automated validation and optional asynchronous Postgres telemetry.**

<p align="center">
  <img src="docs/media/franka-rviz-dry-run.gif" alt="Mock wrist landmarks driving a Franka Panda end-effector target in RViz 2 fake-hardware simulation" width="820">
</p>

The system maps webcam or mock human wrist landmarks to Franka Panda end-effector targets, rejects unsafe commands before the controller boundary, and defaults to dry-run or simulation behavior. The verified public evidence uses synthetic inputs and RViz 2 fake hardware. It is not physical-robot validation or safety certification.

Developed initially during the **HUMANS MOVE Program at the University of Wyoming**. After the program concluded, Eric Raymond independently refactored the prototype into the portfolio system documented here.

## Engineering Highlights

- Reproducible ROS Noetic environment through Docker, Compose, pinned Python dependencies, and simple `make` commands
- Modular perception, mapping, safety, controller, mock-input, and telemetry boundaries
- Dry-run defaults, workspace gating, pose timeouts, E-stop handling, dead-man hooks, and explicit real-robot opt-in
- Pure-Python unit/property tests plus containerized catkin and launch smoke tests in GitHub Actions
- Trusted-oracle QP cross-validation, paired simulation benchmarks, fault injection, and committed evidence
- Non-blocking telemetry queue for safety decisions, timeouts, E-stop events, latency, and node heartbeat data
- Runbooks, claims ledger, evidence provenance, calibration status, and visible limitations

## Quickstart

### One-command simulation proof

```bash
git clone https://github.com/raayraay96/summer25-Franka-ros-noetic.git
cd summer25-Franka-ros-noetic
make sim
```

`make sim` builds the ROS Noetic container, runs the pure-Python tests, confirms the catkin package, and smoke-tests `simulation.launch` with mock landmarks. It does not connect to a physical Franka.

### Local quality gate without ROS

```bash
make install
make ci
```

Useful commands:

```bash
make help             # list supported workflows
make test             # pure-Python tests
make lint             # Black + Flake8
make qp-smoke         # CI-sized trusted-oracle check
make full-validation  # 6,000 QPs + 30-replicate benchmark
make sim-compose      # same simulation path through Compose
```

Model weights stay outside Git. See [`docs/model-setup.md`](docs/model-setup.md). Common failures are covered in the [`operator runbook`](docs/operator-runbook.md).

## What This Proves

- A fragile research prototype can be converted into an understandable, reproducible system.
- Safety-critical decisions can be isolated behind explicit interfaces and conservative defaults.
- Numerical and simulation claims can be tied to automated tests, artifacts, and reproducible commands.
- Linux/ROS workloads can be containerized and validated through CI without hiding environment limitations.
- Operational metrics can leave the control path through a bounded asynchronous queue and land in PostgreSQL.

## Architecture

```text
webcam / rosbag / mock landmarks
              │
              ▼
      pose_estimator_node ───────────────┐
              │ timestamped JSON         │ optional
              ▼                          ▼
 human_to_robot_mapper_node      depth_estimator_node
 shoulder-relative EE target      relative depth only
              │
              ▼
      safety_monitor_node ── telemetry JSON ──► telemetry_node
 workspace · timeout · E-stop                 bounded async queue
              │ accepted command                        │
              ▼                                         ▼
     robot_controller_node                    Supabase/PostgreSQL
       dry_run by default                       optional, off by default
```

| Component | Responsibility |
|---|---|
| `pose_estimator_node` | MediaPipe landmarks with timestamped JSON output |
| `human_to_robot_mapper_node` | Shoulder-relative mapping to an end-effector position target |
| `safety_monitor_node` | Workspace, timeout, E-stop, dead-man, and mode gating |
| `robot_controller_node` | Dry-run logging or explicit simulation/robot boundary |
| `mock_landmark_publisher` | Deterministic synthetic input for CI and demos |
| `telemetry_node` | Bounded background delivery of safety and heartbeat metrics |

Visual references: [architecture](docs/architecture.svg) · [topic graph](docs/topic-graph.svg) · [frame tree](docs/frame-tree.svg) · [ROS 2 simulation](docs/ros2-architecture.svg).

## Validation

### Safety and numerical evidence

- **6,000 seeded QPs, 0 mismatches** against an OSQP + SciPy trusted oracle, with KKT verification ([audit](docs/research/qp-correctness-audit.md)).
- **1,800 paired simulation runs, 0 unsafe accepted next states, 0 solver failures** across 30 replicates and identical replayed inputs ([results](docs/research/simulation-validation-results.md)).
- **10 fault-injection scenarios, 0 unsafe outcomes** in the recorded simulation evidence.

| Combo | Safety ms | Intervention % mean [95% CI] | Unsafe accepted |
|---|---:|---:|---:|
| shoulder_relative + reject | 0.001 | 1.4 [0.9, 1.8] | 0 |
| sew_orientation + reject | 0.002 | 1.4 [0.9, 1.8] | 0 |
| shoulder_relative + cbf_qp | 8.79 | 86.6 [85.0, 88.2] | 0 |
| sew_orientation + cbf_qp | 8.77 | 86.2 [84.5, 87.9] | 0 |

The high CBF-QP intervention rate is a behavior measurement under a tight barrier model, not a superiority claim.

### Reproduction commands

```bash
make qp-audit
make benchmark-full
python scripts/validate_config.py
```

The pull-request workflow runs an 800-case QP subset and benchmark smoke path. The manually triggered [`full-validation.yml`](.github/workflows/full-validation.yml) runs the full evidence commands and uploads artifacts.

### Additional measured results

| Metric | Result | Environment | Evidence |
|---|---:|---|---|
| Mapping stage latency | **0.0049 ms mean; 0.0080 ms p95; n=5,000** | x86_64, Python 3.13.13, no ROS | [`mapping_benchmark.json`](results/mapping_benchmark.json) |
| Map + filter + workspace latency | **0.0298 ms mean; 0% rejects; n=2,000** | x86_64, Python 3.13.13, no ROS | [`latency_benchmark.json`](results/latency_benchmark.json) |
| ROS 2 target / command / joint / safety rate | **about 20 Hz** | Purdue Scholar, RViz 2 fake hardware | [`topic-rates.json`](results/ros2/topic-rates.json) |
| Physical Franka tracking error | **Not measured** | Physical lab setup required | — |
| End-to-end physical latency | **Not measured** | Camera, controller, and robot required | — |

## Demo Evidence

| Asset | Demonstrates | Status |
|---|---|---|
| [GIF](docs/media/franka-rviz-dry-run.gif) · [MP4](docs/media/franka-rviz-dry-run.mp4) | Mock wrist input to mapped target, safety gate, and RViz Panda | Verified fake-hardware run; no Franka connected |
| [GIF](docs/media/safety-workspace-violation.gif) · [MP4](docs/media/safety-workspace-violation.mp4) | Workspace violation and command rejection | Synthetic input; no ROS hardware |
| [GIF](docs/media/perception-headless-relative-depth.gif) · [MP4](docs/media/perception-headless-relative-depth.mp4) | Landmark JSON and relative-depth visualization | Synthetic schema demonstration |
| [v1.1 media](docs/media/v1.1/) | Retargeting, CBF, and gating trajectories | Verified pure-Python synthetic runs |

Provenance and reproduction commands: [`docs/media/README.md`](docs/media/README.md) · [`docs/media/v1.1/README.md`](docs/media/v1.1/README.md) · [`docs/ros2-simulation.md`](docs/ros2-simulation.md).

## Telemetry and PostgreSQL

Telemetry is optional and disabled by default. The safety node emits local JSON metrics; `telemetry_node` places them on a bounded queue and performs PostgREST inserts on a background thread. A slow or unavailable database cannot block the ROS safety callback.

```bash
roslaunch vision_arm_control simulation.launch enable_telemetry:=true telemetry_dry_run:=true
```

The dry-run command logs records locally. For Supabase setup, schema, environment variables, security constraints, and verification SQL, see [`docs/telemetry.md`](docs/telemetry.md) and [`sql/telemetry_schema.sql`](sql/telemetry_schema.sql).

## Research-Informed v1.1

The default remains `shoulder_relative` + `reject`/`clamp` + `dry_run`. Research-informed modes are versioned, validated, and opt-in.

| Inspired by, not reproduced | Implemented here | Not implemented |
|---|---|---|
| SEW-Mimic orientation features ([paper](https://arxiv.org/abs/2602.01632)) | `sew_orientation` feature-level retargeter | Closed-form 7-DoF SEW joint solver |
| CBF-QP safety filtering ([paper](https://arxiv.org/abs/2604.11447)) | Exact, KKT-verified Cartesian `cbf_qp` | Formal certificates or dynamics CBF |
| AnyTeleop modular interfaces ([paper](https://arxiv.org/abs/2307.04577)) | Selectable retargeter, safety, and backend interfaces | Multi-robot AnyTeleop stack |
| Vision shared control ([paper](https://arxiv.org/abs/2508.14994)) | Confidence gating, hold, and recovery | Their quadruped platform or planner |

Evidence: [`docs/research/`](docs/research/) · [claims ledger](docs/research/claims-ledger.md) · [red-team review](docs/research/red-team-review-v1.1.md) · [case study](docs/case-study-v1.1.md).

## Key Engineering Decisions

1. **Image coordinates were being treated as robot-frame targets.** The refactor added an explicit normalized-to-pixel-to-camera-to-base mapping contract and a shoulder-relative mode.
2. **Relative depth existed but was not valid metric control input.** It remains visualization-only and is labeled as `1 / disparity`, not physical distance.
3. **An incomplete IK chain was not acceptable as a primary controller.** It was replaced by a dry-run boundary and separately labeled simulation paths.
4. **Weights, bags, and build outputs damaged reproducibility.** They were removed from the public history and blocked by repository audits.
5. **Machine-specific paths prevented reuse.** They were replaced with ROS parameters, container paths, and `FRANKA_MODEL_DIR`.
6. **Telemetry could not be allowed to affect control timing.** The implementation uses local publication plus a bounded asynchronous writer with drop/failure accounting.

The preserved initial audit is in [`docs/repository-audit.md`](docs/repository-audit.md).

## Limitations

- All public safety and benchmark claims are simulation or pure-Python evidence, not physical Franka validation.
- Software checks are not safety certification and do not replace Franka Desk, a physical E-stop, dead-man control, or lab procedures.
- Relative monocular depth is not metric and is not used as validated robot range data.
- Camera intrinsics and camera-to-base transforms remain examples until measured calibration is recorded.
- ROS Noetic and Ubuntu 20.04 are end-of-life technical debt. The ROS 2 path is a simulation proof, not full hardware parity.
- Gazebo + MoveIt collision-aware execution has not been validated on every supported host.
- The direct Supabase writer is an edge-host portfolio MVP. Production ingestion should use a least-privilege authenticated service boundary.

## What I Learned

- Harden the execution path before expanding the algorithm surface.
- Make unsafe or unverified behavior impossible by default, not merely discouraged in documentation.
- Separate measured evidence from planned work and keep the exact reproduction commands beside each claim.
- Treat containers, CI, telemetry, runbooks, and failure handling as part of the system rather than portfolio decoration.

## Roadmap

| State | Work |
|---|---|
| Implemented | Modular ROS1 nodes, tests, pinned environment, Docker, CI, safety gates, ROS 2 RViz evidence, v1.1 evaluation, telemetry MVP |
| In progress | Measured camera calibration, refreshed RViz recording with selectable v1.1 strategies, MoveIt planning integration |
| Planned | RGB-D metric mapping, verified SEW IK adapter, Gazebo collision demo, physical Franka re-validation |

## Author Contribution

| Work | Contribution |
|---|---|
| HUMANS MOVE research prototype, Summer 2024 | Eric Raymond: perception experiments, ROS integration, teleoperation prototype, and research artifacts completed during the program |
| Independent portfolio engineering, 2026 | Eric Raymond: modular architecture, safety/mapping refactor, tests, benchmarks, containers, CI, runbooks, telemetry, simulation evidence, and documentation |
| Third-party systems | MediaPipe, MonoDepth2, ROS/MoveIt, Franka ecosystem packages, OSQP, SciPy, and Supabase/PostgreSQL remain their maintainers' work |

## Citation and License

Initial prototype work was completed in the **HUMANS MOVE Program, University of Wyoming**. Post-program portfolio engineering is independent. See [`CITATION.cff`](CITATION.cff), [`docs/references.bib`](docs/references.bib), and the [MIT License](LICENSE).
