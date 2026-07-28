# Related Work (v1.1)

Independent portfolio review of four papers that informed this repository’s
retargeting and safety design. **None of the paper authors are collaborators.**
University of Wyoming / HUMANS MOVE Program did not supervise this post-program
work. Methods below are labeled **inspired by** unless a full mathematical
reproduction is implemented and tested (none of the four papers are fully
reproduced here).

---

## 1. SEW-Mimic — Closed-Form Geometric Retargeting

**Citation:** Kong et al., “A Closed-Form Geometric Retargeting Solver for Upper Body Humanoid Robot Teleoperation,” arXiv:2602.01632, 2026.  
**Internal label:** SEW-Mimic  
**Project page:** https://sew-mimic.com/

| Field | Summary |
|-------|---------|
| **Problem addressed** | Slow / suboptimal human→robot arm retargeting when optimizing EE pose only; limited robot workspace vs human. |
| **Inputs** | Shoulder, elbow, wrist (SEW) keypoints (source-agnostic); robot kinematics. |
| **Outputs** | Robot joint configurations aligning upper/lower arm orientations; high-rate (~kHz class) retargeting. |
| **Key method** | Reframe retargeting as **orientation alignment**; closed-form geometric solver with optimality claim under their formulation. |
| **Evaluation** | Computation time, accuracy vs baselines, pilot user study, policy-learning data quality, hardware demos. |
| **Transferable idea used here** | SEW keypoint chain; upper/lower arm unit vectors; elbow bend; arm-plane normal; feature-level orientation-aware target. |
| **Not implemented** | Full closed-form 7-DoF SEW-Mimic joint solver; optimality guarantees; bimanual self-collision filter from the paper; humanoid whole-body path. |
| **Differences** | This repo maps SEW features to **Cartesian EE targets** for a Franka Panda teleop stack with optional safety filters; no claim of reproducing SEW-Mimic joint solutions. |
| **Licensing / code reuse** | Paper is arXiv research; no SEW-Mimic source code was copied. Geometry is re-derived at feature level only. |

---

## 2. CBF Safety for Human-to-Humanoid Imitation

**Citation:** Cai et al., “Safe Human-to-Humanoid Motion Imitation Using Control Barrier Functions,” arXiv:2604.11447, 2026.

| Field | Summary |
|-------|---------|
| **Problem addressed** | Safe vision-based human→humanoid imitation with collision avoidance. |
| **Inputs** | Single-camera skeletal keypoints → joint angles for retargeting. |
| **Outputs** | Filtered imitation commands preventing self- and human–robot collisions. |
| **Key method** | CBF layer as a **QP** over imitation commands (safety filter). |
| **Evaluation** | Simulation of real-time collision-aware imitation. |
| **Transferable idea used here** | CBF-QP as a filter on **nominal teleop commands**; log solver status / interventions. |
| **Not implemented** | Full humanoid self-collision model; human–robot interaction CBF; dynamics-level barriers; formal certificates from the paper. |
| **Differences** | This repo uses a **Cartesian kinematic** CBF-QP (workspace halfspaces + spherical obstacles + velocity limits) in pure Python—not a dynamics-level humanoid CBF stack. |
| **Licensing / code reuse** | No paper code copied. Experimental filter is original and intentionally limited. |

---

## 3. AnyTeleop — Modular Vision Teleoperation

**Citation:** Qin et al., “AnyTeleop: A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System,” arXiv:2307.04577 (RSS 2023).  
**Project page:** https://yzqin.github.io/anyteleop/

| Field | Summary |
|-------|---------|
| **Problem addressed** | Vision teleop systems are hard-wired to one robot / environment. |
| **Inputs** | Camera streams; multi-arm / multi-hand configurations. |
| **Outputs** | Dexterous arm–hand teleoperation across simulators and hardware. |
| **Key method** | Unified modular teleoperation architecture spanning arms, hands, realities, cameras. |
| **Evaluation** | Real-robot success rates; simulation imitation-learning data quality. |
| **Transferable idea used here** | Explicit interfaces: `LandmarkSource` / `RetargetingStrategy` / `SafetyFilter` / `RobotBackend`; selectable strategies behind config. |
| **Not implemented** | Multi-robot AnyTeleop stack; dexterous hand retargeting; their vision pipeline; their simulators. |
| **Differences** | Architecture **inspired by** modular separation only; Franka EE teleop portfolio scope. |
| **Licensing / code reuse** | No AnyTeleop code copied. |

---

## 4. Vision-Based Shared-Control Quadruped Arm Teleoperation

**Citation:** da Silva et al., “A Vision-Based Shared-Control Teleoperation Scheme for Controlling the Robotic Arm of a Four-Legged Robot,” arXiv:2508.14994 (LARS 2025).  
**DOI:** 10.1109/LARS69345.2025.11272961

| Field | Summary |
|-------|---------|
| **Problem addressed** | Non-intuitive joystick teleop of a quadruped-mounted arm; collision risk. |
| **Inputs** | External camera + ML wrist pose of operator. |
| **Outputs** | Real-time arm commands with trajectory planning for collision prevention. |
| **Key method** | Direct wrist→arm mapping + planner safety; real-robot validation. |
| **Evaluation** | Real robot teleoperation robustness. |
| **Transferable idea used here** | Vision wrist mapping as teleop; soft shared-control / safety gating ideas; confidence-aware hold. |
| **Not implemented** | Quadruped platform; their ML pose pipeline; their industrial planner stack. |
| **Differences** | Franka Panda dry-run / sim portfolio; modular safety filters instead of their full planner. |
| **Licensing / code reuse** | No paper code copied. |

---

## Cross-cutting integrity statement

| Claim type | This repository |
|------------|-----------------|
| Exact paper reproduction | **No** for all four |
| “Inspired by” wording | **Yes** |
| Physical Franka validation of new methods | **No** |
| Formal CBF certificates | **No** |
| Metric monocular depth | **No** |
