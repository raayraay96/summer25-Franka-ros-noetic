# Repository Audit — summer25-Franka-ros-noetic

> **Post-merge status (2026-07-25)**  
> This audit was written against pre-`portfolio-v2` `main` (`94933b3`).  
> Branch `portfolio-v2` was merged into `main` and implements the recommended order in §14 (modular nodes, safety defaults, tests, CI, model hygiene, Scholar templates, honest limitations).  
> Follow-up cleanup on `main` after the merge:  
> - Removed machine-specific `src/CMakeLists.txt` symlink  
> - Removed five empty gitlink entries (`franka_ros`, `franka_panda_description`, `monodepth2`, `trac_ik`, `vision_opencv`) that had no `.gitmodules`  
> - Working tree under `src/` now contains only the maintained `vision_arm_control` package  
> Large blobs remain in **Git history** only; see [`docs/history-cleanup.md`](history-cleanup.md) for authorized filter-repo procedure. Do not invent unmeasured hardware results.

---

**Audit date:** 2026-07-24  
**Branch audited:** `main` @ `94933b3` (pre-portfolio-v2)  
**Auditor method:** Full tree inspection, source review, Git object size analysis, path search.  
**Truthfulness rule:** Findings are labeled `Verified`, `Likely`, `Unverified`, or `Blocked by unavailable hardware`.

---

## 1. Current repository tree (tracked content summary)

```text
.
├── README.md                          # short + package-level README exists under package
├── .gitignore
├── Building, cd, make, Processing     # empty accidental root files (Verified)
├── .qdrant-initialized                # empty unrelated marker (Verified)
├── .catkin_tools/                     # local catkin tooling metadata (Verified)
└── src/
    ├── CMakeLists.txt -> /opt/ros/noetic/... (symlink; machine-specific) (Verified)
    ├── franka_ros                     # git submodule (empty checkout; gitlink only)
    ├── franka_panda_description       # git submodule (empty checkout)
    ├── monodepth2                     # git submodule (empty checkout)
    ├── trac_ik                        # git submodule (empty checkout)
    ├── vision_opencv                  # git submodule (empty checkout)
    └── vision_arm_control/
        ├── package.xml                # placeholder maintainer/license
        ├── CMakeLists.txt             # hardcoded Conda PYTHON_EXECUTABLE
        ├── setup.py
        ├── LICENSE                    # MIT text present in package
        ├── README.md                  # detailed but overclaims
        ├── scripts/
        │   ├── main.py                # monolithic perception+control
        │   ├── monodepth2_node.py     # alternate depth node
        │   ├── ros_camera_view.py     # OpenCV webcam publisher
        │   └── print_python_path.py
        ├── launch/
        │   ├── vision_arm_control.launch
        │   └── my_camera.launch
        ├── weights/                   # large .pth files TRACKED in Git
        ├── build_isolated/            # residual build cache (tracked)
        ├── devel_isolated/            # residual devel scripts (tracked)
        └── additions/                 # install helper shell scripts
```

**Submodules:** Five gitlinks are recorded but working trees are empty (no `.gitmodules` file, incomplete submodule init). **Verified.**

---

## 2. Existing architecture (as implemented)

```text
[OpenCV webcam] --CompressedImage--> [VisionArmControl monolithic node]
                                          |
                          +---------------+----------------+
                          |               |                |
                    MediaPipe pose   MonoDepth2 depth   ikpy Chain (2 links)
                          |               |                |
                     wrist xyz*      depth Image pub   JointState publish
                          |               |                |
                          +------ map ------+              |
                                 |                         v
                    PoseStamped (claimed base_link)   /franka_state_controller/joint_commands
```

\* MediaPipe landmark `.x/.y/.z` values are used **directly** as `PoseStamped` positions in frame `base_link` without pixel conversion, camera intrinsics, TF, or depth fusion. **Verified** in `scripts/main.py`.

Depth is published and displayed but **not** used to construct the robot target. **Verified.**

---

## 3. Confirmed working components

| Component | Status | Evidence |
|-----------|--------|----------|
| Package skeleton (catkin package.xml/CMakeLists) | Present | Files exist |
| MediaPipe pose API usage | Code present | `main.py` imports and processes landmarks |
| MonoDepth2 weight load paths | Partial | Paths hardcoded; load logic exists |
| Webcam compressed image publisher | Code present | `ros_camera_view.py` |
| Depth visualization screenshots | External GitHub assets in README | Linked images (not in repo) |
| Physical Franka execution | **Not verified in this environment** | No robot hardware; no logged success evidence in repo |
| Gazebo/MoveIt integration | **Not implemented as runnable launch** | Declared in package.xml deps only |
| TRAC-IK usage | **Not present in active scripts** | README claims TRAC-IK; `main.py` uses incomplete `ikpy` |

---

## 4. Incomplete components

| Item | Label | Notes |
|------|-------|-------|
| Full Panda kinematic chain | **Verified incomplete** | Only base + 2 links; comment `# ... add other links` |
| Camera calibration | **Verified missing** | No intrinsics YAML |
| TF2 camera→base | **Verified missing** | No `tf`/`tf2` usage for transform |
| Metric depth | **Verified not supported** | MonoDepth2 monocular disparity → relative depth only |
| MoveIt planning | **Verified missing** | No MoveIt API calls in scripts |
| Workspace / joint limits | **Verified missing** | No bounds checks |
| Pose-loss timeout | **Verified missing** | No staleness handling |
| E-stop / dead-man | **Verified missing** | Real command path unconstrained |
| Automated tests | **Verified missing** | No `tests/` directory on `main` |
| CI | **Verified missing** | No `.github/workflows` on `main` |
| Simulation demo path | **Verified missing** | No gazebo/moveit launch |

---

## 5. Hardcoded paths (**Verified**)

| Location | Path |
|----------|------|
| `CMakeLists.txt` | `/home/edr/miniconda3/envs/ros_torch/bin/python3` |
| `launch/vision_arm_control.launch` | `/home/edr/catkin_ws/src/vision_arm_control/weights/encoder.pth` |
| `launch/vision_arm_control.launch` | `/home/edr/catkin_ws/src/vision_arm_control/weights/depth.pth` |
| `monodepth2_node.py` default | `/home/edr/catkin_ws/src/vision_arm_control/weights/monodepth2_weights.pth` |
| `main.py` / monodepth2_node | `~/catkin_ws/src/monodepth2` on `sys.path` |

Root README also uses `yourusername` placeholder clone URL. **Verified** (top-level README).

---

## 6. Dependency problems

| Issue | Label |
|-------|-------|
| `package.xml` maintainer `Your Name` / `your_email@example.com` | **Verified** |
| License field `TODO` while MIT file exists in package | **Verified** |
| Version `0.0.0` | **Verified** |
| Declares `franka_gazebo`, `panda_moveit_config` without using them | **Verified** |
| Submodules not initialized; no `.gitmodules` | **Verified** |
| `setup.py` looks for packages under `src/` but no Python package tree exists | **Verified** |
| README requires TensorFlow for MediaPipe (unnecessary for modern MediaPipe) | **Likely outdated** |
| ROS not installed on audit host (`ROS_DISTRO=none`) | **Verified** |
| Full catkin build **Blocked** on this Scholar frontend without Noetic + deps | **Blocked** |

---

## 7. Safety risks (**Verified** unless noted)

1. Incomplete IK still publishes `JointState` toward a Franka-like topic.
2. No workspace clamping — MediaPipe coords can become arbitrary Cartesian targets.
3. No joint-limit validation.
4. No velocity/acceleration limits.
5. No pose-loss timeout (last pose could hold forever if not replaced carefully).
6. No dead-man switch; no `use_robot` gate.
7. GUI `cv2.imshow` in control callback (can block / fail headless).
8. Per-frame `rospy.loginfo` floods and adds latency under load. **Verified.**
9. Software safety cannot replace Franka FCI/Desk safety. **Document as requirement.**

---

## 8. Reproducibility risks

- Machine-specific absolute paths prevent clone-and-run.
- Weights (~111 MB working tree; ~120 MB pack) committed to Git history.
- Submodules empty without URLs.
- Residual `build_isolated`/`devel_isolated` confuse clean builds.
- No `requirements.txt` / Dockerfile / environment pin on `main`.
- Deprecated `ndarray.tostring()` in camera node. **Verified.**

---

## 9. Licensing risks

| Asset | Risk |
|-------|------|
| Package LICENSE (MIT) | Present under package; root LICENSE missing on `main` |
| package.xml `TODO` license | Metadata inconsistency **Verified** |
| MonoDepth2 (Niantic) | Research license restrictions apply; must attribute and not mis-license weights |
| MediaPipe | Apache-2.0 (third-party) |
| Franka / franka_ros | Separate licenses; submodules not vendored as source in tree |

---

## 10. Large-file problems (**Verified**)

| Path | Approx size |
|------|------------:|
| `weights/pose_encoder.pth` | 44.7 MB |
| `weights/encoder.pth` | 44.7 MB |
| `weights/depth.pth` / historical `monodepth2_weights.pth` | 12.0 MB |
| `weights/pose.pth` | 5.0 MB |
| Historical `monodepth2-master/assets/teaser.gif` | 8.8 MB |
| Historical monodepth2 split text files | 1.7–3.0 MB |
| Pack size | **~120.3 MiB** |

GitHub soft/hard limits: objects should stay well under 100 MB; generated binaries should not live in Git.

Adding `.gitignore` alone **does not** purge history. Cleanup requires `git filter-repo` (or equivalent) on an authorized branch/migration, with artifacts preserved outside Git first.

---

## 11. Duplicate code

- `main.py` embeds depth estimation **and** `monodepth2_node.py` is a second depth pipeline. **Verified.**
- Root README vs package README disagree on install steps and claims. **Verified.**
- `ikpy` used in code; TRAC-IK claimed in docs but not called. **Verified.**

---

## 12. Missing tests

No unit, launch, or integration tests on `main`. **Verified.**

---

## 13. README claim vs evidence matrix

| Claim | Evidence |
|-------|----------|
| Human pose via MediaPipe | Code present |
| Depth via MonoDepth2 | Code present; weights present |
| Franka control / mimicry | Partial mapping only; incomplete IK |
| TRAC-IK | **Not in active code** |
| Real-time control | **Unverified** (no measured FPS/latency) |
| Mostly simulated testing | **Unverified** — no sim launch artifacts found |
| Motion smoothing low-pass | **Not found** in `main.py` |
| 2D→3D coordinate transform | **Not correctly implemented** (normalized coords used as base_link) |

---

## 14. Recommended implementation order

1. Preserve weights outside Git; tighten `.gitignore`; document model download.
2. Fix package metadata, paths, accidental files, license at root.
3. Extract pure Python modules (mapping, safety, filters) with unit tests.
4. Split nodes; YAML config; safe launch defaults (`dry_run`, `use_robot:=false`).
5. Frame pipeline scaffolding + calibration docs (example values labeled).
6. Simulation/mock demonstration path without hardware.
7. Benchmark tooling (no fabricated metrics).
8. Portfolio README + diagrams + Scholar SLURM templates + CI.

---

## 15. Environment of this audit

| Item | Value |
|------|-------|
| Host | Purdue Scholar (`/scratch/scholar/edraymon`) |
| Python | 3.9.21 |
| ROS | Not installed on audit host |
| GPU job system | SLURM available via RCAC modules |
| Weights preserved to | `/scratch/scholar/edraymon/franka-teleop-data/models/` |

---

*End of audit. All subsequent portfolio-v2 work should address these verified issues without inventing unmeasured results.*
