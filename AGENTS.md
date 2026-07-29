# AGENTS.md

## Cursor Cloud specific instructions

This repo is a ROS-based Franka Panda teleoperation portfolio. Two developer paths exist:
the **native Python path** (no ROS, works in this VM) and the **ROS paths** (Noetic container /
ROS 2 workspace) which need extra runtimes not present here. The startup update script sets up
only the native path.

### Environment (already provisioned by the update script)
- The native path runs on **Python 3.10** (matching CI), managed via `uv`. The venv lives at
  `~/.venv-franka` (outside the repo tree on purpose — a venv inside `/workspace` trips the
  repo's binary-hygiene checks in `tests/test_tree_layout.py` and `scripts/validate_repository.py`).
- Activate with `source ~/.venv-franka/bin/activate` before running commands below.
- Only `requirements-ci.txt` (numpy, PyYAML, pytest, Pillow, black, flake8) is installed. The
  full `requirements.txt` (torch, mediapipe, opencv) targets Python 3.8–3.10 + ROS and is not
  needed for native dev/tests; it is not installed.

### Native checks (mirror the CI `quality` job)
Run from the repo root with the venv active. Commands are defined in `.github/workflows/ci.yml`:
`pytest -q tests/`, `python scripts/validate_repository.py`, the two `benchmarks/*.py`,
`black --check ...`, `flake8 ...`, and `bash scripts/audit_git_history.sh`.

- The benchmark scripts (`benchmarks/benchmark_mapping.py`, `benchmarks/benchmark_latency.py`)
  **overwrite** `results/mapping_benchmark.json` and `results/latency_benchmark.json` with the
  current host's numbers. Do not commit that machine-specific churn — `git checkout -- results/`
  after running them.
- Repo-hygiene guards forbid any `*.pth/*.pt/*.onnx/*.bag` and any file >10 MiB anywhere under the
  repo root (`.git` excluded). Keep scratch venvs, model weights, and large artifacts outside
  `/workspace`.

### Running the "application" without a robot
There is no long-running server. Core functionality is the teleop pipeline: mock wrist landmarks
→ shoulder-relative EE mapping (`coordinate_mapping.py`) → safety gate (`safety.py`,
`workspace_limits.py`, `pose_timeout.py`). It can be driven directly by importing the package from
`src/vision_arm_control/src`. The ROS entrypoints (`roslaunch vision_arm_control simulation.launch`,
etc.) require ROS Noetic (use `Dockerfile.noetic`) and the ROS 2 workspace requires ROS 2 — neither
runtime is installed in this VM.
