# Scholar Completion Runbook + Meta Prompt

Independent post-program engineering by Eric Raymond (2026).

## Status (completed)

**PR #5 merged to `main`** (squash `2a89ffa`). Both environment-bound gates are
done on Scholar dry_run (no physical Franka):

| Gate | Status | Evidence |
|------|--------|----------|
| **S2-A** RViz + live diagnostics | **Done** (job 459478) | `docs/media/v1.1-hardening/`, `results/v1.1-hardening/ros2/` scenario/rviz JSONL+metrics+bags |
| **S2-B** MoveIt `compute_ik` JointStates | **Done** (job 459481) | `s2b_joint_evidence.json` (`verified_moveit_jointstate_evidence`, 16/16), `scenario_s2b_moveit_ik.jsonl` |

Reproduce with `sbatch scholar/record_v11_hardening.slurm` and/or
`sbatch scholar/record_s2b_ik.slurm` inside `ros2_humble_franka.sif`.

---

Historical meta-prompt below (kept for reproducibility; gates no longer blocked):

## What is already done (no Scholar work needed)

| Prepared here (CI-verified) | Where |
|---|---|
| Exact, KKT-verified CBF-QP + OSQP/linprog oracle (6000 QPs, 0 mismatches) | `safety_filters/cbf_qp.py`, `scripts/qp_cross_validation.py`, `docs/research/qp-correctness-audit.md` |
| Independent discrete-step next-state validator | `safety_filters/cbf_qp.py::validate_next_state` |
| Paired 30-replicate benchmark + fault injection (1800 runs, 0 unsafe) | `benchmarks/benchmark_paired_v11.py`, `results/v1.1-hardening/paired/` |
| Versioned config schema + fixtures | `config_schema.py`, `config/fixtures/`, `scripts/validate_config.py` |
| Explicit frame/unit contracts | `docs/research/frame-contracts.md` |
| **Hardened pipeline wired to ROS 2** (selectable retargeter/safety/gate/obstacles) | `ros2_ws/.../teleop_pipeline_node.py` |
| Structured diagnostics record (all Phase-9 fields, no silent fallback) | `pipeline_diagnostics.py` (+ CI tests) |
| Offline metrics extractor (CI-tested aggregator) | `scripts/extract_run_metrics.py` |
| Demo launch + record script | `.../launch/teleop_pipeline_demo.launch.py`, `scripts/ros2_record_demo.sh` |

The ROS 2 node imports the same numpy-only core that is cross-validated in CI, so
the runtime behavior is the audited behavior — Scholar only needs to *run and
record* it, plus wire the MoveIt IK service for the verified-joint gate.

## Small remaining work on Scholar (pull and run)

```bash
# 0. Pull this branch on the ROS 2 host.
git fetch origin && git checkout research-retargeting-v1.1 && git pull

# 1. Build the ROS 2 workspace and make the core importable.
cd ros2_ws && colcon build --symlink-install && source install/setup.bash && cd ..
export PYTHONPATH="$PWD/src/vision_arm_control/src:${PYTHONPATH:-}"

# 2. Sanity: pure tests + QP oracle + config schema (should pass as in CI).
python3 -m pip install -r requirements-ci.txt
python3 -m pytest -q tests/
python3 scripts/qp_cross_validation.py --generic 400 --cbf 400
python3 scripts/validate_config.py

# 3. Bring up Panda fake hardware / RViz in one terminal:
ros2 launch franka_teleop_bringup panda_rviz_sim.launch.py     # or a MoveIt Panda fake-hw bringup

# 4. Record the three scenarios (writes diagnostics JSONL + rosbag + metrics):
DUR=20 bash scripts/ros2_record_demo.sh results/v1.1-hardening/ros2

# 5. Screen-record RViz to MP4/GIF with the burned-in labels (see below), save to
#    docs/media/v1.1-hardening/end-to-end-rviz.{mp4,gif} and write its README.

# 6. (S2-B) Wire verified IK: implement TeleopPipelineNode._moveit_ik using the
#    /compute_ik service (moveit_msgs/GetPositionIK), enforce PANDA_JOINT_LIMITS
#    (already imported + checked before publishing), then re-run scenario B with
#    ik_backend:=moveit and confirm JointState evidence in the bag.
```

### Required burned-in video labels (S2-A)
`ROS 2 SIMULATION` · `MOCK LANDMARKS` · `PANDA FAKE HARDWARE` · `NO PHYSICAL ROBOT`
· `RETARGETER=<...>` · `SAFETY FILTER=<...>` · `CURRENT GATE STATE`. Show live
diagnostic values from `/teleop/diagnostics` (confidence, accepted/rejected,
intervention magnitude, active constraint, solver status, latency).

### After recording — update claims/evidence (small edits)
- `docs/research/claims-ledger.md`: flip items **10** (RViz demo) and, if MoveIt
  wired, **11** (verified IK) from `P` to `E` with the committed evidence paths.
- `docs/research/red-team-review-v1.1.md`: mark **S2-A** (and **S2-B** if done)
  resolved; update the release-readiness paragraph.
- `docs/research/simulation-validation-results.md`: add the ROS 2 topic rates,
  callback-to-command latency, and (if MoveIt) achieved-EE-orientation metric.
- Commit media + `results/v1.1-hardening/ros2/` + doc edits. Keep PR draft until a
  human authorizes the release.

---

## META PROMPT (paste into a Scholar cloud/agent session)

> You are the release captain finishing PR #5 (`research-retargeting-v1.1`) of
> the Franka teleoperation repo, now on a ROS 2 host with MoveIt 2, a Panda
> MoveItConfig, RViz, and a display. All non-ROS hardening is already done and
> CI-green; do **not** re-do it. Your job is only the two environment-blocked
> gates, truthfully.
>
> Rules (non-negotiable): keep `use_robot=false`/`dry_run`; solver failure ⇒ no
> command; never publish joints that fail `PANDA_JOINT_LIMITS`; never claim
> physical Franka validation, metric depth, calibrated camera-to-base, verified
> IK unless MoveIt actually produced limit-checked joints, or paper parity. Do
> not merge; keep the PR draft. Work on the existing branch; no history rewrite.
>
> Steps:
> 1. `git checkout research-retargeting-v1.1 && git pull`; `cd ros2_ws && colcon
>    build --symlink-install && source install/setup.bash`; `export
>    PYTHONPATH=$PWD/../src/vision_arm_control/src:$PYTHONPATH`.
> 2. Verify parity with CI: `pytest -q tests/`, `python scripts/qp_cross_validation.py`,
>    `python scripts/validate_config.py` — all must pass.
> 3. Launch Panda fake hardware / RViz, then run
>    `bash scripts/ros2_record_demo.sh results/v1.1-hardening/ros2` and confirm
>    `/teleop/diagnostics` shows the selectable retargeter, safety mode, gate
>    state, solver status, active constraints, intervention magnitude, and
>    latency changing live.
> 4. Record ONE RViz MP4 (+GIF) per the three scenarios with the burned-in labels
>    listed in the runbook and live diagnostic overlays; save under
>    `docs/media/v1.1-hardening/` with a `README.md` (exact commands + commit SHA).
> 5. (S2-B) Implement `TeleopPipelineNode._moveit_ik` via the `/compute_ik`
>    service, enforce joint limits, and produce recorded `JointState` evidence
>    with `ik_backend:=moveit`. If MoveIt IK cannot be made reproducible, keep
>    `sew_orientation` feature-only (`joint_positions=None`), record the exact
>    blocker, and do NOT fake joints.
> 6. Run `scripts/extract_run_metrics.py` on each diagnostics JSONL; commit
>    `results/v1.1-hardening/ros2/` + media, then update `claims-ledger.md`,
>    `red-team-review-v1.1.md`, and `simulation-validation-results.md` to flip the
>    now-satisfied gates to Evidence with paths. Push; leave the PR draft; post a
>    PR comment with the demo links, topic rates, latency, and remaining
>    non-claims.
>
> Deliverables: RViz MP4/GIF from the real runtime; `results/v1.1-hardening/ros2/`
> (bags, diagnostics JSONL, metrics JSON); verified-IK JointState evidence or a
> documented blocker; updated ledger/red-team/results docs; a PR comment. Report
> exactly what ran, what each artifact shows, and any gate still open.
