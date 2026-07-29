# v1.1-hardening end-to-end RViz evidence (S2-A)

Independent post-program engineering by Eric Raymond (2026).

| Item | Value |
|------|--------|
| SLURM job | **459478** |
| Host | Scholar `scholar-b000` (`scholar-b000.rcac.purdue.edu`) |
| Container | `ros2_humble_franka.sif` |
| Commit | `5403fe775f80c1fac604d695178314662179f9da` |
| ROS | Humble (`dry_run` only) |
| Backend | RViz 2 fake-hardware / joint_state simulation |
| Physical robot | **No** |
| Inputs | Mock landmarks (no live camera / depth for control) |
| IK in this recording | `geometric_unverified` (not verified MoveIt; S2-B evidence is separate) |
| Scenario A (shoulder+cbf) | ~20.05 Hz, mean latency **0.388** ms, intervention 0%, acceptance 1.0, 428 records |
| RViz demo metrics | ~20.04 Hz, mean latency ~11.6 ms, intervention ~51.3%, 532 records |
| Labels | ROS 2 SIMULATION · MOCK LANDMARKS · PANDA FAKE HARDWARE · NO PHYSICAL ROBOT · RETARGETER · SAFETY FILTER · live diagnostics topic |

## Files

- `end-to-end-rviz.mp4` — labeled screen capture of the real ROS 2 runtime
- `end-to-end-rviz.gif` — short GIF derivative
- `end-to-end-rviz-thumb.png` — thumbnail

## How produced

```bash
sbatch scholar/record_v11_hardening.slurm
# launch: teleop_pipeline_rviz.launch.py
#   mock landmarks → TeleopPipeline (shoulder_relative + cbf_qp) → geometric_unverified IK → RSP → RViz
```

Live diagnostics JSONL and metrics: `results/v1.1-hardening/ros2/`
(`rviz_demo.jsonl`, `rviz_demo_metrics.json`, scenario_a/b/c jsonl+metrics+bags,
`environment.json`). Provenance host/job/container/SHA are in
`results/v1.1-hardening/ros2/environment.json`. Job log:
`franka-teleop-data/logs/v11-s2-459478.out`.

**Not** verified MoveIt IK evidence. This media package is **S2-A only**
(`geometric_unverified` IK). Do not read it as joint retargeting validation or
as S2-B. Verified MoveIt `/compute_ik` JointState evidence (S2-B, claim 11)
lives under results, not media:

- `results/v1.1-hardening/ros2/s2b_joint_evidence.json`
  (`status=verified_moveit_jointstate_evidence`, 16/16, job **459481**)
- `results/v1.1-hardening/ros2/scenario_s2b_moveit_ik.jsonl`
- `results/v1.1-hardening/ros2/scenario_s2b_moveit_ik_bag/`

S2-B is dry_run Cartesian-target MoveIt IK only; `sew_orientation` remains
feature-level. No physical Franka.
