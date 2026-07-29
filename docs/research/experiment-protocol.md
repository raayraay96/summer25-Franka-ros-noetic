# Experiment Protocol — Retargeting v1.1

## Goal

Compare baseline shoulder-relative teleop and SEW-inspired retargeting under
reject, clamp, and experimental CBF-QP safety filters on **deterministic synthetic
landmark trajectories** (CPU pure-Python). Host identity is recorded; do not
compare hosts without noting them.

## Reproduction command

```bash
python3 benchmarks/benchmark_retargeting_v11.py \
  --config benchmarks/config/v11.yaml \
  --output results/v1.1
```

CI smoke:

```bash
python3 benchmarks/benchmark_retargeting_v11.py --smoke --output results/v1.1
```

## Combinations

| Retargeter | Safety |
|------------|--------|
| shoulder_relative | reject |
| shoulder_relative | clamp |
| shoulder_relative | cbf_qp |
| sew_orientation | reject |
| sew_orientation | clamp |
| sew_orientation | cbf_qp |

## Trajectories

1. horizontal_sweep  
2. vertical_sweep  
3. circle  
4. figure_eight  
5. step_input  
6. near_full_extension  
7. noisy_landmarks  
8. one_second_dropout  
9. workspace_boundary_crossing  
10. obstacle_intersection  

## Metric definitions

| Metric | Definition |
|--------|------------|
| Mapping latency mean/median/p95/p99/max | Wall time of `retarget()` over steps with valid landmarks (ms) |
| Safety-filter latency mean/p95 | `SafetyFilterResult.compute_time_s` (ms) |
| Orientation-alignment error | Mean \(\arccos(\hat d_i \cdot \hat d_{i+1})\) over consecutive lower-arm directions (rad) |
| EE path length | \(\sum_i \|p_{i+1}-p_i\|\) over commanded positions (m) |
| Path jerk proxy | Mean \(\|\Delta a\|/\Delta t\) with discrete velocity/acceleration (m/s³ proxy) |
| Constraint-violation count | Sum of per-step solver-reported residual violations |
| Intervention count / % | Steps where filter intervened; percent of steps |
| Mean intervention magnitude | Mean \(\|u^\star - u_{\mathrm{nom}}\|\) when intervened |
| Solver-failure count | Steps with `solver_failure` reject |
| Dropped-command count | Steps with `commanded=False` |
| Recovery time | Time from reacquisition after dropout until first commanded step (s) |
| Joint-limit violations | Always 0 at feature-level (no joint outputs) |
| Min obstacle clearance | \(\min_i (\|p_i - c\| - r)\) for configured spheres (m) |
| Repeatability | Path-length mean/stdev across `--repeat N` runs with fixed seeds |

## Outputs

```
results/v1.1/
  benchmark-summary.json
  benchmark-runs.csv
  environment.json
  config-snapshot.yaml
  plots/
  logs/
```

## What is not measured

- Physical Franka tracking accuracy  
- Full ROS perception latency  
- Formal CBF certificates  
- SEW-Mimic joint retargeting error (IK layer not implemented)  
