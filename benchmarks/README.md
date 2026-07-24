# Benchmarks

These scripts measure **what can be measured without Franka hardware**.

| Script | What it measures | Requires |
|--------|------------------|----------|
| `benchmark_mapping.py` | Coordinate mapping latency | numpy |
| `benchmark_latency.py` | Map + filter + workspace stages | numpy |
| `benchmark_perception.py` | MediaPipe on synthetic frames | mediapipe, opencv |

## Run

```bash
python benchmarks/benchmark_mapping.py
python benchmarks/benchmark_latency.py
python benchmarks/benchmark_perception.py  # optional deps
```

Outputs land in `results/*.json`.

## Not yet measured (physical system)

- End-to-end camera → Franka latency
- Tracking error vs motion capture
- Collision-free success rate
- Real-robot FPS under load
- GPU MonoDepth2 throughput on Scholar (use SLURM templates)

When you measure them, record hardware, OS, ROS version, model, resolution, command, and date.
