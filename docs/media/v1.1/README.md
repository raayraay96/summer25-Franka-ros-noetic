# v1.1 Simulation Media

**Provenance:** Generated on Purdue Scholar frontend with pure-Python teleop pipeline  
**Script:** `scripts/generate_v11_demo_media.py`  
**Not:** physical Franka · Gazebo · live RViz re-record (unless noted)  
**Labels burned into frames:** SIMULATION · MOCK LANDMARKS · NO PHYSICAL ROBOT · RETARGETER · SAFETY FILTER

## Demos

| Demo | Files | Content |
|------|-------|---------|
| A | [`demo_a_shoulder_vs_sew.gif`](demo_a_shoulder_vs_sew.gif) · [mp4](demo_a_shoulder_vs_sew.mp4) | Side-by-side shoulder_relative vs sew_orientation (reject) |
| B | [`demo_b_cbf_obstacle.gif`](demo_b_cbf_obstacle.gif) · [mp4](demo_b_cbf_obstacle.mp4) | Nominal path vs CBF-QP near simulated sphere |
| C | [`demo_c_confidence_gating.gif`](demo_c_confidence_gating.gif) · [mp4](demo_c_confidence_gating.mp4) | Landmark dropout → hold → stop → recovery |

## Reproduce

```bash
export PYTHONPATH=src/vision_arm_control/src
python3 -m pip install matplotlib pillow imageio imageio-ffmpeg
python3 scripts/generate_v11_demo_media.py
```

## Truth labels

| Claim | True? |
|-------|-------|
| Simulation | Yes (matplotlib trajectories) |
| Mock landmarks | Yes |
| No physical robot | Yes |
| RViz 2 fake-hardware re-run for v1.1 | **No** (v1.0 RViz media remains under `docs/media/`) |
| Gazebo | No |
