# Demo Media Provenance

All media is labeled to distinguish simulation, synthetic visualization, and historical research output.

| File | Provenance | What it proves |
|---|---|---|
| `franka-rviz-dry-run.*` | Purdue Scholar jobs 459328/459329; ROS 2 Humble + RViz 2 fake hardware | Mock landmark trajectory passes mapping and safety and produces moving Panda joint states at about 20 Hz |
| `safety-workspace-violation.*` | `scripts/generate_demo_media.py`; repository workspace validation utility | An out-of-bounds synthetic target is rejected before command acceptance |
| `perception-headless-relative-depth.*` | `scripts/generate_demo_media.py`; synthetic landmarks and relative disparity | Message shape and relative-depth semantics only; not webcam or metric-depth proof |

Rebuild the synthetic media:

```bash
python scripts/generate_demo_media.py
```

The RViz demo is not Gazebo and does not use physical hardware. The historical depth screenshots linked from the README are research-period perception artifacts, not evidence of Franka tracking.
