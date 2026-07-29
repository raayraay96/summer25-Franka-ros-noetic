# Frame and Unit Contracts (Phase 3)

Independent post-program engineering by Eric Raymond (2026). This document makes
the retargeting frames, axes, units, and calibration status **explicit** so no
implicit image-axis interpretation or metric-depth claim is hidden in code.

## Named frames

| Frame | Meaning |
|---|---|
| `image_normalized` | MediaPipe-style normalized image coords; x,y in [0,1]; z is *relative* landmark depth |
| `camera_optical` | Pinhole camera optical frame (only used by the metric back-projection path) |
| `human_shoulder` | Shoulder-anchored body frame for shoulder-relative displacement |
| `robot_base` | Franka base frame (`base_link`) — the target frame for EE position |
| `end_effector` | Panda EE (`panda_hand` / flange) — orientation target only |

## RetargetingTarget provenance fields

`RetargetingTarget` now carries an explicit contract (`landmarks/schema.py`):

| Field | Image-relative path | Metric-ish body path |
|---|---|---|
| `source_frame` | `image_normalized` | `human_shoulder` |
| `target_frame` | `robot_base` | `robot_base` |
| `units` | `normalized_and_scaled` | `metric_scaled` |
| `z_is_relative` | `True` (NOT metric) | `False` |
| `calibration_status` | `example_not_calibrated` | `example_not_calibrated` |
| `transform_provenance` | `configured_axis_scale_offset;image_relative_z_scale` | same |

## Image-relative path — exact statement

- `x` and `y` are **normalized image coordinates** (dimensionless, [0,1]).
- `z` is **relative monocular landmark depth**, not meters. No metric scale is
  claimed and no camera-to-base calibration is implied.
- The relative-depth axis is de-weighted by the **named** parameter
  `image_relative_z_scale` (default `0.5`) in `RetargetingConfig`. This replaces
  the previous undocumented `chain[2] * 0.5` literal. It exists because the
  MediaPipe z channel has a different, uncalibrated scale and higher noise than
  the in-plane x/y; the factor reduces its influence. It is a documented
  heuristic, not a calibrated or metric transform.

## Axis / scale / offset mapping (example, not calibrated)

The workspace mapping is `position = workspace_origin + workspace_scale * disp`.
The intended configurable contract (mirrored in `config/mapping.yaml`) is:

```yaml
frame_mapping:
  source_frame: image_normalized
  target_frame: robot_base
  axis_order: [x, y, z]          # applied before scale/offset
  scale:  [0.55, 0.55, 0.35]      # workspace_scale
  offset: [0.45, 0.0, 0.45]       # workspace_origin
  image_relative_z_scale: 0.5     # named z de-weighting (relative depth only)
  status: example_not_calibrated
```

## Non-claims

- No metric depth from the monocular path.
- No calibrated camera-to-base geometry (status is `example_not_calibrated`
  until measured calibration evidence exists).
- `sew_orientation` emits **feature-level** Cartesian position + an orientation
  *direction* signal. `joint_positions` remains `None` with
  `ik_status = missing_ik_layer_feature_level_only`; no joint retargeting is
  claimed (see `simulation-validation-results.md` / final limitations).
