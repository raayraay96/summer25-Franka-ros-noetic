# SEW-Inspired Retargeting (v1.1)

**Status:** Paper-inspired baseline, feature-level output  
**Inspired by:** SEW-Mimic (arXiv:2602.01632)  
**Module:** `vision_arm_control.retargeting.sew_orientation`

## What is implemented

Given shoulder, elbow, and wrist landmarks (3D or documented 2.5D):

1. **Upper-arm unit:** `normalize(elbow − shoulder)`
2. **Lower-arm unit:** `normalize(wrist − elbow)`
3. **Elbow bend angle** between those units
4. **Arm-plane normal** `normalize(upper × lower)` when not collinear
5. **Confidence-weighted validity** and explicit failure codes
6. **Cartesian EE target** in robot base frame via configurable workspace origin/scale
7. **Orientation / direction** heuristic from lower-arm unit (+ plane normal)

## What is not implemented

- Closed-form SEW-Mimic **7-DoF joint** solver
- Optimality guarantees from the paper
- Analytical or MoveIt IK claiming SEW joint mimicry
- Bimanual self-collision retargeting from the paper

`RetargetingTarget.joint_positions` is always `None` with  
`ik_status = "missing_ik_layer_feature_level_only"`.

## Failure states

| Code | Condition |
|------|-----------|
| `missing_shoulder` / `missing_elbow` / `missing_wrist` | Absent keypoint |
| `low_confidence` | Below threshold |
| `upper_arm_norm_near_zero` / `lower_arm_norm_near_zero` | Degenerate segment |
| `collinear_arm_plane` | Soft: plane invalid; position still produced when units OK |
| `non_finite` | NaN/Inf |
| `stale_timestamp` | Handled in confidence gate |

## Fallback

Invalid landmarks never invent motion: the **confidence gate** holds the last safe target for a short interval, then stops.

## Configuration

```yaml
mapping:
  mode: sew_orientation   # default remains shoulder_relative
```

## Truthfulness

This is **inspired by** SEW-Mimic orientation alignment ideas. It is **not** a full SEW-Mimic reproduction.
