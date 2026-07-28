# Calibration

## Coordinate chain

```text
normalized image (u, v)
  → pixel coordinates
  → camera-frame 3D (only with a valid depth source or documented proxy)
  → measured TF: camera_optical_frame → panda_link0
  → workspace validation
```

## Camera intrinsics

Configuration: `src/vision_arm_control/config/camera_intrinsics.yaml`.

**Current status: `example`.** The committed `fx`, `fy`, `cx`, `cy`, distortion, and image size are not a measured calibration for a specific camera. Set `status: measured` only after recording the camera model, capture date, calibration target, method, and residual error.

```bash
python scripts/calibration_check.py
```

## Camera-to-base extrinsics

Configuration: `src/vision_arm_control/config/mapping.yaml`.

**Current status: `example`.** Measure an eye-to-hand or eye-in-hand transform with an appropriate hand-eye, fiducial, CAD/metrology, or lab-approved method. Do not present an example transform as lab evidence.

## Relative depth semantics

The monocular visualization uses:

```text
relative_depth = 1 / max(disparity, epsilon)
```

That value is **not meters**. MonoDepth2-style monocular output has unknown scale without a measured scale recovery procedure. It is retained for perception research and visualization, not used as validated metric control input.

| Source | Metric? | Intended use |
|---|---|---|
| MonoDepth2-style disparity | No | Relative ordering / visualization |
| MediaPipe landmark `z` | No | Pose-relative feature only |
| Calibrated RGB-D camera | Yes, within sensor uncertainty | Preferred 3D mapping source |
| Monocular scale fit | Approximate | Only with documented targets and error analysis |

Shoulder-relative 2D mapping is the safer default until camera intrinsics, scale, and the camera-to-base transform are measured.
See also: [safety](safety.md) · [model setup](model-setup.md) · [repository audit](repository-audit.md).
