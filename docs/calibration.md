# Calibration

## Goal

Convert MediaPipe normalized image coordinates into robot base-frame targets using a documented chain:

```text
normalized image (u,v)
  → pixel coordinates
  → camera-frame 3D (requires depth or proxy)
  → TF camera_optical_frame → panda_link0
  → workspace validation
```

## Intrinsics

File: `src/vision_arm_control/config/camera_intrinsics.yaml`

Current status: **`example`** — not a measured calibration for a specific lab camera.

To measure:

1. Use a checkerboard with `camera_calibration` (ROS) or OpenCV calibrateCamera.
2. Record `fx, fy, cx, cy`, distortion, and image size.
3. Set `status: measured` only after a real capture session.

Validate:

```bash
python scripts/calibration_check.py
```

## Extrinsics (camera → base)

File: `src/vision_arm_control/config/mapping.yaml` → `frames.camera_in_base_*`

Current status: **`example`**.

Recommended methods:

- Hand-eye calibration if the camera is eye-in-hand or eye-to-hand fixed.
- Measured CAD / metrology for a rigidly mounted scene camera.
- AprilTag on the robot base for a coarse extrinsic.

## Depth

| Source | Metric? | Notes |
|--------|---------|-------|
| MonoDepth2 monocular | **No** (relative) | Visualization / proxy only without scale |
| RGB-D camera (RealSense, etc.) | Yes (within sensor accuracy) | Preferred for 3D mapping |
| Calibrated monocular scale | Approx. | Requires known target or offline fit |

## Known limitations

- Example intrinsics/extrinsics must not be presented as lab measurements.
- MediaPipe `z` is not metric camera depth.
- Without measured TF, shoulder-relative 2D mapping is the safer teleoperation mode.

## Related documentation

- Safety gates that consume mapped targets: [`docs/safety.md`](safety.md)
- Model weights (not metric depth): [`docs/model-setup.md`](model-setup.md)
- Repository audit (historical mapping issues): [`docs/repository-audit.md`](repository-audit.md)

## After a real calibration session

1. Update `camera_intrinsics.yaml` and set `status: measured`.
2. Update `mapping.yaml` `frames.camera_to_base_status` to `measured` only with evidence.
3. Document capture method, date, camera model, and residual error here.
4. Only then consider updating README metrics away from “Not yet measured”.
