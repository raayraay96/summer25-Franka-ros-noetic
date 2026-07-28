# Model Setup

Model weights are deliberately outside Git because pretrained checkpoints are large, change independently from code, and retain third-party license terms. Keeping them external makes clones small, prevents accidental redistribution, and lets CI exercise mapping and safety without downloading a model.

## Expected files

| File | Role | Required for no-ROS tests? |
|---|---|---|
| `encoder.pth` | MonoDepth2-style encoder | No |
| `depth.pth` | Relative-depth decoder | No |
| `pose_encoder.pth`, `pose.pth`, `poses.npy` | Optional research artifacts | No |

## Configure the model directory

```bash
export FRANKA_MODEL_DIR=/absolute/path/to/models
bash scripts/download_models.sh "$FRANKA_MODEL_DIR"
```

Purdue Scholar example:

```bash
export FRANKA_MODEL_DIR="$RCAC_SCRATCH/franka-teleop-data/models"
bash scripts/download_models.sh "$FRANKA_MODEL_DIR"
```

The script verifies expected filenames and optional SHA-256 checksums. It does not rehost research weights. Obtain official pretrained models from [Niantic's MonoDepth2 repository](https://github.com/nianticlabs/monodepth2#pretrained-models), or copy authorized HUMANS MOVE research artifacts from the project archive.

## Depth warning

MonoDepth2-style inference produces relative depth from disparity. It is not metric distance without measured scale calibration. See [calibration](calibration.md).

## Repository history

The `v1.0-portfolio` migration rebuilt the public branches from a clean root after removing historical model, bag, and build artifacts. CI runs [`scripts/audit_git_history.sh`](../scripts/audit_git_history.sh) to reject any blob over 10 MiB, forbidden model/bag paths, or a packed repository over 20 MiB.
