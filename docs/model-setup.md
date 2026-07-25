# Model setup

Model weights are **not** stored in Git.

## Required files

| File | Role |
|------|------|
| `encoder.pth` | MonoDepth2-style ResNet encoder |
| `depth.pth` | Depth decoder |

Optional (research artifacts from HUMANS MOVE):

| File | Role |
|------|------|
| `pose_encoder.pth` | Pose network encoder (if used) |
| `pose.pth` | Pose network weights (if used) |
| `poses.npy` | Auxiliary numpy artifact |

## Where to put them

```bash
export FRANKA_MODEL_DIR=/path/to/models
# Purdue Scholar example:
export FRANKA_MODEL_DIR=$RCAC_SCRATCH/franka-teleop-data/models
```

## Download / install

```bash
bash scripts/download_models.sh "$FRANKA_MODEL_DIR"
```

Official MonoDepth2 pretrained models: [nianticlabs/monodepth2](https://github.com/nianticlabs/monodepth2#pretrained-models).

Research weights from the Wyoming HUMANS MOVE work should be copied from your archive or Scholar data directory — they are not rehosted by this repository.

## Depth semantics

MonoDepth2 monocular inference produces **relative** depth from disparity.  
It is **not** metric distance unless you calibrate scale (e.g., against RGB-D or known target size).

## License

Comply with MonoDepth2 / Niantic research terms. Cite MonoDepth2 in publications and demos.

## History cleanup note

`main` history still contains large `.pth` objects (~120 MB pack).  
`portfolio-v2` stops tracking weights going forward. To purge history entirely (authorized migration only):

```bash
# After installing git-filter-repo and backing up:
git filter-repo --path src/vision_arm_control/weights/ --invert-paths
# plus other large historical paths as needed
```

Do **not** rewrite `main` without explicit authorization.
