# Large-file history cleanup procedure

## Current state

- Weights were preserved to `$RCAC_SCRATCH/franka-teleop-data/models/` (or equivalent).
- `portfolio-v2` stops tracking `weights/` and binary artifacts via `.gitignore`.
- Git **history on `main` still contains** large blobs (~120 MB pack).

## Affected paths (from audit)

- `src/vision_arm_control/weights/*.pth`
- historical `monodepth2-master/` assets and splits
- residual build artifacts

## Safe migration (do not run on `main` without authorization)

```bash
# 1. Backup
git clone --mirror https://github.com/raayraay96/summer25-Franka-ros-noetic.git backup-mirror.git

# 2. Install git-filter-repo
python3 -m pip install git-filter-repo

# 3. On a migration branch / fresh clone only:
git filter-repo \
  --path src/vision_arm_control/weights/ \
  --path src/vision_arm_control/monodepth2-master/ \
  --invert-paths

# 4. Force-push only after review and authorization:
# git push origin --force portfolio-v2
```

## Alternative: orphan portfolio branch

Ship `portfolio-v2` as a clean tree without rewriting `main` history (this branch already removes weights from the working tree going forward). Clone depth for reviewers:

```bash
git clone --branch portfolio-v2 --single-branch \
  https://github.com/raayraay96/summer25-Franka-ros-noetic.git
```

Note: default clone of full history may still download historical blobs until filter-repo is applied.
