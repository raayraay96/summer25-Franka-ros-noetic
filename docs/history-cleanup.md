# Large-File History Cleanup

## Completed migration

The portfolio release rebuilt `main` and `portfolio-v2` from a clean root commit containing only the reviewed source tree and compressed demo media. Historical `models/`, `*.pt`, `*.pth`, `*.bag`, catkin outputs, machine-specific paths, and abandoned branches were removed from reachable public history.

## Verification

```bash
git fetch --all --prune
git count-objects -vH
bash scripts/audit_git_history.sh
```

The automated audit fails when:

- any reachable blob exceeds 10 MiB;
- a reachable path contains `models/`, `*.pt`, `*.pth`, or `*.bag`;
- packed reachable objects reach 20 MiB.

## Recovery note

Research weights remain in the authorized project archive or `$RCAC_SCRATCH/franka-teleop-data/models`. They are intentionally not part of the Git repository or release assets.
