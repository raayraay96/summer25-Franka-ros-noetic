# Purdue Scholar workflow

Scholar is the heavy workspace. GitHub holds code + small portfolio artifacts only.

## Data layout

```text
$RCAC_SCRATCH/franka-teleop-data/
├── models/          # .pth weights
├── datasets/
├── rosbags/
├── raw_video/
├── outputs/
├── benchmarks/
└── logs/
```

Scratch is for temporary large computation; it is **not backed up**. Archive important final results to a persistent location.

## Environment

```bash
export REPO_DIR=/path/to/summer25-Franka-ros-noetic
export RCAC_SCRATCH=/scratch/scholar/$USER   # usually already set
export FRANKA_MODEL_DIR=$RCAC_SCRATCH/franka-teleop-data/models
```

## Submit jobs (via SLURM)

On Scholar, include a valid account (example: `scholar` for CPU partitions):

```bash
cd $REPO_DIR
sbatch --account=scholar scholar/run_benchmark.slurm
sbatch --account=scholar scholar/run_perception.slurm   # GPU if configured in the script
sbatch --account=scholar scholar/generate_results.slurm
```

Do **not** run GPU workloads on the frontend/login node.

GPU perception jobs require a GPU account/partition (for example `gpu` / `scholar-gpu`) if your site associations require it.

## What returns to GitHub

Commit only:

- `results/metrics.csv`
- `results/*.json` (small)
- `docs/*.svg`
- compressed demo media under `assets/` if small

Never commit `.pth`, bags, raw mp4, or datasets.
