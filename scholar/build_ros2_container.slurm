#!/bin/bash
#SBATCH --job-name=ros2-sif-build
#SBATCH --account=scholar
#SBATCH --partition=scholar-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/scholar/edraymon/franka-teleop-data/logs/ros2-sif-build-%j.out
#SBATCH --error=/scratch/scholar/edraymon/franka-teleop-data/logs/ros2-sif-build-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/scholar/edraymon/franka-teleoperation}"
DATA="${RCAC_SCRATCH:-/scratch/scholar/edraymon}/franka-teleop-data"
CTR_DIR="$DATA/containers"
CACHE="$DATA/ros2-cache"
mkdir -p "$CTR_DIR" "$CACHE" "$DATA/logs"
export APPTAINER_CACHEDIR="$CACHE/apptainer"
export APPTAINER_TMPDIR="$CACHE/tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

SIF="$CTR_DIR/ros2_humble_franka.sif"
DEF="$REPO_DIR/scholar/ros2_humble.def"

echo "host=$(hostname) job=$SLURM_JOB_ID date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "building $SIF from $DEF"
apptainer --version

if [[ -f "$SIF" ]]; then
  echo "Existing SIF found: $SIF"
  ls -lh "$SIF"
  # Rebuild only if FORCE_REBUILD=1
  if [[ "${FORCE_REBUILD:-0}" != "1" ]]; then
    echo "Skipping rebuild (set FORCE_REBUILD=1 to rebuild)"
    apptainer exec "$SIF" bash -lc 'source /opt/ros/humble/setup.bash && ros2 --help | head -3 && cat /etc/ros2_pkg_smoke.txt | head'
    exit 0
  fi
  rm -f "$SIF"
fi

# Strategy 1: build from definition (pulls docker base + apt packages)
cd "$CTR_DIR"
apptainer build --force "$SIF" "$DEF"

ls -lh "$SIF"
apptainer exec "$SIF" bash -lc 'source /opt/ros/humble/setup.bash; ros2 pkg prefix moveit_resources_panda_moveit_config; ros2 pkg list | grep -E "moveit_resources_panda|controller_manager|rviz2" | head'
echo "BUILD_OK sif=$SIF"
