#!/bin/bash
#SBATCH --job-name=franka-perception
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

# Purdue Scholar GPU perception job template.
# Do not run GPU work on the login/frontend node.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
DATA_ROOT="${DATA_ROOT:-${RCAC_SCRATCH}/franka-teleop-data}"
export FRANKA_MODEL_DIR="${FRANKA_MODEL_DIR:-$DATA_ROOT/models}"
OUT_DIR="${OUT_DIR:-$DATA_ROOT/outputs/perception_${SLURM_JOB_ID}}"
mkdir -p "$OUT_DIR" "$DATA_ROOT/logs"

echo "host=$(hostname)"
echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "REPO_DIR=$REPO_DIR"
echo "FRANKA_MODEL_DIR=$FRANKA_MODEL_DIR"
echo "OUT_DIR=$OUT_DIR"
nvidia-smi || true

module load cuda 2>/dev/null || true

cd "$REPO_DIR"
python3 benchmarks/benchmark_perception.py | tee "$OUT_DIR/perception_benchmark.json.stdout"
# Copy lightweight metrics only (not weights)
cp -f results/perception_benchmark.json "$OUT_DIR/" 2>/dev/null || true

echo "Done. Lightweight results in $OUT_DIR"
