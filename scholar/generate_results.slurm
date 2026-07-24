#!/bin/bash
#SBATCH --job-name=franka-results
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
DATA_ROOT="${DATA_ROOT:-${RCAC_SCRATCH}/franka-teleop-data}"
OUT_DIR="${OUT_DIR:-$DATA_ROOT/outputs/results_${SLURM_JOB_ID}}"
mkdir -p "$OUT_DIR"

cd "$REPO_DIR"
python3 scripts/generate_diagrams.py
python3 benchmarks/benchmark_mapping.py
python3 benchmarks/benchmark_latency.py

# Environment metadata for reproducibility
python3 - <<PY
import json, platform, os
from datetime import datetime, timezone
meta = {
  "hostname": platform.node(),
  "platform": platform.platform(),
  "python": platform.python_version(),
  "date_utc": datetime.now(timezone.utc).isoformat(),
  "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}
open("results/environment.json","w").write(json.dumps(meta, indent=2)+"\n")
print(meta)
PY

cp -f docs/*.svg "$OUT_DIR/" 2>/dev/null || true
cp -f results/* "$OUT_DIR/" 2>/dev/null || true
echo "Copy only lightweight plots/metrics back into the git repo for portfolio commits."
echo "OUT_DIR=$OUT_DIR"
