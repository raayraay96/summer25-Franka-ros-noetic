#!/bin/bash
#SBATCH --job-name=franka-bench
# Optional: #SBATCH --account=scholar   # set per site associations
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
# GPU optional — mapping benchmarks are CPU-only
# #SBATCH --gres=gpu:1


set -euo pipefail
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
DATA_ROOT="${DATA_ROOT:-${RCAC_SCRATCH}/franka-teleop-data}"
OUT_DIR="${OUT_DIR:-$DATA_ROOT/benchmarks/run_${SLURM_JOB_ID}}"
mkdir -p "$OUT_DIR"

cd "$REPO_DIR"
python3 benchmarks/benchmark_mapping.py
python3 benchmarks/benchmark_latency.py
python3 benchmarks/benchmark_perception.py || true

cp -f results/*.json "$OUT_DIR/" 2>/dev/null || true
# Optional: generate CSV aggregate for GitHub commit
python3 - <<'PY'
import json, csv
from pathlib import Path
rows = []
for p in Path("results").glob("*.json"):
    data = json.loads(p.read_text())
    rows.append({
        "file": p.name,
        "metric": data.get("metric"),
        "mean_ms": data.get("mean_ms", data.get("status")),
        "date_utc": data.get("date_utc"),
        "notes": data.get("notes", ""),
    })
out = Path("results/metrics.csv")
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["file", "metric", "mean_ms", "date_utc", "notes"])
    w.writeheader()
    w.writerows(rows)
print("wrote", out)
PY
cp -f results/metrics.csv "$OUT_DIR/" 2>/dev/null || true
echo "Benchmarks complete → $OUT_DIR"
