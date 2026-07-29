# Baseline State — research-retargeting-v1.1

**Recorded before any v1.1 refactor.**  
**Date (UTC):** 2026-07-28  
**Host:** Purdue Scholar frontend (`Linux-5.14.0-570.26.1.el9_6.x86_64`)

---

## Branch and commit

| Item | Value |
|------|-------|
| Base branch | `main` (tracking `origin/main`) |
| Base commit SHA | `f305ccdf8565f2cc48b5975ce06cd1c30f84fbab` |
| Working branch | `research-retargeting-v1.1` (created from base) |
| Tag preserved | `v1.0-portfolio` @ `a9b00b6870d9717ecc6d45704f4ef3aa7eb95a7e` |
| History rewrite | None |

Note: local `main` had diverged from `origin/main`; the research branch was created from **`origin/main`** (`f305ccd`) after a hard reset to remote, which is the current published portfolio main.

---

## Commands run

```bash
git fetch origin main
git reset --hard origin/main
git checkout -b research-retargeting-v1.1

export PYTHONPATH="src/vision_arm_control/src:${PYTHONPATH:-}"
python3 -m pytest -q tests/
python3 scripts/validate_repository.py
python3 benchmarks/benchmark_mapping.py
python3 benchmarks/benchmark_latency.py
```

---

## Existing test count

| Suite | Result |
|-------|--------|
| Pure-logic `tests/` | **80 passed** in 0.44 s |
| Repository validation | **OK** (`validated 42 README links`) |
| ROS 2 package tests | Not re-run on this host (no ROS 2 module path required for pure suite) |
| ROS Noetic container | **Not run** on this host (Docker unavailable) |

---

## Existing benchmark results (this host)

### Mapping (`benchmarks/benchmark_mapping.py` → `results/mapping_benchmark.json`)

| Metric | Value |
|--------|------:|
| n | 5000 |
| mean_ms | 0.005385 |
| p50_ms | 0.005301 |
| p95_ms | 0.005546 |
| max_ms | 0.029952 |
| hardware | x86_64 |
| python | 3.9.21 |

### Pipeline CPU stages (`benchmarks/benchmark_latency.py` → `results/latency_benchmark.json`)

| Metric | Value |
|--------|------:|
| n | 2000 |
| mean_ms | 0.019139 |
| p50_ms | 0.019165 |
| reject_rate | 0.0 |
| notes | map + filter + workspace only; not full ROS/perception |

---

## Available ROS environments

| Environment | Available on this host? |
|-------------|-------------------------|
| Python 3.9 + pure-logic package path | Yes |
| ROS Noetic (`roscore`) | **No** |
| Docker / Noetic container build | **No** (`docker` not in PATH) |
| ROS 2 Humble | Not checked for interactive use; Scholar SLURM path documented historically |
| Physical Franka | **Unavailable** |
| MoveIt / Gazebo live run | **Not executed** for this baseline |

---

## Unavailable hardware or dependencies

- Physical Franka Emika Panda
- Calibrated metric depth camera
- Live ROS master on the Scholar frontend
- Docker daemon for local Noetic container reproduction
- OSQP / CVXPY (not installed; not required by baseline)

---

## Files expected to change in v1.1

- New pure-Python packages under `src/vision_arm_control/src/vision_arm_control/{retargeting,safety_filters,landmarks,backends}/`
- Config YAML (`mapping.yaml`, `safety.yaml`, new retargeting/safety flags)
- Mapper / safety node wiring for strategy selection
- `tests/` additions for SEW geometry, CBF-QP, confidence gating
- `benchmarks/benchmark_retargeting_v11.py` + config
- `results/v1.1/` measured outputs
- `docs/research/*`, `docs/case-study-v1.1.md`, `docs/media/v1.1/`
- README compact v1.1 section
- CI workflow extensions
- Optional ROS 2 thin adapters for selectable modes

---

## Risks

1. **Interface refactor** may break launch defaults if not carefully gated behind YAML flags.
2. **SEW closed-form 7-DoF joint solve** requires verified IK; if not reliable, output stays at target-feature level only.
3. **CBF-QP** without OSQP needs a carefully tested low-dimensional solver; solver failure must stop, not pass-through.
4. **CI Noetic job** cannot be validated on this host without Docker; rely on GitHub Actions.
5. **Demo media** without RViz on this host must be labeled as pure-Python / synthetic visualization if RViz is not re-run.
6. **Paper over-claim risk** — every method must be labeled “inspired by” unless mathematically reproduced.

---

*End of baseline record. Implementation proceeds only after this file is committed.*
