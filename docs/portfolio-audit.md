# Portfolio Readiness Audit

**Audit date:** 2026-08-13  
**Scope:** `portfolio-observability-mvp` rebased with the current `main` safety and validation work.  
**Audience:** Robotics, software, infrastructure, and research-engineering reviewers.

## Executive Assessment

This repository is **portfolio-ready as a simulation-first robotics systems project**. It presents a credible engineering narrative: an initial vision-teleoperation prototype was reorganized into a reproducible ROS system with conservative safety defaults, documented proof boundaries, testable numerical methods, containerized validation, and optional non-blocking operational telemetry.

The project is deliberately accurate about its limits. Public artifacts demonstrate deterministic mock-input, RViz fake-hardware, and pure-Python simulation behavior; they do **not** demonstrate a calibrated camera, a physical Franka Panda, certified safety, or production robot control. This distinction is maintained in the [README](../README.md), [safety guide](safety.md), [calibration status](calibration.md), and [claims ledger](research/claims-ledger.md).

| Assessment area | Status | Evidence |
|---|---|---|
| Problem framing and technical narrative | Ready | [README](../README.md), [case study](case-study-v1.1.md) |
| Safety-by-default behavior | Ready for simulation scope | [safety guide](safety.md), [launch configuration](../src/vision_arm_control/launch/simulation.launch) |
| Reproducibility and developer experience | Ready | [Makefile](../Makefile), [Docker Compose](../docker-compose.yml), [operator runbook](operator-runbook.md) |
| Automated validation | Ready for declared CI scope | [CI workflow](../.github/workflows/ci.yml), [full-validation workflow](../.github/workflows/full-validation.yml) |
| Numerical and simulation evidence | Ready with explicit scope limits | [QP audit](research/qp-correctness-audit.md), [simulation validation](research/simulation-validation-results.md) |
| Observability | Ready as an opt-in portfolio MVP | [telemetry guide](telemetry.md), [schema](../sql/telemetry_schema.sql) |
| Physical-robot readiness | Not claimed | [limitations](../README.md#limitations), [real-robot safety requirements](safety.md) |

## Audit Verification

The following checks were executed against the reconciled branch during this audit. The host environment used a compatible modern Python toolchain for pure-Python validation; the repository's pinned ROS Noetic container workflow remains the canonical environment for ROS launch verification.

| Check | Result | Interpretation |
|---|---:|---|
| `python3 -m pytest -q tests/` | **168 passed** | Pure-Python mapping, safety, configuration, benchmark-honesty, QP, and telemetry checks passed. |
| `python3 scripts/validate_repository.py` | **41 README links validated** | Portfolio-facing local documentation links are intact. |
| `python3 scripts/validate_config.py` | **5 fixtures validated** | Versioned configuration fixtures meet the declared schema. |
| Pinned Black and Flake8 checks | **Pass** | Maintained Python surfaces conform to the versions specified in CI. |
| ROS Noetic container smoke test | **Not run locally** | The audit host did not have a container runtime; this check is retained in [CI](../.github/workflows/ci.yml). |

## Remediation Completed

The audit identified one configuration-integrity issue in the telemetry writer: fractional positive queue or batch values could be coerced to zero, which would violate the bounded-queue design because Python's queue treats `maxsize=0` as unbounded. The writer now normalizes parameters before validation and rejects non-positive integral capacity, non-finite or non-positive flush intervals, and negative retry counts. Regression coverage verifies these invalid configurations in [`tests/test_telemetry.py`](../tests/test_telemetry.py).

> The telemetry path remains outside the control loop. The safety callback emits a local ROS message; a bounded background worker performs logging or optional network delivery. Queue drops and write failures are counted rather than delaying a safety decision.

## Reviewer Walkthrough

A reviewer can establish the project quickly with `make help`, inspect the architecture diagram, then run `make ci` for the local quality gate. The strongest evidence trail is the path from the [system overview](../README.md) to the [claims ledger](research/claims-ledger.md), benchmark artifacts under [`results/`](../results/), and the CI definitions that reproduce their smoke-level checks.

For a demonstration-oriented review, start with the [mock-landmark RViz recording](media/franka-rviz-dry-run.gif), then read its [provenance and reproduction notes](media/README.md). For an operational review, use the [operator runbook](operator-runbook.md) and the [telemetry guide](telemetry.md). Neither route requires or implies physical robot access.

## Remaining Work Before Physical Deployment

The next engineering phase is not feature expansion; it is **physical validation and safety integration**. It requires measured camera intrinsics and camera-to-base calibration, controller-level verification on a real Franka, collision-aware planning validation, end-to-end latency and tracking-error measurements, a physical E-stop and dead-man procedure, and an approved lab safety plan. Those activities must be performed under appropriate hardware, institutional, and manufacturer controls rather than inferred from these simulation artifacts.

## Portfolio Positioning

This project is best presented as a **robotics systems engineering case study** rather than as a claim of production-ready teleoperation. It provides concrete evidence of interface design, safety gating, numerical verification, test design, reproducibility, CI, containerization, observability, failure-mode documentation, and honest scientific communication. The project’s most compelling feature is that its proof boundary is explicit and reproducible.
