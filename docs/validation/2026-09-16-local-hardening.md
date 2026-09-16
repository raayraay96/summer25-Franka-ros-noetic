# Local validation and timestamp hardening — 2026-09-16

Base: `ef1c70b` (`main`). Execution environment: Linux x86_64, Python 3.10.21, dependencies from `requirements-ci.txt` in an isolated virtual environment.

## Changes

- Removed both GitHub Actions workflows and recorded the no-Actions instruction in `AGENTS.md`.
- Added `make check`, including the existing ROS 2 pure-logic suite. `make ci` is a compatibility alias.
- Fixed the ROS1 mapper and safety node treating callback receipt time as acquisition time. Old poses could previously be accepted and restamped as fresh. Acquisition stamps now survive both stages, and future/non-finite timestamps fail the shared pose-timeout check.
- Changed the Noetic smoke gate from accepting a roslaunch timeout to requiring three controller output messages. It rejects an existing master, waits for its own master to start, prints failure logs, and cleans up its launch. Core simulation nodes are required.
- Limited repository hygiene checks to Git-visible source so local virtual environments do not break validation.

## Verification

Command: `make check PYTHON=/tmp/franka-validation-venv/bin/python`.

- 191 project tests passed, including real ROS callback code exercised with message/transport stand-ins and smoke-runner success/timeout/empty-output cases.
- 14 ROS 2 pure-logic tests passed.
- Black and Flake8 passed, including both changed ROS1 node scripts.
- 41 README links and five config fixtures validated.
- 800 QP problems: zero feasibility, objective, solution-distance, or KKT mismatches against the trusted oracle.
- Six-row benchmark smoke completed across two combinations and three trajectories; zero solver failures.
- Shell syntax and Git whitespace checks passed.

Timestamp regression tests failed before the fix: stale and future poses were forwarded, and valid stamps were overwritten. They passed after the fix.

## Unverified boundaries

This host has neither Docker nor ROS installed. The updated live ROS integration gate and catkin/container build were **not run here**. The shell tests exercise the gate with command stand-ins; they are not ROS integration evidence. Run `make sim` on a Docker-equipped host or follow `docs/local-validation.md` on a configured Noetic host.

No physical robot, live camera, calibrated mapping, Gazebo execution, or MoveIt hardware execution was tested. The full 6,000-case numerical audit and 30-replicate benchmark were not rerun. Existing committed historical claims remain distinct from this run.

The project remains a simulation-backed portfolio prototype; this change does not establish physical end-to-end readiness.
