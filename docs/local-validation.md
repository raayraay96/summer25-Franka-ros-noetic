# Local validation

GitHub Actions is not used for this project. All commands run on the developer's machine, in Docker, or on a configured Scholar host.

## Python gate (Mac or Linux, no ROS required)

Use Python 3.10 for the pinned dependency set. The existing pins do not support Python 3.12+.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
make install
make check
```

`make check` fails on formatting, unit tests, ROS 2 pure-logic tests, invalid repository/config files, QP oracle mismatches, or benchmark failures. The QP and benchmark reports go to `results/v1.1-local/`. `make ci` remains a local compatibility alias; it does not call GitHub.

The repository validator honors `.gitignore`, so `.venv/` and generated results do not get mistaken for committed source or model weights.

## ROS Noetic integration gate

With Docker available:

```bash
make sim
```

This builds the existing Linux ROS Noetic image, runs its unit tests and catkin build, then starts mock landmarks → mapper → safety → dry-run controller. Success requires three messages at the controller output within 30 seconds and a running launch process. A timeout, early exit, or empty output fails. Required pipeline nodes terminate the launch on failure. No physical robot is commanded.

On a host with Noetic and a built catkin workspace:

```bash
ROS_SETUP_FILE=/opt/ros/noetic/setup.bash \
CATKIN_SETUP_FILE="$HOME/catkin_ws/devel/setup.bash" \
bash scripts/ci_smoke_noetic.sh
```

Run with an unused ROS master. The script refuses an existing master so another running demo cannot produce a false success. ROS setup paths are configurable; the script retains its old filename for existing callers.

## Full numerical evidence

```bash
make full-validation
bash scripts/audit_git_history.sh
```

Full validation regenerates the committed numerical evidence paths; review those diffs before committing. The history audit fetches Git refs and checks object sizes/model artifacts. Neither command uses Actions.

## Timestamp contract

Landmark JSON must carry a numeric `header.stamp` in ROS seconds, using the same clock as the nodes. Missing, malformed, non-finite, future, or expired acquisition times are rejected. The mapper preserves the acquisition timestamp; the safety node checks that timestamp and preserves it in the accepted command. Rosbag replay must use compatible ROS time, rather than replaying old stamps against the current wall clock.

These gates do not establish physical tracking, robot dynamics safety, camera calibration, or camera-to-MoveIt integration. Verify those separately in an authorized lab environment.
