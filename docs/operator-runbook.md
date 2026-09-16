# Operator Runbook

## Fast path

```bash
make help
make test
make sim
```

`make sim` builds the ROS Noetic container and runs the headless mock-landmark simulation smoke test. It does not connect to a physical Franka.

## Common failures

### Docker cannot reach Ubuntu or ROS package mirrors

Confirm normal network/DNS access, then retry `make docker-build`. Do not weaken certificate validation or add untrusted package mirrors.

### `docker: command not found`

Install Docker Desktop or Docker Engine. The pure-Python gate still works with:

```bash
python3 -m pip install -r requirements-ci.txt
make check
```

### `rospack find vision_arm_control` fails

Run inside the project image, or source both environments in a configured catkin workspace:

```bash
source /opt/ros/noetic/setup.bash
source /ws/devel/setup.bash
rospack find vision_arm_control
```

### Simulation times out or exits early

The local smoke gate requires three messages on `/robot_controller_node/executed_pose` within 30 seconds. Timeout is a failure, even if roslaunch remained alive. The script prints launch and topic logs on failure and stops its launch on exit. Use an isolated ROS master; an existing master is rejected to prevent unrelated nodes from satisfying the gate.

### Model weights are missing

The default mock-landmark simulation does not need model weights. For perception paths, set `FRANKA_MODEL_DIR` and follow [`model-setup.md`](model-setup.md). Never commit weights to this repository.

### Telemetry shows logs but no Supabase rows

Check all of the following:

```bash
echo "$TELEMETRY_ENABLED"        # true
echo "$TELEMETRY_DRY_RUN"        # false
echo "$SUPABASE_URL"             # project URL
test -n "$SUPABASE_TELEMETRY_KEY" && echo "key is set"
```

Then confirm the SQL schema exists. HTTP `401` or `403` means the runtime key or database policy is wrong. Do not print the key itself.

### Telemetry queue drops events

A growing `dropped_events` heartbeat value means the network sink is slower than the event rate. Keep the control path non-blocking. Increase batching first; increase queue size only after measuring memory and outage behavior.

### Full validation is too slow for every edit

Use `make check` during development. Run `make full-validation` before publishing new benchmark claims or replacing committed evidence.
