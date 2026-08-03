# Telemetry and Supabase

## Purpose

The telemetry path demonstrates non-blocking observability around the safety gate. It does not control the robot and is disabled by default.

```text
safety_monitor_node
  └─ /safety_monitor_node/telemetry (JSON/String)
       └─ telemetry_node
            └─ bounded in-memory queue
                 └─ background batch writer
                      └─ Supabase PostgREST -> Postgres
```

The safety callback only publishes a local ROS message. Network writes happen on a separate bounded worker. If the queue is full, the event is dropped and counted rather than blocking the control loop.

## Metrics

- `safety_decision`: accepted/rejected decision, reason, and safety-gate evaluation latency
- `estop_trigger` and `estop_clear`
- `pose_timeout` is represented as a rejected `safety_decision`
- `node_heartbeat`: queue depth, dropped events, failed events, and successful writes

## Setup

1. Apply [`sql/telemetry_schema.sql`](../sql/telemetry_schema.sql) in Supabase.
2. Copy `.env.example` to `.env` and inject real values outside Git.
3. Start the simulation with telemetry explicitly enabled:

```bash
set -a
source .env
set +a
roslaunch vision_arm_control simulation.launch enable_telemetry:=true telemetry_dry_run:=false
```

For a local proof without network writes:

```bash
roslaunch vision_arm_control simulation.launch enable_telemetry:=true telemetry_dry_run:=true
```

## Security boundary

Never commit a service-role key, expose it in frontend code, bake it into a Docker image, or add it to GitHub Actions. The direct PostgREST writer is acceptable only on a trusted edge host for this portfolio MVP. A production deployment should put an authenticated Supabase Edge Function or a dedicated least-privilege ingestion service between the robot network and Postgres.

## Verify inserts

```sql
select
  run_id,
  event_type,
  safety_reason,
  accepted,
  round(latency_ms::numeric, 3) as latency_ms,
  observed_at
from public.franka_telemetry
order by observed_at desc
limit 20;
```
