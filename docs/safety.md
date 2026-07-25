# Safety

Software safeguards in this repository are **assistive only**. They do not replace:

- Franka Desk safety configuration
- Hardware emergency stop
- FCI reflex / collision behavior
- Trained operator procedures
- Laboratory risk assessment

## Defaults

| Setting | Default | Meaning |
|---------|---------|---------|
| `controller.mode` | `dry_run` | No hardware execution |
| `controller.use_robot` | `false` | Robot disabled |
| `safety.allow_real_robot` | `false` | Real mode blocked in config |
| Launch `use_robot` | `false` | Explicit opt-in required |

## Implemented software controls

- Cartesian workspace bounds (`config/safety.yaml`)
- Optional clamp vs reject modes
- Approximate joint-limit checks (Panda numeric limits)
- Cartesian velocity limiting (command shaping)
- Max command rate
- Pose-loss timeout
- Command staleness monitor
- Emergency-stop topic (`~/emergency_stop`)
- Dead-man enable topic (`~/deadman`) for real-robot mode

## Real robot checklist

1. Confirm reduced speed and force limits in Desk.
2. Clear workspace; verify E-stop in hand.
3. Run simulation / dry_run first.
4. Launch with `use_robot:=true` only when intentional.
5. Hold dead-man enable; release stops motion policy.
6. Assert emergency stop topic on any anomaly.

## What is not claimed

- Formal ISO safety certification
- Guaranteed collision-free motion without MoveIt planning success
- Replacement for manufacturer safety systems
