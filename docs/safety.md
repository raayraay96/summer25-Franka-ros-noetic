# Safety

> **Default off. Software limits are not a substitute for Franka Desk, a physical E-stop, a dead-man control, trained supervision, or laboratory procedures.**

Software safeguards in this repository are assistive only. They do not replace manufacturer safety systems, FCI reflex behavior, a lab risk assessment, or operator training.

## Safe defaults

| Setting | Default | Effect |
|---|---|---|
| `controller.mode` | `dry_run` | Targets are logged; no hardware command is sent |
| `controller.use_robot` | `false` | Robot execution disabled |
| `safety.allow_real_robot` | `false` | Real mode blocked in configuration |
| launch `use_robot` | `false` | Explicit opt-in required |
| `require_deadman` | `true` in real mode | Motion authority must be continuously enabled |

## Implemented software controls

- Cartesian workspace reject/clamp policy
- Cartesian velocity limiting and command-rate limiting
- Approximate Panda joint-limit checks in simulation utilities
- Pose-loss timeout and stale-command rejection
- Emergency-stop topic (`~/emergency_stop`)
- Dead-man topic (`~/deadman`)
- Real-robot configuration and launch opt-in gates

## Physical Franka requirements

Before any hardware run:

1. Configure reduced speed, force, and collision thresholds in **Franka Desk**.
2. Verify the **physical E-stop** before enabling motion and keep it reachable.
3. Use a tested **dead-man control** that removes motion authority when released.
4. Clear the workspace and complete the lab's written risk assessment and preflight checklist.
5. Run unit tests, `dry_run`, and simulation with the exact configuration first.
6. Confirm measured camera calibration and transforms; do not use example values.
7. Start at reduced speed with a trained spotter and preserve logs for review.

## Not claimed

- ISO or regulatory safety certification
- Guaranteed collision avoidance without a successful planning scene and controller validation
- Physical Franka validation in CI
- Replacement for manufacturer or laboratory safety systems

Runtime sources of truth: `src/vision_arm_control/config/safety.yaml` and `controller.yaml`.
See also: [calibration](calibration.md) · [model setup](model-setup.md) · [repository audit](repository-audit.md).
