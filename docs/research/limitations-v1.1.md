# Limitations — v1.1

## Research integrity

- SEW-Mimic is **inspired**, not reproduced (no closed-form 7-DoF joint solver).
- CBF-QP is **experimental kinematic** filtering only — no formal certificate.
- AnyTeleop / shared-control papers informed interfaces only.

## System

- Defaults remain `shoulder_relative` + `reject` + `dry_run`.
- No metric monocular depth for control.
- No physical Franka validation of v1.1 methods.
- Joint IK layer for SEW features is **explicitly missing**.
- Demo media for v1.1 is **pure-Python matplotlib simulation** of targets, not a re-recorded RViz/Gazebo stack (unless separately produced).
- Docker / ROS Noetic container was not rebuilt on the Scholar frontend for this session (`docker` unavailable); CI is the authority for container green status.
- CBF-QP high intervention rate reflects tight velocity/barrier modeling, not certified safety performance.
- Benchmarks are host-specific; do not compare numbers across machines without naming each host.

## Safety non-claims

- Not ISO / manufacturer certified  
- Not collision-free physical execution guarantee  
- Not a substitute for Desk, FCI, or E-stop  
