# Experimental Cartesian CBF-QP Safety Filter (v1.1)

**Status:** Experimental kinematic filter  
**Inspired by:** CBF-QP safety layers for imitation (arXiv:2604.11447)  
**Module:** `vision_arm_control.safety_filters.cbf_qp`

## Scope (modeled exactly)

- Decision variable: Cartesian velocity \(u \in \mathbb{R}^3\) (or increment \(u\,\Delta t\))
- Axis-aligned workspace halfspace CBFs with margin
- Configurable spherical obstacles
- Box velocity limits
- Objective: minimize \(\|u - u_{\mathrm{nom}}\|^2\)

Discrete CBF inequality used:

\[
\nabla h(p)^\top u + \alpha h(p) \ge 0
\]

## Solver

Pure-NumPy active-set style QP for 3 variables (no OSQP/CVXPY required for Noetic container stability). Solver failure **rejects** the command — never pass-through.

## Non-claims

- Not manufacturer-certified safety
- Not dynamics-level CBF / full inertia model
- No formal forward-invariance certificate beyond the exact discrete model
- Not a replacement for Franka Desk / FCI / E-stop

## Configuration example

```yaml
safety_filter:
  mode: cbf_qp
  dt: 0.05
  alpha: 4.0
  workspace_margin_m: 0.03
  max_cartesian_velocity_mps: 0.20
  stop_on_solver_failure: true
  obstacles:
    - id: demo_sphere
      center: [0.48, 0.00, 0.35]
      radius_m: 0.10
      margin_m: 0.05
```

## Baseline filters preserved

| Mode | Behavior |
|------|----------|
| `reject` | Hard reject outside workspace |
| `clamp` | Clamp into workspace |
| `cbf_qp` | Experimental filter above |

Default remains `reject` / dry-run.
