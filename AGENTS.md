# Project instructions

- Do not use GitHub Actions for this project. Do not add workflows, dispatch runs, or require Actions checks. Run validation locally or on a configured Scholar host.
- Use `make check` for the local Python quality gate and `make sim` for the ROS Noetic container smoke gate.
- Preserve dry-run defaults and explicit real-robot opt-in. Simulation evidence is not physical-hardware validation.
- Preserve acquisition timestamps from landmarks through mapped and accepted poses. Never make stale input appear fresh by stamping it at receipt.
- Report checks actually run and environment blockers. Do not claim an end-to-end ROS run from static checks or Python test doubles.
