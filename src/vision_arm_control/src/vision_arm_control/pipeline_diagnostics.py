"""Structured pipeline diagnostics + metrics (pure Python, no ROS).

Shared by the ROS 2 teleop pipeline node (which publishes/logs these records)
and by the offline metrics extractor (`scripts/extract_run_metrics.py`). Keeping
this logic ROS-free means it is unit-tested in CI and the ROS node stays a thin
publisher shell.

`DiagnosticsBuilder.update(...)` returns one JSON-serializable record per step
covering the observability fields required for the simulation demo:
pipeline state, strategy, safety mode, source timestamp, observation age,
confidence, retargeter status, gate state, solver status, active constraints,
intervention magnitude, backend result, latency, dropped-command count, last
stop reason, and — on any mode change — an explicit, non-silent transition
record (previous mode, new mode, reason, timestamp, command sent, holding,
stopped).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class DiagnosticsBuilder:
    strategy: str
    safety_mode: str
    _prev_state: Optional[str] = None
    _dropped: int = 0
    _last_stop_reason: str = ""

    def update(self, step: Any, gate_status: str, now: float) -> Dict[str, Any]:
        """Build a diagnostics record for one pipeline step.

        ``step`` is a ``PipelineStepResult``; ``gate_status`` is the gate state
        string (e.g. "pass"/"hold"/"stopped"/"recovering").
        """
        safety = step.safety
        backend = step.backend
        retargeted = step.retargeted
        gated = step.gated

        if step.commanded:
            state = "commanded"
        elif gate_status == "hold":
            state = "holding"
        elif gate_status == "recovering":
            state = "recovering"
        else:
            state = "stopped"

        if not step.commanded:
            self._dropped += 1
        if state == "stopped":
            self._last_stop_reason = getattr(gated, "reason", "") or self._last_stop_reason

        source_ts = float(getattr(retargeted, "timestamp", now))
        rec: Dict[str, Any] = {
            "t": float(now),
            "pipeline_state": state,
            "strategy": self.strategy,
            "safety_mode": self.safety_mode,
            "source_timestamp": source_ts,
            "observation_age_s": float(now) - source_ts,
            "confidence": float(getattr(gated, "confidence", 0.0)),
            "retargeter_status": getattr(getattr(retargeted, "status", None), "value", "n/a"),
            "gate_state": gate_status,
            "solver_status": getattr(safety, "solver_status", "n/a") if safety else "n/a",
            "active_constraints": list(getattr(safety, "active_constraints", []) or []) if safety else [],
            "intervention_magnitude": float(getattr(safety, "intervention_magnitude", 0.0))
            if safety
            else 0.0,
            "backend_accepted": (None if backend is None else bool(backend.accepted)),
            "latency_ms": (float(safety.compute_time_s) * 1000.0 if safety else 0.0),
            "dropped_command_count": self._dropped,
            "last_stop_reason": self._last_stop_reason,
            "commanded": bool(step.commanded),
        }
        if state != self._prev_state:
            rec["mode_transition"] = {
                "previous_mode": self._prev_state,
                "new_mode": state,
                "reason": getattr(gated, "reason", "") or getattr(safety, "reason", "") if safety else "",
                "timestamp": float(now),
                "command_sent": bool(step.commanded),
                "holding": state == "holding",
                "stopped": state == "stopped",
            }
        self._prev_state = state
        return rec


def summarize_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate a list of diagnostics records into run-level metrics."""
    n = len(records)
    if n == 0:
        return {"n_records": 0}
    ts = [float(r.get("t", 0.0)) for r in records]
    elapsed = max(ts) - min(ts) if n > 1 else 0.0
    commanded = sum(1 for r in records if r.get("commanded"))
    states: Dict[str, int] = {}
    solver: Dict[str, int] = {}
    interventions = 0
    interv_mags: List[float] = []
    latencies: List[float] = []
    transitions = 0
    for r in records:
        states[r.get("pipeline_state", "?")] = states.get(r.get("pipeline_state", "?"), 0) + 1
        s = r.get("solver_status", "n/a")
        solver[s] = solver.get(s, 0) + 1
        if float(r.get("intervention_magnitude", 0.0)) > 1e-9 or r.get("active_constraints"):
            interventions += 1
            interv_mags.append(float(r.get("intervention_magnitude", 0.0)))
        latencies.append(float(r.get("latency_ms", 0.0)))
        if "mode_transition" in r:
            transitions += 1
    solver_failures = sum(v for k, v in solver.items() if "solver_failure" in k or k == "error")
    return {
        "n_records": n,
        "elapsed_s": elapsed,
        "command_rate_hz": (commanded / elapsed if elapsed > 0 else float("nan")),
        "acceptance_rate": commanded / n,
        "dropped_rate": (n - commanded) / n,
        "state_counts": states,
        "solver_status_counts": solver,
        "solver_failures": solver_failures,
        "intervention_pct": 100.0 * interventions / n,
        "mean_intervention_magnitude": (sum(interv_mags) / len(interv_mags) if interv_mags else 0.0),
        "mean_latency_ms": (sum(latencies) / len(latencies) if latencies else 0.0),
        "mode_transitions": transitions,
        "final_dropped_command_count": int(records[-1].get("dropped_command_count", 0)),
    }


@dataclass
class DiagnosticsLog:
    """Accumulates records and can serialize to JSONL / a summary dict."""

    records: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, record: Dict[str, Any]) -> None:
        self.records.append(record)

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(r, sort_keys=True) for r in self.records) + ("\n" if self.records else "")

    def summary(self) -> Dict[str, Any]:
        return summarize_records(self.records)
