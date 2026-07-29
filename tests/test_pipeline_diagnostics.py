"""Tests for pure-Python pipeline diagnostics + metrics (no ROS)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vision_arm_control" / "src"))

from vision_arm_control.landmarks.confidence import ConfidenceGateConfig  # noqa: E402
from vision_arm_control.landmarks.schema import ArmLandmarks, LandmarkPoint  # noqa: E402
from vision_arm_control.pipeline import PipelineConfig, TeleopPipeline  # noqa: E402
from vision_arm_control.pipeline_diagnostics import (  # noqa: E402
    DiagnosticsBuilder,
    DiagnosticsLog,
    summarize_records,
)
from vision_arm_control.retargeting.base import RetargetingConfig  # noqa: E402
from vision_arm_control.safety_filters.base import SafetyFilterConfig  # noqa: E402


def _lm(t, wx=0.55):
    s = (0.45, 0.35, 0.0)
    return ArmLandmarks(
        shoulder=LandmarkPoint(*s, confidence=1.0, name="s"),
        elbow=LandmarkPoint(0.51, 0.40, 0.0, confidence=1.0, name="e"),
        wrist=LandmarkPoint(wx, 0.47, 0.0, confidence=1.0, name="w"),
        timestamp=t,
    )


def _pipeline():
    return TeleopPipeline(
        PipelineConfig(
            retargeting=RetargetingConfig(name="shoulder_relative"),
            safety=SafetyFilterConfig(mode="reject"),
            gate=ConfidenceGateConfig(),
            backend="dry_run",
        )
    )


def test_builder_produces_required_fields():
    pipe = _pipeline()
    b = DiagnosticsBuilder(strategy="shoulder_relative", safety_mode="reject")
    step = pipe.step(_lm(0.1), now=0.1)
    rec = b.update(step, pipe.gate.status.value, now=0.1)
    for key in [
        "pipeline_state",
        "strategy",
        "safety_mode",
        "source_timestamp",
        "observation_age_s",
        "confidence",
        "retargeter_status",
        "gate_state",
        "solver_status",
        "active_constraints",
        "intervention_magnitude",
        "backend_accepted",
        "latency_ms",
        "dropped_command_count",
        "last_stop_reason",
    ]:
        assert key in rec, key


def test_dropout_emits_stop_transition_no_silent_fallback():
    pipe = _pipeline()
    b = DiagnosticsBuilder(strategy="shoulder_relative", safety_mode="reject")
    log = DiagnosticsLog()
    # a few valid frames, then a long dropout (None) to force hold->stop
    t = 0.0
    for _ in range(3):
        step = pipe.step(_lm(t), now=t)
        log.add(b.update(step, pipe.gate.status.value, now=t))
        t += 0.05
    for _ in range(20):
        step = pipe.step(None, now=t)
        log.add(b.update(step, pipe.gate.status.value, now=t))
        t += 0.05
    states = {r["pipeline_state"] for r in log.records}
    assert "stopped" in states  # eventually stops, never silently continues
    # every mode change carried an explicit transition record
    transitions = [r["mode_transition"] for r in log.records if "mode_transition" in r]
    assert transitions
    assert all({"previous_mode", "new_mode", "reason", "timestamp"} <= set(tr) for tr in transitions)


def test_summarize_and_jsonl_roundtrip():
    pipe = _pipeline()
    b = DiagnosticsBuilder(strategy="shoulder_relative", safety_mode="reject")
    log = DiagnosticsLog()
    t = 0.0
    for _ in range(10):
        step = pipe.step(_lm(t, wx=0.55 + 0.01 * t), now=t)
        log.add(b.update(step, pipe.gate.status.value, now=t))
        t += 0.05
    summ = log.summary()
    assert summ["n_records"] == 10
    assert 0.0 <= summ["acceptance_rate"] <= 1.0
    assert "state_counts" in summ and "solver_status_counts" in summ
    # JSONL serializes and every line is valid json
    lines = [ln for ln in log.to_jsonl().splitlines() if ln.strip()]
    assert len(lines) == 10
    for ln in lines:
        json.loads(ln)


def test_summarize_empty():
    assert summarize_records([])["n_records"] == 0
