"""Pure-Python tests for the non-blocking telemetry path."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vision_arm_control" / "src"))

from vision_arm_control.telemetry import AsyncTelemetryClient, TelemetryEvent  # noqa: E402


class RecordingSink:
    def __init__(self):
        self.records = []

    def write(self, records):
        self.records.extend(records)


def test_event_omits_none_fields():
    record = TelemetryEvent(event_type="node_heartbeat", run_id="run-1", source="test").to_record()
    assert record["event_type"] == "node_heartbeat"
    assert "latency_ms" not in record
    assert record["metadata"] == {}


def test_async_client_flushes_batch():
    sink = RecordingSink()
    client = AsyncTelemetryClient(sink, batch_size=10, flush_interval_sec=0.01)
    client.start()
    assert client.submit(
        TelemetryEvent(
            event_type="safety_decision",
            run_id="run-1",
            source="test",
            accepted=False,
            safety_reason="pose_timeout",
            latency_ms=0.25,
        )
    )
    deadline = time.time() + 1.0
    while not sink.records and time.time() < deadline:
        time.sleep(0.01)
    client.stop()
    assert sink.records[0]["safety_reason"] == "pose_timeout"
    assert client.written_events == 1


def test_submit_never_blocks_when_queue_is_full():
    sink = RecordingSink()
    client = AsyncTelemetryClient(sink, queue_size=1)
    event = TelemetryEvent(event_type="test", run_id="run-1", source="test")
    assert client.submit(event)
    assert not client.submit(event)
    assert client.dropped_events == 1
