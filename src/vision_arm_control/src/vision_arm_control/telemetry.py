"""Non-blocking telemetry primitives for ROS edge nodes.

The control path only calls ``submit``. Network I/O runs on a bounded background
worker so telemetry cannot block a ROS callback or grow memory without limit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import math
from queue import Empty, Full, Queue
from threading import Event, Thread
import time
from typing import Any, Dict, Iterable, List, Optional
from urllib.request import Request, urlopen

LOGGER = logging.getLogger(__name__)


def utc_now_iso() -> str:
    """Return a Supabase/Postgres-friendly UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TelemetryEvent:
    """One normalized telemetry record."""

    event_type: str
    run_id: str
    source: str
    observed_at: str = field(default_factory=utc_now_iso)
    safety_reason: Optional[str] = None
    accepted: Optional[bool] = None
    latency_ms: Optional[float] = None
    value: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> Dict[str, Any]:
        record = asdict(self)
        return {key: value for key, value in record.items() if value is not None}


class SupabaseRestSink:
    """Write batches through Supabase's PostgREST endpoint."""

    def __init__(self, url: str, api_key: str, table: str = "franka_telemetry", timeout_sec: float = 5.0):
        if not url or not api_key:
            raise ValueError("Supabase URL and telemetry key are required")
        self.endpoint = f"{url.rstrip('/')}/rest/v1/{table}"
        self.api_key = api_key
        self.timeout_sec = float(timeout_sec)

    def write(self, records: Iterable[Dict[str, Any]]) -> None:
        payload = list(records)
        if not payload:
            return
        request = Request(
            self.endpoint,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            method="POST",
            headers={
                "apikey": self.api_key,
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
        )
        with urlopen(request, timeout=self.timeout_sec) as response:
            status = int(response.getcode())
            if status < 200 or status >= 300:
                raise RuntimeError(f"Supabase insert returned HTTP {status}")


class LoggingSink:
    """Safe local sink for demos and CI. It never sends data off-host."""

    def __init__(self, emit: Any = None):
        self.emit = emit or LOGGER.info

    def write(self, records: Iterable[Dict[str, Any]]) -> None:
        for record in records:
            self.emit("telemetry=%s", json.dumps(record, sort_keys=True))


class AsyncTelemetryClient:
    """Bounded asynchronous telemetry queue with retry accounting."""

    def __init__(
        self,
        sink: Any,
        queue_size: int = 1000,
        batch_size: int = 50,
        flush_interval_sec: float = 1.0,
        max_retries: int = 2,
    ):
        queue_size = int(queue_size)
        batch_size = int(batch_size)
        flush_interval_sec = float(flush_interval_sec)
        max_retries = int(max_retries)
        if queue_size <= 0 or batch_size <= 0:
            raise ValueError("queue_size and batch_size must be positive integers")
        if not math.isfinite(flush_interval_sec) or flush_interval_sec <= 0:
            raise ValueError("flush_interval_sec must be a positive finite number")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.sink = sink
        self.batch_size = batch_size
        self.flush_interval_sec = flush_interval_sec
        self.max_retries = max_retries
        self._queue = Queue(maxsize=queue_size)
        self._stop = Event()
        self._thread = Thread(target=self._worker, name="telemetry-writer", daemon=True)
        self._started = False
        self.dropped_events = 0
        self.failed_events = 0
        self.written_events = 0

    @property
    def queue_depth(self) -> int:
        return int(self._queue.qsize())

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def submit(self, event: TelemetryEvent) -> bool:
        """Queue an event without blocking the caller."""

        try:
            self._queue.put_nowait(event)
            return True
        except Full:
            self.dropped_events += 1
            return False

    def stop(self, timeout_sec: float = 5.0) -> None:
        self._stop.set()
        if self._started:
            self._thread.join(timeout=max(0.0, float(timeout_sec)))

    def _take_batch(self) -> List[TelemetryEvent]:
        batch: List[TelemetryEvent] = []
        try:
            batch.append(self._queue.get(timeout=self.flush_interval_sec))
        except Empty:
            return batch
        while len(batch) < self.batch_size:
            try:
                batch.append(self._queue.get_nowait())
            except Empty:
                break
        return batch

    def _write_with_retry(self, batch: List[TelemetryEvent]) -> None:
        records = [event.to_record() for event in batch]
        for attempt in range(self.max_retries + 1):
            try:
                self.sink.write(records)
                self.written_events += len(batch)
                return
            except Exception:  # Network and sink failures must stay off the control path.
                if attempt >= self.max_retries:
                    self.failed_events += len(batch)
                    LOGGER.exception("telemetry batch failed after retries")
                    return
                time.sleep(min(0.25 * (2**attempt), 1.0))

    def _worker(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            batch = self._take_batch()
            if not batch:
                continue
            self._write_with_retry(batch)
            for _ in batch:
                self._queue.task_done()
