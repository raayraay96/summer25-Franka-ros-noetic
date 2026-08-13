#!/usr/bin/env python3
"""Forward structured ROS safety metrics to an asynchronous telemetry sink."""

from __future__ import annotations

import json
import os
import uuid

import rospy
from std_msgs.msg import String

from vision_arm_control.telemetry import (
    AsyncTelemetryClient,
    LoggingSink,
    SupabaseRestSink,
    TelemetryEvent,
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


ALLOWED_EVENT_TYPES = {"safety_decision", "estop_trigger", "estop_clear", "node_heartbeat"}


class TelemetryNode:
    def __init__(self) -> None:
        rospy.init_node("telemetry_node")
        requested_run_id = str(rospy.get_param("~run_id", os.getenv("TELEMETRY_RUN_ID", uuid.uuid4())))
        try:
            self.run_id = str(uuid.UUID(requested_run_id))
        except ValueError:
            self.run_id = str(uuid.uuid4())
            rospy.logwarn("Invalid telemetry run_id; generated %s", self.run_id)
        self.source = str(
            rospy.get_param("~source", os.getenv("TELEMETRY_SOURCE", "ros1-noetic-simulation"))
        )
        enabled = bool(rospy.get_param("~enabled", _env_flag("TELEMETRY_ENABLED", False)))
        dry_run = bool(rospy.get_param("~dry_run", _env_flag("TELEMETRY_DRY_RUN", True)))

        if enabled and not dry_run:
            url = os.getenv("SUPABASE_URL", "")
            key = os.getenv("SUPABASE_TELEMETRY_KEY", "")
            table = os.getenv("SUPABASE_TELEMETRY_TABLE", "franka_telemetry")
            if not url or not key:
                raise RuntimeError(
                    "TELEMETRY_ENABLED=true requires SUPABASE_URL and SUPABASE_TELEMETRY_KEY"
                )
            sink = SupabaseRestSink(url=url, api_key=key, table=table)
            rospy.loginfo("telemetry_node using Supabase sink table=%s", table)
        else:
            sink = LoggingSink(emit=rospy.loginfo)
            rospy.loginfo("telemetry_node using local logging sink (no network writes)")

        self.client = AsyncTelemetryClient(
            sink=sink,
            queue_size=int(rospy.get_param("~queue_size", 1000)),
            batch_size=int(rospy.get_param("~batch_size", 50)),
            flush_interval_sec=float(rospy.get_param("~flush_interval_sec", 1.0)),
        )
        self.client.start()
        rospy.on_shutdown(self._shutdown)

        input_topic = str(rospy.get_param("~input_topic", "/safety_monitor_node/telemetry"))
        self.subscriber = rospy.Subscriber(input_topic, String, self._event_callback, queue_size=100)
        heartbeat_sec = float(rospy.get_param("~heartbeat_sec", 5.0))
        self.heartbeat_timer = rospy.Timer(rospy.Duration(heartbeat_sec), self._heartbeat)
        rospy.loginfo("telemetry_node run_id=%s input=%s", self.run_id, input_topic)

    def _event_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except (TypeError, ValueError):
            rospy.logwarn_throttle(10.0, "telemetry_node dropped invalid JSON")
            return

        event_type = str(payload.get("event_type", ""))
        if event_type not in ALLOWED_EVENT_TYPES:
            rospy.logwarn_throttle(10.0, "telemetry_node dropped unsupported event_type")
            return
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            rospy.logwarn_throttle(10.0, "telemetry_node dropped non-object metadata")
            return

        event = TelemetryEvent(
            event_type=event_type,
            run_id=self.run_id,
            source=self.source,
            safety_reason=payload.get("safety_reason"),
            accepted=payload.get("accepted"),
            latency_ms=payload.get("latency_ms"),
            value=payload.get("value"),
            metadata=metadata,
        )
        if not self.client.submit(event):
            rospy.logwarn_throttle(10.0, "telemetry queue full; event dropped")

    def _heartbeat(self, _event: object) -> None:
        heartbeat = TelemetryEvent(
            event_type="node_heartbeat",
            run_id=self.run_id,
            source=self.source,
            value=float(self.client.queue_depth),
            metadata={
                "queue_depth": self.client.queue_depth,
                "dropped_events": self.client.dropped_events,
                "failed_events": self.client.failed_events,
                "written_events": self.client.written_events,
            },
        )
        self.client.submit(heartbeat)

    def _shutdown(self) -> None:
        self.client.stop(timeout_sec=5.0)


if __name__ == "__main__":
    try:
        TelemetryNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
