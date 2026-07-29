"""Versioned teleop configuration schema and validator (Phase 8).

Fails fast with a useful message on invalid configuration. No silent coercion of
malformed vectors. Pure Python (+ optional YAML at the edges).

A valid config is a mapping like::

    schema_version: "1.0"
    dt: 0.05
    initial_position: [0.45, 0.0, 0.45]
    retargeter: {name: shoulder_relative, treat_z_as_image_relative: true,
                 image_relative_z_scale: 0.5}
    workspace: {x_min: .., x_max: .., y_min: .., y_max: .., z_min: .., z_max: ..}
    safety:
      mode: cbf_qp                      # reject | clamp | cbf_qp
      workspace_margin_m: 0.03
      alpha: 4.0
      per_axis_velocity_limit_mps: 0.20
      allow_projection_fallback: false
      stop_on_solver_failure: true
      obstacles: [{id: s, center: [..], radius_m: .., margin_m: ..}]
    gate: {min_keypoint_confidence: 0.5, hold_interval_sec: 0.25, ...}
    backend: dry_run                     # dry_run | ros2_fake_hardware | optional_moveit
"""
from __future__ import annotations

from typing import Any, Mapping

SCHEMA_VERSION = "1.0"

ALLOWED_RETARGETERS = {"shoulder_relative", "sew_orientation"}
ALLOWED_SAFETY_MODES = {"reject", "clamp", "cbf_qp"}
ALLOWED_BACKENDS = {"dry_run", "ros2_fake_hardware", "optional_moveit"}
ALLOWED_FRAMES = {
    "image_normalized",
    "camera_optical",
    "human_shoulder",
    "robot_base",
    "end_effector",
}


class ConfigError(ValueError):
    """Raised on any invalid teleop configuration."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def _num(d: Mapping[str, Any], key: str, ctx: str, *, lo=None, hi=None, allow_eq_lo=True) -> float:
    _require(key in d, f"{ctx}: missing required key '{key}'")
    v = d[key]
    _require(
        isinstance(v, (int, float)) and not isinstance(v, bool),
        f"{ctx}.{key}: expected number, got {type(v).__name__}",
    )
    v = float(v)
    if lo is not None:
        ok_lo = v >= lo if allow_eq_lo else v > lo
        rel = ">=" if allow_eq_lo else ">"
        _require(ok_lo, f"{ctx}.{key}={v} out of range ({rel} {lo})")
    if hi is not None:
        _require(v <= hi, f"{ctx}.{key}={v} out of range (<= {hi})")
    return v


def _unknown_keys(d: Mapping[str, Any], allowed: set, ctx: str) -> None:
    extra = set(d) - allowed
    _require(not extra, f"{ctx}: unknown key(s): {sorted(extra)}; allowed: {sorted(allowed)}")


def _vec3(v: Any, ctx: str) -> None:
    _require(
        isinstance(v, (list, tuple))
        and len(v) == 3
        and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v),
        f"{ctx}: expected a length-3 numeric vector, got {v!r}",
    )


def _validate_workspace(ws: Mapping[str, Any]) -> None:
    _require(isinstance(ws, Mapping), "workspace: expected a mapping")
    _unknown_keys(ws, {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}, "workspace")
    for axis in "xyz":
        lo = _num(ws, f"{axis}_min", "workspace")
        hi = _num(ws, f"{axis}_max", "workspace")
        _require(lo < hi, f"workspace: {axis}_min ({lo}) must be < {axis}_max ({hi})")


def _validate_safety(safety: Mapping[str, Any], workspace: Mapping[str, Any]) -> None:
    _require(isinstance(safety, Mapping), "safety: expected a mapping")
    _unknown_keys(
        safety,
        {
            "mode",
            "workspace_margin_m",
            "alpha",
            "per_axis_velocity_limit_mps",
            "max_cartesian_velocity_mps",
            "allow_projection_fallback",
            "stop_on_solver_failure",
            "obstacles",
        },
        "safety",
    )
    mode = safety.get("mode")
    _require(
        mode in ALLOWED_SAFETY_MODES, f"safety.mode='{mode}' invalid; allowed: {sorted(ALLOWED_SAFETY_MODES)}"
    )

    margin = _num(safety, "workspace_margin_m", "safety", lo=0.0) if "workspace_margin_m" in safety else 0.0
    # margin must be smaller than half the smallest workspace extent
    if workspace:
        extents = [float(workspace[f"{a}_max"]) - float(workspace[f"{a}_min"]) for a in "xyz"]
        half = 0.5 * min(extents)
        _require(
            margin < half,
            f"safety.workspace_margin_m={margin} must be < half the smallest workspace extent ({half:.3f})",
        )

    for k in ("per_axis_velocity_limit_mps", "max_cartesian_velocity_mps"):
        if k in safety:
            _num(safety, k, "safety", lo=0.0, allow_eq_lo=False)
    if "alpha" in safety:
        _num(safety, "alpha", "safety", lo=0.0, allow_eq_lo=False)
    for flag in ("allow_projection_fallback", "stop_on_solver_failure"):
        if flag in safety:
            _require(isinstance(safety[flag], bool), f"safety.{flag}: expected bool")

    obstacles = safety.get("obstacles", [])
    _require(isinstance(obstacles, (list, tuple)), "safety.obstacles: expected a list")
    if obstacles and mode != "cbf_qp":
        raise ConfigError(
            f"safety.obstacles defined but safety.mode='{mode}' ignores obstacles (use mode: cbf_qp)"
        )
    seen = set()
    for i, obs in enumerate(obstacles):
        ctx = f"safety.obstacles[{i}]"
        _require(isinstance(obs, Mapping), f"{ctx}: expected a mapping")
        _unknown_keys(obs, {"id", "center", "radius_m", "margin_m"}, ctx)
        _require(
            "id" in obs and isinstance(obs["id"], str) and obs["id"],
            f"{ctx}: 'id' must be a non-empty string",
        )
        _require(obs["id"] not in seen, f"{ctx}: duplicate obstacle id '{obs['id']}'")
        seen.add(obs["id"])
        _vec3(obs.get("center"), f"{ctx}.center")
        _num(obs, "radius_m", ctx, lo=0.0, allow_eq_lo=False)
        if "margin_m" in obs:
            _num(obs, "margin_m", ctx, lo=0.0)


def _validate_retargeter(ret: Mapping[str, Any]) -> None:
    _require(isinstance(ret, Mapping), "retargeter: expected a mapping")
    _unknown_keys(
        ret,
        {
            "name",
            "treat_z_as_image_relative",
            "image_relative_z_scale",
            "workspace_origin",
            "workspace_scale",
            "min_confidence",
        },
        "retargeter",
    )
    name = ret.get("name")
    _require(
        name in ALLOWED_RETARGETERS,
        f"retargeter.name='{name}' invalid; allowed: {sorted(ALLOWED_RETARGETERS)}",
    )
    if "image_relative_z_scale" in ret:
        _num(ret, "image_relative_z_scale", "retargeter", lo=0.0)
    if "treat_z_as_image_relative" in ret:
        _require(
            isinstance(ret["treat_z_as_image_relative"], bool),
            "retargeter.treat_z_as_image_relative: expected bool",
        )
    for k in ("workspace_origin", "workspace_scale"):
        if k in ret:
            _vec3(ret[k], f"retargeter.{k}")


def validate_teleop_config(cfg: Mapping[str, Any]) -> None:
    """Validate a teleop config mapping; raise ConfigError on any problem."""
    _require(isinstance(cfg, Mapping), "config: expected a mapping at top level")
    _unknown_keys(
        cfg,
        {
            "schema_version",
            "dt",
            "initial_position",
            "retargeter",
            "workspace",
            "safety",
            "gate",
            "backend",
            "frame_mapping",
        },
        "config",
    )
    ver = cfg.get("schema_version")
    _require(
        str(ver) == SCHEMA_VERSION, f"config.schema_version='{ver}' unsupported; expected '{SCHEMA_VERSION}'"
    )

    _num(cfg, "dt", "config", lo=0.0, allow_eq_lo=False)
    if "initial_position" in cfg:
        _vec3(cfg["initial_position"], "config.initial_position")

    _require("workspace" in cfg, "config: missing required 'workspace'")
    _validate_workspace(cfg["workspace"])

    _require("retargeter" in cfg, "config: missing required 'retargeter'")
    _validate_retargeter(cfg["retargeter"])

    _require("safety" in cfg, "config: missing required 'safety'")
    _validate_safety(cfg["safety"], cfg["workspace"])

    backend = cfg.get("backend", "dry_run")
    _require(
        backend in ALLOWED_BACKENDS,
        f"config.backend='{backend}' invalid; allowed: {sorted(ALLOWED_BACKENDS)}",
    )

    fm = cfg.get("frame_mapping")
    if fm is not None:
        _require(isinstance(fm, Mapping), "frame_mapping: expected a mapping")
        for fk in ("source_frame", "target_frame"):
            if fk in fm:
                _require(
                    fm[fk] in ALLOWED_FRAMES,
                    f"frame_mapping.{fk}='{fm[fk]}' invalid; allowed: {sorted(ALLOWED_FRAMES)}",
                )
