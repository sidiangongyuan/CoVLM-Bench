"""Current-frame L1 evidence tokens for structured-evidence V2X planning.

The feature builder intentionally uses only L1 perception and refined motion
fields from the current frame.  It does not read L4 selected critical objects,
CoT text, GT command, or future trajectory labels.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_COVLA_ROOT = Path("data/covlm_bench")
DEFAULT_L1_DIR = DEFAULT_COVLA_ROOT / "l1_perception"
DEFAULT_L1_REFINED_DIR = DEFAULT_COVLA_ROOT / "l1_motion_refined"

CATEGORY_VOCAB = [
    "car",
    "truck",
    "bus",
    "van",
    "pedestrian",
    "cyclist",
    "bicycle",
    "motorcycle",
    "traffic_cone",
    "barrier",
    "object",
    "unknown",
]
VISIBILITY_VOCAB = ["ego_only", "both", "infra_only", "unknown"]
MOTION_VOCAB = [
    "parked",
    "stopped_at_signal",
    "going_straight",
    "moving_slowly",
    "turning",
    "unknown",
]
HEADING_VOCAB = [
    "same direction",
    "opposite direction",
    "crossing",
    "cross-oriented",
    "oblique",
    "unknown",
]
POSITION_VOCAB = [
    "front",
    "front_left",
    "front_right",
    "left",
    "right",
    "rear",
    "rear_left",
    "rear_right",
    "overlap",
    "unknown",
]
ROLE_VOCAB = ["infra_priority", "ego_anchor", "distance_fill", "unused"]
ROUTE_SIDE_VOCAB = ["left_of_route", "right_of_route", "on_route", "unknown"]
ROUTE_AHEAD_VOCAB = ["behind", "near_route", "ahead_route", "far_ahead", "unknown"]
ROUTE_VISIBILITY_ROLE_VOCAB = [
    "infra_only_route",
    "shared_route",
    "ego_only_route",
    "unknown_route",
]
ROUTE_EVIDENCE_ROLE_VOCAB = [
    "infra_route_constraint",
    "shared_route_context",
    "ego_route_anchor",
    "route_context_fill",
    "unused",
]

NUMERIC_FEATURES = [
    "local_x_norm",
    "local_y_norm",
    "distance_norm",
    "instant_speed_norm",
    "smoothed_speed_norm",
    "yaw_change_norm",
    "is_front",
    "is_lateral",
    "is_rear",
    "is_nonparked",
]
FEATURE_DIM = (
    len(NUMERIC_FEATURES)
    + len(CATEGORY_VOCAB)
    + len(VISIBILITY_VOCAB)
    + len(MOTION_VOCAB)
    + len(HEADING_VOCAB)
    + len(POSITION_VOCAB)
    + len(ROLE_VOCAB)
)
ROUTE_NUMERIC_FEATURES = NUMERIC_FEATURES + [
    "route_corridor_distance_norm",
    "route_closest_step_norm",
    "route_within_corridor",
    "route_forward_margin_norm",
    "route_relevance_score_norm",
]
ROUTE_FEATURE_DIM = (
    len(ROUTE_NUMERIC_FEATURES)
    + len(CATEGORY_VOCAB)
    + len(VISIBILITY_VOCAB)
    + len(MOTION_VOCAB)
    + len(HEADING_VOCAB)
    + len(POSITION_VOCAB)
    + len(ROUTE_SIDE_VOCAB)
    + len(ROUTE_AHEAD_VOCAB)
    + len(ROUTE_VISIBILITY_ROLE_VOCAB)
    + len(ROUTE_EVIDENCE_ROLE_VOCAB)
)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _clip(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


def _one_hot(value: str, vocab: Sequence[str]) -> List[float]:
    norm = str(value or "unknown").strip().lower().replace("_", " ")
    out = [0.0] * len(vocab)
    for idx, item in enumerate(vocab):
        if norm == item.lower().replace("_", " "):
            out[idx] = 1.0
            return out
    out[-1] = 1.0
    return out


def heading_label(yaw_rad: Any) -> str:
    rel = abs(_safe_float(yaw_rad, 0.0)) % (2 * math.pi)
    if rel > math.pi:
        rel = 2 * math.pi - rel
    if rel < math.pi / 4:
        return "same direction"
    if rel > 3 * math.pi / 4:
        return "opposite direction"
    return "cross-oriented"


def position_bin(local_x: float, local_y: float) -> str:
    """Return a coarse ego-relative bin.

    In the L1 labels local_x is forward/backward and local_y is lateral.
    """
    front = local_x > 5.0
    rear = local_x < -5.0
    left = local_y > 2.0
    right = local_y < -2.0
    if front and left:
        return "front_left"
    if front and right:
        return "front_right"
    if rear and left:
        return "rear_left"
    if rear and right:
        return "rear_right"
    if front:
        return "front"
    if rear:
        return "rear"
    if left:
        return "left"
    if right:
        return "right"
    return "overlap"


def _merge_refined(l1_data: Dict[str, Any], refined_path: Path) -> bool:
    objects = l1_data.get("objects", [])
    if not isinstance(objects, list) or not refined_path.exists():
        return False
    refined = _load_json(refined_path).get("objects_refined", [])
    if not isinstance(refined, list):
        return False
    by_index = {
        int(row["object_index"]): row
        for row in refined
        if isinstance(row, dict) and row.get("object_index") is not None
    }
    merged = False
    for idx, obj in enumerate(objects):
        if not isinstance(obj, dict):
            continue
        row = by_index.get(idx)
        if not row:
            continue
        for key in (
            "track_id",
            "motion_state_refined",
            "heading_refined",
            "smoothed_speed_mps",
            "yaw_change_deg",
            "track_frames_in_window",
            "window_extent_sec",
        ):
            if key in row:
                obj[key] = row[key]
        merged = True
    return merged


def load_l1_records(
    split: str,
    token: str,
    *,
    l1_dir: Path = DEFAULT_L1_DIR,
    refined_dir: Path = DEFAULT_L1_REFINED_DIR,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    l1_path = l1_dir / f"{split}_{token}.json"
    if not l1_path.exists():
        raise FileNotFoundError(f"L1 perception file missing: {l1_path}")
    l1_data = _load_json(l1_path)
    refined_path = refined_dir / f"{split}_{token}.json"
    refined_merged = _merge_refined(l1_data, refined_path)
    records: List[Dict[str, Any]] = []
    for idx, obj in enumerate(l1_data.get("objects", []) or []):
        if not isinstance(obj, dict):
            continue
        ego_relative = obj.get("ego_relative", {}) if isinstance(obj.get("ego_relative"), dict) else {}
        local_x = _safe_float(ego_relative.get("local_x"), 0.0)
        local_y = _safe_float(ego_relative.get("local_y"), 0.0)
        distance = _safe_float(ego_relative.get("distance_m"), math.hypot(local_x, local_y))
        instant_speed = _safe_float(obj.get("speed_mps"), 0.0)
        smoothed_speed = _safe_float(obj.get("smoothed_speed_mps"), instant_speed)
        motion = str(obj.get("motion_state_refined") or "unknown")
        heading = str(obj.get("heading_refined") or heading_label(obj.get("yaw_rad")))
        records.append(
            {
                "l1_index": idx,
                "category": str(obj.get("category") or obj.get("category_raw") or "object"),
                "visibility_source": str(obj.get("visibility_source") or "unknown"),
                "distance_m": distance,
                "local_x": local_x,
                "local_y": local_y,
                "position_bin": position_bin(local_x, local_y),
                "relative_position": str(obj.get("position_description") or "unknown"),
                "motion_state_refined": motion,
                "heading_refined": heading,
                "instant_speed_mps": instant_speed,
                "smoothed_speed_mps": smoothed_speed,
                "yaw_change_deg": _safe_float(obj.get("yaw_change_deg"), 0.0),
                "track_id": obj.get("track_id"),
            }
        )
    meta = {
        "l1_path": str(l1_path),
        "refined_path": str(refined_path),
        "refined_merged": refined_merged,
        "num_l1_objects": len(records),
    }
    return records, meta


def is_nonparked(rec: Dict[str, Any]) -> bool:
    return str(rec.get("motion_state_refined") or "unknown") != "parked"


def _near_front_lateral(rec: Dict[str, Any], max_distance: float = 80.0) -> bool:
    distance = _safe_float(rec.get("distance_m"), 1e9)
    local_x = _safe_float(rec.get("local_x"), 0.0)
    local_y = abs(_safe_float(rec.get("local_y"), 0.0))
    return distance <= max_distance and local_x >= -15.0 and local_y <= 60.0


def _rank_key(rec: Dict[str, Any]) -> Tuple[float, float, float]:
    distance = _safe_float(rec.get("distance_m"), 1e9)
    local_x = _safe_float(rec.get("local_x"), 0.0)
    lateral_abs = abs(_safe_float(rec.get("local_y"), 0.0))
    rear_penalty = 25.0 if local_x < -5.0 else 0.0
    lateral_penalty = max(0.0, lateral_abs - 20.0) * 0.5
    return (distance + rear_penalty + lateral_penalty, distance, lateral_abs)


def select_evidence_records(records: Sequence[Dict[str, Any]], top_k: int = 12) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    used = set()

    def add(pool: Iterable[Dict[str, Any]], role: str, limit: int) -> None:
        nonlocal selected
        for rec in pool:
            idx = int(rec.get("l1_index", -1))
            if idx in used:
                continue
            new_rec = dict(rec)
            new_rec["evidence_role"] = role
            selected.append(new_rec)
            used.add(idx)
            if len([r for r in selected if r.get("evidence_role") == role]) >= limit:
                return
            if len(selected) >= top_k:
                return

    infra_pool = sorted(
        [
            r
            for r in records
            if r.get("visibility_source") == "infra_only"
            and is_nonparked(r)
            and _near_front_lateral(r, 80.0)
        ],
        key=_rank_key,
    )
    add(infra_pool, "infra_priority", min(6, top_k))

    anchor_pool = sorted(
        [
            r
            for r in records
            if r.get("visibility_source") in {"ego_only", "both", "unknown"}
            and is_nonparked(r)
            and _safe_float(r.get("distance_m"), 1e9) <= 60.0
            and _safe_float(r.get("local_x"), 0.0) >= -20.0
        ],
        key=_rank_key,
    )
    add(anchor_pool, "ego_anchor", min(4, top_k - len(selected)))

    fill_pool = sorted([r for r in records if is_nonparked(r)], key=_rank_key)
    add(fill_pool, "distance_fill", top_k)

    nearest_infra = None
    infra_count = 0
    near_infra_count = 0
    for rec in records:
        if rec.get("visibility_source") != "infra_only" or not is_nonparked(rec):
            continue
        infra_count += 1
        distance = _safe_float(rec.get("distance_m"), 1e9)
        nearest_infra = distance if nearest_infra is None else min(nearest_infra, distance)
        if _near_front_lateral(rec, 40.0):
            near_infra_count += 1
    if near_infra_count > 0:
        relevance_prior = 0.8
    elif infra_count >= 3 and nearest_infra is not None and nearest_infra <= 80.0:
        relevance_prior = 0.5
    else:
        relevance_prior = 0.0

    summary = {
        "top_k": top_k,
        "selected_count": len(selected),
        "infra_priority_selected": sum(r.get("evidence_role") == "infra_priority" for r in selected),
        "ego_anchor_selected": sum(r.get("evidence_role") == "ego_anchor" for r in selected),
        "distance_fill_selected": sum(r.get("evidence_role") == "distance_fill" for r in selected),
        "infra_only_nonparked_count": infra_count,
        "near_infra_only_nonparked_count": near_infra_count,
        "nearest_infra_only_nonparked_distance": nearest_infra,
        "v2x_relevance_prior": relevance_prior,
        "selected_l1_indices": [int(r.get("l1_index", -1)) for r in selected],
    }
    return selected[:top_k], summary


def _route_points_from_waypoints(
    base_waypoints: Sequence[Sequence[float]],
    near_horizon_steps: int,
) -> List[Tuple[float, float]]:
    """Map CoVLM waypoints [x_lateral, y_forward] to L1 local (forward, lateral)."""
    points: List[Tuple[float, float]] = []
    limit = max(1, min(int(near_horizon_steps), len(base_waypoints)))
    for step in base_waypoints[:limit]:
        if not isinstance(step, (list, tuple)) or len(step) < 2:
            continue
        lateral = _safe_float(step[0], 0.0)
        forward = _safe_float(step[1], 0.0)
        points.append((forward, lateral))
    return points


def _closest_route_point(
    local_x: float,
    local_y: float,
    route_points: Sequence[Tuple[float, float]],
) -> Tuple[float, int, float, float]:
    if not route_points:
        return math.hypot(local_x, local_y), 0, 0.0, 0.0
    best_idx = 0
    best_dist = float("inf")
    best_forward, best_lateral = route_points[0]
    for idx, (forward, lateral) in enumerate(route_points):
        dist = math.hypot(local_x - forward, local_y - lateral)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
            best_forward = forward
            best_lateral = lateral
    return best_dist, best_idx, best_forward, best_lateral


def _route_side(local_y: float, route_lateral: float, corridor_width_m: float) -> str:
    margin = local_y - route_lateral
    side_eps = max(1.0, min(2.0, corridor_width_m * 0.25))
    if margin > side_eps:
        return "left_of_route"
    if margin < -side_eps:
        return "right_of_route"
    return "on_route"


def _route_ahead_bin(local_x: float, route_forward: float, max_route_forward: float) -> str:
    if local_x < -5.0:
        return "behind"
    if local_x <= route_forward + 6.0:
        return "near_route"
    if local_x <= max_route_forward + 15.0:
        return "ahead_route"
    return "far_ahead"


def _route_visibility_role(visibility: str) -> str:
    if visibility == "infra_only":
        return "infra_only_route"
    if visibility == "both":
        return "shared_route"
    if visibility == "ego_only":
        return "ego_only_route"
    return "unknown_route"


def _annotate_route_record(
    rec: Dict[str, Any],
    route_points: Sequence[Tuple[float, float]],
    *,
    corridor_width_m: float,
) -> Dict[str, Any]:
    local_x = _safe_float(rec.get("local_x"), 0.0)
    local_y = _safe_float(rec.get("local_y"), 0.0)
    route_dist, closest_step, route_forward, route_lateral = _closest_route_point(
        local_x,
        local_y,
        route_points,
    )
    max_forward = max((p[0] for p in route_points), default=0.0)
    visibility = str(rec.get("visibility_source") or "unknown")
    within = route_dist <= float(corridor_width_m)
    near_forward = -8.0 <= local_x <= max_forward + 12.0
    nonparked = is_nonparked(rec)
    visibility_bonus = 1.0 if visibility == "infra_only" else 0.55 if visibility == "both" else 0.25
    route_bonus = 1.0 - _clip(route_dist / max(float(corridor_width_m) * 2.0, 1.0), 0.0, 1.0)
    forward_bonus = 1.0 if near_forward else 0.0
    motion_bonus = 1.0 if nonparked else 0.0
    relevance_score = 0.45 * route_bonus + 0.25 * visibility_bonus + 0.2 * forward_bonus + 0.1 * motion_bonus
    out = dict(rec)
    out.update(
        {
            "route_corridor_distance_m": route_dist,
            "route_closest_step": int(closest_step),
            "route_closest_forward_m": route_forward,
            "route_closest_lateral_m": route_lateral,
            "route_max_forward_m": max_forward,
            "route_within_corridor": bool(within),
            "route_side_of_route": _route_side(local_y, route_lateral, float(corridor_width_m)),
            "route_ahead_behind_bin": _route_ahead_bin(local_x, route_forward, max_forward),
            "route_visibility_role": _route_visibility_role(visibility),
            "route_relevance_score": relevance_score,
        }
    )
    return out


def _trajectory_geometry(base_waypoints: Sequence[Sequence[float]]) -> Dict[str, Any]:
    points = _route_points_from_waypoints(base_waypoints, len(base_waypoints))
    lateral = [p[1] for p in points]
    forward = [p[0] for p in points]
    if not points:
        return {
            "lateral_delta": 0.0,
            "lateral_span": 0.0,
            "forward_span": 0.0,
            "lateral_direction": "straight",
            "lateral_maneuver_like": False,
            "turn_like": False,
        }
    lateral_delta = lateral[-1] - lateral[0]
    lateral_span = max(lateral) - min(lateral)
    forward_span = max(forward) - min(forward)
    direction = "left" if lateral_delta > 0.6 else "right" if lateral_delta < -0.6 else "straight"
    lateral_maneuver_like = abs(lateral_delta) >= 1.2 or lateral_span >= 1.8
    turn_like = lateral_maneuver_like and forward_span <= 28.0
    return {
        "lateral_delta": lateral_delta,
        "lateral_span": lateral_span,
        "forward_span": forward_span,
        "lateral_direction": direction,
        "lateral_maneuver_like": lateral_maneuver_like,
        "turn_like": turn_like,
    }


def lateral_shape_protection_from_waypoints(
    base_waypoints: Sequence[Sequence[float]],
    *,
    protected_residual_scale: float = 0.2,
) -> Dict[str, Any]:
    """Detect gentle lateral-shift-like base routes without reading labels."""
    points = _route_points_from_waypoints(base_waypoints, len(base_waypoints))
    if len(points) < 2:
        return {
            "lateral_shape_protection": False,
            "lateral_shape_residual_scale": 1.0,
            "lateral_shape_score": 0.0,
            "lateral_shape_detection_source": "ego_base_waypoints_only",
            "lateral_shape_features": {},
        }
    forward = [p[0] for p in points]
    lateral = [p[1] for p in points]
    lateral_delta = lateral[-1] - lateral[0]
    lateral_span = max(lateral) - min(lateral)
    forward_span = max(forward) - min(forward)
    headings = [
        math.atan2(lateral[idx + 1] - lateral[idx], forward[idx + 1] - forward[idx])
        for idx in range(len(points) - 1)
    ]
    heading_change_abs = abs(headings[-1] - headings[0]) if len(headings) >= 2 else 0.0
    final_heading_abs = abs(headings[-1]) if headings else 0.0
    curvature: List[float] = []
    for idx in range(len(points) - 2):
        v1 = (
            lateral[idx + 1] - lateral[idx],
            forward[idx + 1] - forward[idx],
        )
        v2 = (
            lateral[idx + 2] - lateral[idx + 1],
            forward[idx + 2] - forward[idx + 1],
        )
        n1 = math.hypot(v1[0], v1[1])
        n2 = math.hypot(v2[0], v2[1])
        if n1 <= 1e-6 or n2 <= 1e-6:
            continue
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        curvature.append(abs(math.atan2(cross, dot)))
    curvature_sum = sum(curvature)
    lateral_forward_ratio = abs(lateral_delta) / max(forward_span, 1e-6)

    protected = (
        abs(lateral_delta) >= 0.3
        and lateral_forward_ratio <= 0.08
        and heading_change_abs <= 0.2
        and curvature_sum <= 0.2
        and forward_span >= 12.0
        and final_heading_abs <= 0.08
    )
    score = 0.0
    if protected:
        score = min(1.0, abs(lateral_delta) / 1.5)
    return {
        "lateral_shape_protection": bool(protected),
        "lateral_shape_residual_scale": float(protected_residual_scale if protected else 1.0),
        "lateral_shape_score": float(score),
        "lateral_shape_detection_source": "ego_base_waypoints_only",
        "lateral_shape_features": {
            "lateral_delta_m": round(float(lateral_delta), 6),
            "lateral_span_m": round(float(lateral_span), 6),
            "forward_span_m": round(float(forward_span), 6),
            "lateral_forward_ratio": round(float(lateral_forward_ratio), 6),
            "heading_change_abs_rad": round(float(heading_change_abs), 6),
            "final_heading_abs_rad": round(float(final_heading_abs), 6),
            "curvature_sum_rad": round(float(curvature_sum), 6),
        },
    }


def _motion_score(rec: Dict[str, Any]) -> float:
    motion = str(rec.get("motion_state_refined") or "unknown")
    speed = max(
        _safe_float(rec.get("instant_speed_mps"), 0.0),
        _safe_float(rec.get("smoothed_speed_mps"), 0.0),
    )
    moving = 0.0 if motion == "parked" else 0.45
    if motion in {"going_straight", "turning", "moving_slowly"}:
        moving += 0.25
    return _clip(moving + 0.30 * _clip(speed / 12.0, 0.0, 1.0), 0.0, 1.0)


def _visibility_score(rec: Dict[str, Any]) -> float:
    visibility = str(rec.get("visibility_source") or "unknown")
    if visibility == "infra_only":
        return 1.0
    if visibility == "both":
        return 0.72
    if visibility == "ego_only":
        return 0.24
    return 0.12


def _front_lateral_score(rec: Dict[str, Any]) -> float:
    local_x = _safe_float(rec.get("local_x"), 0.0)
    local_y = abs(_safe_float(rec.get("local_y"), 0.0))
    max_forward = _safe_float(rec.get("route_max_forward_m"), 30.0)
    if local_x < -8.0:
        forward = 0.0
    elif local_x <= max_forward + 20.0:
        forward = 1.0
    else:
        forward = math.exp(-(local_x - max_forward - 20.0) / 60.0)
    lateral = math.exp(-max(0.0, local_y - 3.0) / 28.0)
    return _clip(0.65 * forward + 0.35 * lateral, 0.0, 1.0)


def _route_distance_score(rec: Dict[str, Any], scale: float = 24.0) -> float:
    route_dist = _safe_float(rec.get("route_corridor_distance_m"), 1e9)
    return math.exp(-max(0.0, route_dist) / max(scale, 1.0))


def _distance_decay_score(rec: Dict[str, Any]) -> float:
    distance = _safe_float(rec.get("distance_m"), 1e9)
    return math.exp(-max(0.0, distance) / 90.0)


def _corridor_soft_score(rec: Dict[str, Any]) -> float:
    score = (
        0.32 * _visibility_score(rec)
        + 0.20 * _front_lateral_score(rec)
        + 0.24 * _route_distance_score(rec, 24.0)
        + 0.14 * _motion_score(rec)
        + 0.10 * _distance_decay_score(rec)
    )
    return _clip(score, 0.0, 1.0)


def _risk_zone_score(rec: Dict[str, Any], geometry: Dict[str, Any]) -> float:
    local_x = _safe_float(rec.get("local_x"), 0.0)
    local_y = _safe_float(rec.get("local_y"), 0.0)
    route_side = str(rec.get("route_side_of_route") or "unknown")
    heading = str(rec.get("heading_refined") or "unknown")
    motion = str(rec.get("motion_state_refined") or "unknown")
    near_front = 1.0 if -4.0 <= local_x <= 55.0 and abs(local_y) <= 25.0 else 0.0
    crossing = 1.0 if "cross" in heading or motion == "turning" else 0.0
    lateral_dir = str(geometry.get("lateral_direction") or "straight")
    side_match = 0.0
    if lateral_dir == "left" and route_side == "left_of_route":
        side_match = 1.0
    elif lateral_dir == "right" and route_side == "right_of_route":
        side_match = 1.0
    maneuver_boost = 1.0 if geometry.get("lateral_maneuver_like") or geometry.get("turn_like") else 0.25
    score = (
        0.24 * _visibility_score(rec)
        + 0.22 * _front_lateral_score(rec)
        + 0.16 * _route_distance_score(rec, 30.0)
        + 0.12 * _motion_score(rec)
        + 0.12 * near_front
        + 0.08 * crossing
        + 0.06 * side_match * maneuver_boost
    )
    return _clip(score, 0.0, 1.0)


def _broad_l1_prior(records: Sequence[Dict[str, Any]]) -> float:
    _selected, summary = select_evidence_records(records, top_k=12)
    return float(summary.get("v2x_relevance_prior", 0.0))


def _variant_scores(
    annotated: Sequence[Dict[str, Any]],
    *,
    variant: str,
    geometry: Dict[str, Any],
    broad_prior: float,
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for rec in annotated:
        soft = _corridor_soft_score(rec)
        risk = _risk_zone_score(rec, geometry)
        if variant == "corridor_soft":
            final = soft
        elif variant in {"risk_zone", "risk_zone_lateral_protected"}:
            final = risk
        elif variant == "hybrid":
            final = _clip(0.55 * soft + 0.30 * risk + 0.15 * broad_prior, 0.0, 1.0)
        else:
            final = _safe_float(rec.get("route_relevance_score"), 0.0)
        out = dict(rec)
        out["route_soft_score"] = soft
        out["route_risk_score"] = risk
        out["route_hybrid_score"] = _clip(0.55 * soft + 0.30 * risk + 0.15 * broad_prior, 0.0, 1.0)
        out["route_variant_score"] = final
        scored.append(out)
    return scored


def _route_rank_key(rec: Dict[str, Any]) -> Tuple[float, float, float, float]:
    route_dist = _safe_float(rec.get("route_corridor_distance_m"), 1e9)
    closest_step = _safe_float(rec.get("route_closest_step"), 99.0)
    distance = _safe_float(rec.get("distance_m"), 1e9)
    relevance = _safe_float(rec.get("route_relevance_score"), 0.0)
    visibility = str(rec.get("visibility_source") or "unknown")
    visibility_penalty = 0.0 if visibility == "infra_only" else 0.5 if visibility == "both" else 1.0
    rear_penalty = 2.0 if str(rec.get("route_ahead_behind_bin")) == "behind" else 0.0
    return (
        route_dist + 0.5 * closest_step + visibility_penalty + rear_penalty - relevance,
        route_dist,
        closest_step,
        distance,
    )


def _variant_rank_key(rec: Dict[str, Any]) -> Tuple[float, float, float, float]:
    return (
        -_safe_float(rec.get("route_variant_score"), 0.0),
        _safe_float(rec.get("route_corridor_distance_m"), 1e9),
        _safe_float(rec.get("distance_m"), 1e9),
        _safe_float(rec.get("route_closest_step"), 99.0),
    )


def _variant_prior(
    scored: Sequence[Dict[str, Any]],
    *,
    variant: str,
    selected: Sequence[Dict[str, Any]],
    broad_prior: float,
) -> Tuple[float, int, int, Optional[float]]:
    if variant == "strict":
        infra_count = sum(
            1
            for rec in scored
            if rec.get("visibility_source") == "infra_only"
            and is_nonparked(rec)
            and bool(rec.get("route_within_corridor"))
            and str(rec.get("route_ahead_behind_bin")) in {"near_route", "ahead_route"}
        )
        shared_count = sum(
            1
            for rec in scored
            if rec.get("visibility_source") == "both"
            and is_nonparked(rec)
            and bool(rec.get("route_within_corridor"))
        )
        prior = 0.9 if infra_count > 0 else 0.55 if shared_count > 0 else 0.0
        nearest = None
        for rec in scored:
            if rec.get("visibility_source") != "infra_only" or not is_nonparked(rec):
                continue
            dist = _safe_float(rec.get("route_corridor_distance_m"), 1e9)
            nearest = dist if nearest is None else min(nearest, dist)
        return prior, infra_count, shared_count, nearest

    threshold = {
        "corridor_soft": 0.58,
        "risk_zone": 0.58,
        "risk_zone_lateral_protected": 0.58,
        "hybrid": 0.53,
    }.get(variant, 0.58)
    candidates = [
        rec
        for rec in selected
        if is_nonparked(rec)
        and rec.get("visibility_source") in {"infra_only", "both"}
        and _safe_float(rec.get("route_variant_score"), 0.0) >= threshold
    ]
    if not candidates and variant == "hybrid" and broad_prior > 0.0:
        candidates = [
            rec
            for rec in selected
            if is_nonparked(rec)
            and rec.get("visibility_source") in {"infra_only", "both"}
            and _safe_float(rec.get("route_variant_score"), 0.0) >= 0.48
        ]
    infra_count = sum(1 for rec in candidates if rec.get("visibility_source") == "infra_only")
    shared_count = sum(1 for rec in candidates if rec.get("visibility_source") == "both")
    nearest = None
    for rec in candidates:
        if rec.get("visibility_source") != "infra_only":
            continue
        dist = _safe_float(rec.get("route_corridor_distance_m"), 1e9)
        nearest = dist if nearest is None else min(nearest, dist)
    if not candidates:
        return 0.0, 0, 0, nearest
    prior = max(_safe_float(rec.get("route_variant_score"), 0.0) for rec in candidates)
    if infra_count > 0:
        prior = max(prior, 0.62)
    elif shared_count > 0:
        prior = max(prior, 0.50)
    return _clip(prior, 0.0, 0.95), infra_count, shared_count, nearest


def select_route_conditioned_evidence_records(
    records: Sequence[Dict[str, Any]],
    base_waypoints: Sequence[Sequence[float]],
    *,
    top_k: int = 12,
    corridor_width_m: float = 6.0,
    near_horizon_steps: int = 6,
    relevance_variant: str = "strict",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    variant = str(relevance_variant or "strict").lower()
    if variant not in {"strict", "corridor_soft", "risk_zone", "risk_zone_lateral_protected", "hybrid"}:
        raise ValueError(f"Unsupported route relevance variant: {relevance_variant}")
    route_points = _route_points_from_waypoints(base_waypoints, near_horizon_steps)
    geometry = _trajectory_geometry(base_waypoints)
    broad_prior = _broad_l1_prior(records)
    annotated = [
        _annotate_route_record(rec, route_points, corridor_width_m=corridor_width_m)
        for rec in records
    ]
    scored = _variant_scores(annotated, variant=variant, geometry=geometry, broad_prior=broad_prior)
    selected: List[Dict[str, Any]] = []
    used = set()

    def add(pool: Iterable[Dict[str, Any]], role: str, limit: int) -> None:
        nonlocal selected
        role_count = 0
        for rec in pool:
            if len(selected) >= top_k:
                return
            idx = int(rec.get("l1_index", -1))
            if idx in used:
                continue
            new_rec = dict(rec)
            new_rec["route_evidence_role"] = role
            selected.append(new_rec)
            used.add(idx)
            role_count += 1
            if role_count >= limit or len(selected) >= top_k:
                return

    if variant == "strict":
        route_pool = [
            rec
            for rec in scored
            if is_nonparked(rec)
            and bool(rec.get("route_within_corridor"))
            and str(rec.get("route_ahead_behind_bin")) in {"near_route", "ahead_route"}
        ]
        infra_pool = sorted(
            [rec for rec in route_pool if rec.get("visibility_source") == "infra_only"],
            key=_route_rank_key,
        )
        add(infra_pool, "infra_route_constraint", min(7, top_k))
        shared_pool = sorted(
            [rec for rec in route_pool if rec.get("visibility_source") == "both"],
            key=_route_rank_key,
        )
        add(shared_pool, "shared_route_context", min(3, top_k - len(selected)))
        ego_pool = sorted(
            [rec for rec in route_pool if rec.get("visibility_source") in {"ego_only", "unknown"}],
            key=_route_rank_key,
        )
        add(ego_pool, "ego_route_anchor", min(4, top_k - len(selected)))
        fill_pool = sorted([rec for rec in scored if is_nonparked(rec)], key=_route_rank_key)
        add(fill_pool, "route_context_fill", top_k)
    else:
        route_pool = [rec for rec in scored if is_nonparked(rec)]
        infra_pool = sorted(
            [rec for rec in route_pool if rec.get("visibility_source") == "infra_only"],
            key=_variant_rank_key,
        )
        add(infra_pool, "infra_route_constraint", min(6, top_k))
        shared_pool = sorted(
            [rec for rec in route_pool if rec.get("visibility_source") == "both"],
            key=_variant_rank_key,
        )
        add(shared_pool, "shared_route_context", min(4, top_k - len(selected)))
        ego_pool = sorted(
            [rec for rec in route_pool if rec.get("visibility_source") in {"ego_only", "unknown"}],
            key=_variant_rank_key,
        )
        add(ego_pool, "ego_route_anchor", min(3, top_k - len(selected)))
        fill_pool = sorted(route_pool, key=_variant_rank_key)
        add(fill_pool, "route_context_fill", top_k)

    relevance_prior, infra_route_count, shared_route_count, nearest_infra_route = _variant_prior(
        scored,
        variant=variant,
        selected=selected,
        broad_prior=broad_prior,
    )
    lateral_shape = lateral_shape_protection_from_waypoints(base_waypoints)

    summary = {
        "top_k": top_k,
        "route_relevance_variant": variant,
        "corridor_width_m": float(corridor_width_m),
        "near_horizon_steps": int(near_horizon_steps),
        "selected_count": len(selected),
        "infra_route_constraint_selected": sum(
            r.get("route_evidence_role") == "infra_route_constraint" for r in selected
        ),
        "shared_route_context_selected": sum(
            r.get("route_evidence_role") == "shared_route_context" for r in selected
        ),
        "ego_route_anchor_selected": sum(
            r.get("route_evidence_role") == "ego_route_anchor" for r in selected
        ),
        "route_context_fill_selected": sum(
            r.get("route_evidence_role") == "route_context_fill" for r in selected
        ),
        "infra_only_route_relevant_count": infra_route_count,
        "shared_route_relevant_count": shared_route_count,
        "nearest_infra_route_corridor_distance": nearest_infra_route,
        "broad_l1_relevance_prior": broad_prior,
        "route_relevance_prior": relevance_prior,
        "v2x_relevance_prior": relevance_prior,
        "route_geometry": geometry,
        **lateral_shape,
        "selected_l1_indices": [int(r.get("l1_index", -1)) for r in selected],
        "route_points_forward_lateral": [
            [round(float(forward), 3), round(float(lateral), 3)]
            for forward, lateral in route_points
        ],
    }
    return selected[:top_k], summary


def encode_record(rec: Dict[str, Any]) -> List[float]:
    local_x = _safe_float(rec.get("local_x"), 0.0)
    local_y = _safe_float(rec.get("local_y"), 0.0)
    distance = _safe_float(rec.get("distance_m"), math.hypot(local_x, local_y))
    instant_speed = _safe_float(rec.get("instant_speed_mps"), 0.0)
    smoothed_speed = _safe_float(rec.get("smoothed_speed_mps"), instant_speed)
    yaw_change = _safe_float(rec.get("yaw_change_deg"), 0.0)
    numeric = [
        _clip(local_x / 80.0, -2.0, 2.0),
        _clip(local_y / 40.0, -2.0, 2.0),
        _clip(distance / 100.0, 0.0, 3.0),
        _clip(instant_speed / 20.0, 0.0, 3.0),
        _clip(smoothed_speed / 20.0, 0.0, 3.0),
        _clip(yaw_change / 90.0, -2.0, 2.0),
        1.0 if local_x > 5.0 else 0.0,
        1.0 if abs(local_y) > 2.0 and local_x >= -5.0 else 0.0,
        1.0 if local_x < -5.0 else 0.0,
        1.0 if is_nonparked(rec) else 0.0,
    ]
    feat = (
        numeric
        + _one_hot(str(rec.get("category") or "object"), CATEGORY_VOCAB)
        + _one_hot(str(rec.get("visibility_source") or "unknown"), VISIBILITY_VOCAB)
        + _one_hot(str(rec.get("motion_state_refined") or "unknown"), MOTION_VOCAB)
        + _one_hot(str(rec.get("heading_refined") or "unknown"), HEADING_VOCAB)
        + _one_hot(str(rec.get("position_bin") or "unknown"), POSITION_VOCAB)
        + _one_hot(str(rec.get("evidence_role") or "unused"), ROLE_VOCAB)
    )
    if len(feat) != FEATURE_DIM:
        raise RuntimeError(f"Evidence feature dim mismatch: got {len(feat)}, expected {FEATURE_DIM}")
    return [float(x) for x in feat]


def encode_route_record(rec: Dict[str, Any]) -> List[float]:
    local_x = _safe_float(rec.get("local_x"), 0.0)
    local_y = _safe_float(rec.get("local_y"), 0.0)
    distance = _safe_float(rec.get("distance_m"), math.hypot(local_x, local_y))
    instant_speed = _safe_float(rec.get("instant_speed_mps"), 0.0)
    smoothed_speed = _safe_float(rec.get("smoothed_speed_mps"), instant_speed)
    yaw_change = _safe_float(rec.get("yaw_change_deg"), 0.0)
    route_dist = _safe_float(rec.get("route_corridor_distance_m"), 100.0)
    route_step = _safe_float(rec.get("route_closest_step"), 0.0)
    route_forward = _safe_float(rec.get("route_closest_forward_m"), 0.0)
    route_relevance = _safe_float(rec.get("route_relevance_score"), 0.0)
    numeric = [
        _clip(local_x / 80.0, -2.0, 2.0),
        _clip(local_y / 40.0, -2.0, 2.0),
        _clip(distance / 100.0, 0.0, 3.0),
        _clip(instant_speed / 20.0, 0.0, 3.0),
        _clip(smoothed_speed / 20.0, 0.0, 3.0),
        _clip(yaw_change / 90.0, -2.0, 2.0),
        1.0 if local_x > 5.0 else 0.0,
        1.0 if abs(local_y) > 2.0 and local_x >= -5.0 else 0.0,
        1.0 if local_x < -5.0 else 0.0,
        1.0 if is_nonparked(rec) else 0.0,
        _clip(route_dist / 20.0, 0.0, 5.0),
        _clip(route_step / 5.0, 0.0, 1.0),
        1.0 if bool(rec.get("route_within_corridor")) else 0.0,
        _clip((local_x - route_forward) / 40.0, -2.0, 2.0),
        _clip(route_relevance, 0.0, 1.0),
    ]
    feat = (
        numeric
        + _one_hot(str(rec.get("category") or "object"), CATEGORY_VOCAB)
        + _one_hot(str(rec.get("visibility_source") or "unknown"), VISIBILITY_VOCAB)
        + _one_hot(str(rec.get("motion_state_refined") or "unknown"), MOTION_VOCAB)
        + _one_hot(str(rec.get("heading_refined") or "unknown"), HEADING_VOCAB)
        + _one_hot(str(rec.get("position_bin") or "unknown"), POSITION_VOCAB)
        + _one_hot(str(rec.get("route_side_of_route") or "unknown"), ROUTE_SIDE_VOCAB)
        + _one_hot(str(rec.get("route_ahead_behind_bin") or "unknown"), ROUTE_AHEAD_VOCAB)
        + _one_hot(str(rec.get("route_visibility_role") or "unknown_route"), ROUTE_VISIBILITY_ROLE_VOCAB)
        + _one_hot(str(rec.get("route_evidence_role") or "unused"), ROUTE_EVIDENCE_ROLE_VOCAB)
    )
    if len(feat) != ROUTE_FEATURE_DIM:
        raise RuntimeError(
            f"Route evidence feature dim mismatch: got {len(feat)}, expected {ROUTE_FEATURE_DIM}"
        )
    return [float(x) for x in feat]


def build_l1_evidence(
    split: str,
    token: str,
    *,
    top_k: int = 12,
    l1_dir: Path = DEFAULT_L1_DIR,
    refined_dir: Path = DEFAULT_L1_REFINED_DIR,
) -> Dict[str, Any]:
    records, meta = load_l1_records(split, token, l1_dir=l1_dir, refined_dir=refined_dir)
    selected, summary = select_evidence_records(records, top_k=top_k)
    tokens = [[0.0] * FEATURE_DIM for _ in range(top_k)]
    mask = [0.0] * top_k
    public_records: List[Dict[str, Any]] = []
    for idx, rec in enumerate(selected[:top_k]):
        tokens[idx] = encode_record(rec)
        mask[idx] = 1.0
        public_records.append(
            {
                "l1_index": rec.get("l1_index"),
                "category": rec.get("category"),
                "visibility_source": rec.get("visibility_source"),
                "distance_m": round(_safe_float(rec.get("distance_m"), 0.0), 3),
                "local_xy": [
                    round(_safe_float(rec.get("local_x"), 0.0), 3),
                    round(_safe_float(rec.get("local_y"), 0.0), 3),
                ],
                "position_bin": rec.get("position_bin"),
                "motion_state_refined": rec.get("motion_state_refined"),
                "heading_refined": rec.get("heading_refined"),
                "instant_speed_mps": round(_safe_float(rec.get("instant_speed_mps"), 0.0), 3),
                "smoothed_speed_mps": round(_safe_float(rec.get("smoothed_speed_mps"), 0.0), 3),
                "evidence_role": rec.get("evidence_role"),
            }
        )
    return {
        "tokens": tokens,
        "mask": mask,
        "feature_dim": FEATURE_DIM,
        "v2x_relevance_prior": float(summary["v2x_relevance_prior"]),
        "summary": {**meta, **summary, "selected_records": public_records},
        "provenance": "l1_perception_plus_l1_motion_refined_current_frame_only",
    }


def build_route_conditioned_l1_evidence(
    split: str,
    token: str,
    base_waypoints: Sequence[Sequence[float]],
    *,
    top_k: int = 12,
    corridor_width_m: float = 6.0,
    near_horizon_steps: int = 6,
    relevance_variant: str = "strict",
    l1_dir: Path = DEFAULT_L1_DIR,
    refined_dir: Path = DEFAULT_L1_REFINED_DIR,
) -> Dict[str, Any]:
    records, meta = load_l1_records(split, token, l1_dir=l1_dir, refined_dir=refined_dir)
    selected, summary = select_route_conditioned_evidence_records(
        records,
        base_waypoints,
        top_k=top_k,
        corridor_width_m=corridor_width_m,
        near_horizon_steps=near_horizon_steps,
        relevance_variant=relevance_variant,
    )
    tokens = [[0.0] * ROUTE_FEATURE_DIM for _ in range(top_k)]
    mask = [0.0] * top_k
    public_records: List[Dict[str, Any]] = []
    for idx, rec in enumerate(selected[:top_k]):
        tokens[idx] = encode_route_record(rec)
        mask[idx] = 1.0
        public_records.append(
            {
                "l1_index": rec.get("l1_index"),
                "category": rec.get("category"),
                "visibility_source": rec.get("visibility_source"),
                "distance_m": round(_safe_float(rec.get("distance_m"), 0.0), 3),
                "local_xy": [
                    round(_safe_float(rec.get("local_x"), 0.0), 3),
                    round(_safe_float(rec.get("local_y"), 0.0), 3),
                ],
                "position_bin": rec.get("position_bin"),
                "motion_state_refined": rec.get("motion_state_refined"),
                "heading_refined": rec.get("heading_refined"),
                "instant_speed_mps": round(_safe_float(rec.get("instant_speed_mps"), 0.0), 3),
                "smoothed_speed_mps": round(_safe_float(rec.get("smoothed_speed_mps"), 0.0), 3),
                "route_corridor_distance_m": round(
                    _safe_float(rec.get("route_corridor_distance_m"), 0.0),
                    3,
                ),
                "route_closest_step": int(rec.get("route_closest_step", 0)),
                "route_side_of_route": rec.get("route_side_of_route"),
                "route_ahead_behind_bin": rec.get("route_ahead_behind_bin"),
                "route_visibility_role": rec.get("route_visibility_role"),
                "route_evidence_role": rec.get("route_evidence_role"),
                "route_relevance_score": round(_safe_float(rec.get("route_relevance_score"), 0.0), 4),
                "route_variant_score": round(_safe_float(rec.get("route_variant_score"), 0.0), 4),
                "route_soft_score": round(_safe_float(rec.get("route_soft_score"), 0.0), 4),
                "route_risk_score": round(_safe_float(rec.get("route_risk_score"), 0.0), 4),
            }
        )
    return {
        "tokens": tokens,
        "mask": mask,
        "feature_dim": ROUTE_FEATURE_DIM,
        "v2x_relevance_prior": float(summary["v2x_relevance_prior"]),
        "route_relevance_prior": float(summary["route_relevance_prior"]),
        "summary": {**meta, **summary, "selected_records": public_records},
        "provenance": (
            "l1_perception_plus_l1_motion_refined_current_frame_only;"
            "route_corridor_from_frozen_ego_base_waypoints"
        ),
    }
