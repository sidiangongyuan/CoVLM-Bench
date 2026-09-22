#!/usr/bin/env python3
"""Audit and optionally enrich CoVLM index rows with leak-free ego status."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_V2X_ROOT = Path("data/V2X-Seq-SPD")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def vehicle_records(v2x_root: Path) -> List[Dict[str, Any]]:
    path = v2x_root / "vehicle-side" / "data_info.json"
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return [rec for rec in data if isinstance(rec, dict)]


def build_prev_lookup(records: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    by_frame = {str(rec["frame_id"]): rec for rec in records if rec.get("frame_id") is not None}
    by_scene: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if rec.get("sequence_id") is not None:
            by_scene[str(rec["sequence_id"])].append(rec)
    prev_lookup: Dict[str, str] = {}
    for scene_rows in by_scene.values():
        scene_rows.sort(key=lambda rec: (float(rec.get("pointcloud_timestamp", 0.0)), str(rec.get("frame_id", ""))))
        for idx, rec in enumerate(scene_rows):
            frame = str(rec["frame_id"])
            prev_lookup[frame] = str(scene_rows[idx - 1]["frame_id"]) if idx > 0 else ""
    return by_frame, prev_lookup


def load_pose(v2x_root: Path, rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    rel = rec.get("calib_novatel_to_world_path")
    if not rel:
        return None
    path = v2x_root / "vehicle-side" / str(rel)
    if not path.exists():
        return None
    data = load_json(path)
    translation = data.get("translation")
    rotation = data.get("rotation")
    try:
        flat_translation = [float(x[0] if isinstance(x, list) else x) for x in translation]
        rot = [[float(v) for v in row] for row in rotation]
    except Exception:
        return None
    if len(flat_translation) < 2 or len(rot) < 2 or len(rot[0]) < 2:
        return None
    return {"translation": flat_translation, "rotation": rot, "path": str(path)}


def yaw_from_rotation(rot: List[List[float]]) -> float:
    return math.atan2(float(rot[1][0]), float(rot[0][0]))


def wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def compute_status(
    *,
    row: Dict[str, Any],
    by_frame: Dict[str, Dict[str, Any]],
    prev_lookup: Dict[str, str],
    v2x_root: Path,
) -> Tuple[Dict[str, Any], Optional[str]]:
    token = str(row.get("token", ""))
    cur = by_frame.get(token)
    if cur is None:
        return {"available": False, "reason": "missing_current_record"}, "missing_current_record"
    prev_token = prev_lookup.get(token, "")
    if not prev_token:
        return {"available": False, "reason": "missing_previous_frame"}, "missing_previous_frame"
    prev = by_frame.get(prev_token)
    if prev is None:
        return {"available": False, "reason": "missing_previous_record", "prev_token": prev_token}, "missing_previous_record"
    if str(prev.get("sequence_id")) != str(cur.get("sequence_id")):
        return {"available": False, "reason": "scene_boundary", "prev_token": prev_token}, "scene_boundary"
    cur_pose = load_pose(v2x_root, cur)
    prev_pose = load_pose(v2x_root, prev)
    if cur_pose is None:
        return {"available": False, "reason": "missing_current_pose", "prev_token": prev_token}, "missing_current_pose"
    if prev_pose is None:
        return {"available": False, "reason": "missing_previous_pose", "prev_token": prev_token}, "missing_previous_pose"
    try:
        cur_ts = float(cur["pointcloud_timestamp"]) / 1e6
        prev_ts = float(prev["pointcloud_timestamp"]) / 1e6
    except Exception:
        return {"available": False, "reason": "invalid_timestamp", "prev_token": prev_token}, "invalid_timestamp"
    dt = cur_ts - prev_ts
    if not math.isfinite(dt) or dt <= 0:
        return {"available": False, "reason": "non_positive_dt", "prev_token": prev_token, "dt_sec": dt}, "non_positive_dt"
    dx = float(cur_pose["translation"][0]) - float(prev_pose["translation"][0])
    dy = float(cur_pose["translation"][1]) - float(prev_pose["translation"][1])
    speed = math.hypot(dx, dy) / dt
    cur_yaw = yaw_from_rotation(cur_pose["rotation"])
    prev_yaw = yaw_from_rotation(prev_pose["rotation"])
    heading_delta = wrap_angle(cur_yaw - prev_yaw)
    yaw_rate = heading_delta / dt
    status = {
        "available": True,
        "source": "vehicle-side current and previous pointcloud timestamp plus novatel_to_world pose",
        "token": token,
        "prev_token": prev_token,
        "scene_token": row.get("scene_token") or cur.get("sequence_id"),
        "dt_sec": dt,
        "speed_mps": speed,
        "heading_delta_rad": heading_delta,
        "yaw_rate_radps": yaw_rate,
    }
    return status, None


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return float(ordered[pos])


def summarize_status(rows: List[Dict[str, Any]], enriched_rows: List[Dict[str, Any]], missing: Counter[str]) -> Dict[str, Any]:
    speeds = [float(r["ego_status"]["speed_mps"]) for r in enriched_rows if r.get("ego_status", {}).get("available")]
    yaw_rates = [abs(float(r["ego_status"]["yaw_rate_radps"])) for r in enriched_rows if r.get("ego_status", {}).get("available")]
    by_split = Counter(str(r.get("split", "unknown")) for r in rows)
    available_by_split = Counter(
        str(r.get("split", "unknown")) for r in enriched_rows if r.get("ego_status", {}).get("available")
    )
    total = len(rows)
    available = len(speeds)
    return {
        "total_index_rows": total,
        "available_count": available,
        "coverage": available / max(total, 1),
        "missing_count": total - available,
        "missing_reasons": dict(missing),
        "by_split": dict(by_split),
        "available_by_split": dict(available_by_split),
        "speed_mps": {
            "min": min(speeds) if speeds else 0.0,
            "mean": sum(speeds) / max(len(speeds), 1),
            "p50": percentile(speeds, 50),
            "p95": percentile(speeds, 95),
            "max": max(speeds) if speeds else 0.0,
        },
        "abs_yaw_rate_radps": {
            "mean": sum(yaw_rates) / max(len(yaw_rates), 1),
            "p95": percentile(yaw_rates, 95),
            "max": max(yaw_rates) if yaw_rates else 0.0,
        },
    }


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    summary = payload["summary"]
    leak = payload["leakage_assessment"]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Ego-Status Audit",
        "",
        f"- Created UTC: `{payload['created_at_utc']}`",
        f"- Index: `{payload['index_path']}`",
        f"- V2X root: `{payload['v2x_root']}`",
        f"- Leak-free: `{leak['leak_free']}`",
        f"- Safe minimal signal: `{payload['safe_minimal_signal']['definition']}`",
        "",
        "## Coverage",
        f"- Total rows: {summary['total_index_rows']}",
        f"- Available: {summary['available_count']} ({summary['coverage']:.4f})",
        f"- Missing: {summary['missing_count']}",
        f"- Missing reasons: `{summary['missing_reasons']}`",
        f"- Available by split: `{summary['available_by_split']}`",
        "",
        "## Distribution",
        f"- Speed mean/p50/p95/max: {summary['speed_mps']['mean']:.3f} / {summary['speed_mps']['p50']:.3f} / {summary['speed_mps']['p95']:.3f} / {summary['speed_mps']['max']:.3f} m/s",
        f"- Absolute yaw-rate mean/p95/max: {summary['abs_yaw_rate_radps']['mean']:.3f} / {summary['abs_yaw_rate_radps']['p95']:.3f} / {summary['abs_yaw_rate_radps']['max']:.3f} rad/s",
        "",
        "## Leakage Assessment",
        f"- Inputs used: {', '.join(leak['inputs_used'])}",
        f"- Inputs not used: {', '.join(leak['inputs_not_used'])}",
        f"- Risk note: {leak['risk_note']}",
        "",
        "## Output",
        f"- Enriched index: `{payload.get('enriched_index_path')}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit leak-free ego status from current/previous SPD ego pose.")
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--v2x-root", default=str(DEFAULT_V2X_ROOT), type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--enriched-index", default=None, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.index)
    records = vehicle_records(args.v2x_root)
    by_frame, prev_lookup = build_prev_lookup(records)
    enriched_rows: List[Dict[str, Any]] = []
    missing: Counter[str] = Counter()
    examples: List[Dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        status, reason = compute_status(row=row, by_frame=by_frame, prev_lookup=prev_lookup, v2x_root=args.v2x_root)
        new_row["ego_status"] = status
        enriched_rows.append(new_row)
        if reason:
            missing[reason] += 1
        elif len(examples) < 5:
            examples.append(
                {
                    "sample_id": row.get("sample_id"),
                    "token": row.get("token"),
                    "split": row.get("split"),
                    "ego_status": status,
                }
            )
    enriched_index = args.enriched_index
    if enriched_index is not None:
        write_jsonl(enriched_index, enriched_rows)
    payload = {
        "created_at_utc": utc_now(),
        "index_path": str(args.index),
        "v2x_root": str(args.v2x_root),
        "enriched_index_path": str(enriched_index) if enriched_index is not None else None,
        "summary": summarize_status(rows, enriched_rows, missing),
        "safe_minimal_signal": {
            "definition": "current speed, heading delta/yaw-rate, and availability flag from current and previous ego pose only",
            "fields": ["available", "speed_mps", "heading_delta_rad", "yaw_rate_radps", "dt_sec"],
        },
        "leakage_assessment": {
            "leak_free": True,
            "inputs_used": [
                "current frame token",
                "previous frame token from same sequence",
                "current and previous pointcloud timestamps",
                "current and previous vehicle-side novatel_to_world pose",
            ],
            "inputs_not_used": [
                "future waypoints",
                "future frame pose",
                "target-derived motion summaries",
                "ground-truth command",
                "generated CoT",
            ],
            "risk_note": "The signal is causal under the current protocol because it uses only current and previous ego pose/timestamp. Missing first-in-sequence rows carry availability=false.",
        },
        "examples": examples,
    }
    write_json(args.output_json, payload)
    write_markdown(args.output_md, payload)
    print(
        "[DONE] Ego-status audit saved to {}; coverage={:.4f}; enriched_index={}".format(
            args.output_json,
            payload["summary"]["coverage"],
            enriched_index,
        )
    )


if __name__ == "__main__":
    main()
