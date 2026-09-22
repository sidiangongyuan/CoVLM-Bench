#!/usr/bin/env python3
"""Build a JSONL index for CoVLM-Bench L4 CoT samples.

Example:
    python -m projects.covla_baseline.data.build_index \
    --l4-dir data/covlm_bench/l4_cot_v3_full_latest \
    --output output/covla_baseline/index_l4_full_fixed.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_COVLA_ROOT = Path("data/covlm_bench")
DEFAULT_V2X_ROOT = Path("data/V2X-Seq-SPD")
DEFAULT_L4_DIR = DEFAULT_COVLA_ROOT / "l4_cot_v3_full_latest"
DEFAULT_OUTPUT = Path("output/covla_baseline/index_l4_full_fixed.jsonl")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_json_files(l4_dir: Path, limit: Optional[int] = None) -> Iterable[Path]:
    files = sorted(p for p in l4_dir.glob("*.json") if p.is_file())
    if limit is not None:
        files = files[:limit]
    return files


def first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def get_nested(data: Dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def infer_token_and_split(path: Path, data: Dict[str, Any]) -> Tuple[str, str, str]:
    sample_id = path.stem
    split = str(first_non_empty(data.get("split"), sample_id.split("_")[0] if "_" in sample_id else "train"))
    token = str(first_non_empty(data.get("token"), sample_id.split("_", 1)[-1]))
    return sample_id, token, split


def normalize_waypoints(raw: Any) -> Optional[List[List[float]]]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        raw = first_non_empty(raw.get("waypoints"), raw.get("trajectory"), raw.get("future_waypoints"))
    if not isinstance(raw, list):
        return None
    waypoints: List[List[float]] = []
    for item in raw:
        if isinstance(item, dict):
            x = first_non_empty(item.get("x"), item.get("dx"), item.get("lateral"))
            y = first_non_empty(item.get("y"), item.get("dy"), item.get("longitudinal"))
            if x is None or y is None:
                return None
            waypoints.append([float(x), float(y)])
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            waypoints.append([float(item[0]), float(item[1])])
        else:
            return None
    if len(waypoints) != 6:
        if len(waypoints) == 0:
            return None
        if len(waypoints) == 1:
            # End-of-sequence samples may keep only one valid future point.  To
            # preserve the non-empty GT action for the 6-step baseline head,
            # resample a straight segment from the current origin to that point.
            x, y = waypoints[0]
            waypoints = [[x * (i + 1) / 6.0, y * (i + 1) / 6.0] for i in range(6)]
            return waypoints
        # Some pilot L4 files contain 5-step trajectories.  The baseline always
        # trains a 6-step head, so resample in normalized time instead of
        # dropping otherwise valid non-empty samples.
        src_count = len(waypoints)
        resampled: List[List[float]] = []
        for i in range(6):
            pos = i * (src_count - 1) / 5.0
            lo = int(pos)
            hi = min(lo + 1, src_count - 1)
            ratio = pos - lo
            x = waypoints[lo][0] * (1.0 - ratio) + waypoints[hi][0] * ratio
            y = waypoints[lo][1] * (1.0 - ratio) + waypoints[hi][1] * ratio
            resampled.append([float(x), float(y)])
        waypoints = resampled
    return waypoints


def format_waypoints(waypoints: List[List[float]]) -> str:
    return "[" + ", ".join(f"[{x:.2f}, {y:.2f}]" for x, y in waypoints) + "]"


def strip_text_waypoint_lines(text: Any) -> str:
    """Remove text waypoint sections without touching other reviewed prose."""
    lines = str(text or "").splitlines()
    kept = [line for line in lines if not line.strip().lower().startswith("waypoints")]
    return "\n".join(kept).strip()


def format_l4_target(
    data: Dict[str, Any],
    command: str,
    waypoints: List[List[float]],
    target_parts: int = 4,
) -> str:
    if target_parts not in {3, 4}:
        raise ValueError(f"target_parts must be 3 or 4, got {target_parts}")

    l4 = data.get("l4_cot", {}) if isinstance(data.get("l4_cot"), dict) else {}
    action = l4.get("action", {}) if isinstance(l4.get("action"), dict) else {}

    scene = first_non_empty(l4.get("scene_overview"), data.get("scene_overview"), "")
    critical = first_non_empty(l4.get("critical_objects_v2x"), data.get("critical_objects_v2x"), "")
    reasoning = first_non_empty(l4.get("decision_reasoning"), data.get("decision_reasoning"), "")
    action_text = strip_text_waypoint_lines(
        first_non_empty(action.get("natural_language"), get_nested(data, "gt_action", "natural_language"), "")
    )

    parts = [
        "Part 1 - Scene overview:\n" + str(scene).strip(),
        "Part 2 - V2X-aware critical objects:\n" + str(critical).strip(),
        "Part 3 - Decision reasoning:\n" + str(reasoning).strip(),
    ]
    if target_parts == 4:
        parts.append(
            "Part 4 - Action:\n"
            + str(action_text).strip()
            + "\nCommand: "
            + command
        )
    return "\n\n".join(parts).strip() + "\n"


def load_cooperative_lookup(v2x_root: Path) -> Dict[str, Dict[str, Any]]:
    path = v2x_root / "cooperative" / "data_info.json"
    if not path.exists():
        return {}
    try:
        data = load_json(path)
    except Exception as exc:  # pragma: no cover - best-effort fallback only
        print(f"[WARN] Failed to parse {path}: {exc}", file=sys.stderr)
        return {}
    records = data if isinstance(data, list) else list(data.values()) if isinstance(data, dict) else []
    lookup: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        keys = [
            rec.get("token"),
            rec.get("vehicle_frame"),
            rec.get("vehicle_frame_id"),
            rec.get("veh_frame_id"),
            rec.get("frame_id"),
        ]
        for key in keys:
            if key is not None:
                lookup[str(key)] = rec
    return lookup


def fallback_image_paths(token: str, rec: Optional[Dict[str, Any]], v2x_root: Path) -> Tuple[Optional[str], Optional[str]]:
    rec = rec or {}
    ego_frame = first_non_empty(
        rec.get("vehicle_frame"), rec.get("vehicle_frame_id"), rec.get("veh_frame_id"), rec.get("frame_id"), token
    )
    infra_frame = first_non_empty(
        rec.get("infrastructure_frame"), rec.get("infrastructure_frame_id"), rec.get("inf_frame_id"), rec.get("infra_frame_id")
    )

    ego_path = v2x_root / "vehicle-side" / "image" / f"{ego_frame}.jpg"
    infra_path = v2x_root / "infrastructure-side" / "image" / f"{infra_frame}.jpg" if infra_frame else None
    return str(ego_path) if ego_path.exists() else None, str(infra_path) if infra_path and infra_path.exists() else None


def parse_one_file(
    path: Path,
    coop_lookup: Dict[str, Dict[str, Any]],
    v2x_root: Path,
    target_parts: int = 4,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    data = load_json(path)
    if not isinstance(data, dict):
        return None, ["top-level JSON is not an object"]

    sample_id, token, split = infer_token_and_split(path, data)
    rec = coop_lookup.get(token)

    ego_fallback, infra_fallback = fallback_image_paths(token, rec, v2x_root)
    ego_image = first_non_empty(data.get("ego_image"), get_nested(data, "images", "ego"), ego_fallback)
    infra_image = first_non_empty(data.get("infra_image"), get_nested(data, "images", "infra"), infra_fallback)

    gt_action = data.get("gt_action", {}) if isinstance(data.get("gt_action"), dict) else {}
    l4_action = get_nested(data, "l4_cot", "action") or {}
    command = str(first_non_empty(gt_action.get("command"), l4_action.get("command"), data.get("command"), "UNKNOWN"))
    waypoints = normalize_waypoints(first_non_empty(gt_action.get("waypoints"), l4_action.get("waypoints"), data.get("waypoints")))

    missing: List[str] = []
    if not ego_image:
        missing.append("ego_image")
    if not infra_image:
        missing.append("infra_image")
    if waypoints is None:
        missing.append("6-step waypoints")
    if command == "UNKNOWN":
        missing.append("command")
    if missing:
        return None, missing

    target_text = format_l4_target(data, command, waypoints, target_parts=target_parts)
    item = {
        "sample_id": sample_id,
        "token": token,
        "split": split,
        "scene_token": first_non_empty(data.get("scene_token"), data.get("scene"), rec.get("scene_id") if rec else None),
        "ego_image": str(ego_image),
        "infra_image": str(infra_image),
        "l4_path": str(path),
        "l3_path": first_non_empty(get_nested(data, "input_bundle", "l3_path"), data.get("l3_path")),
        "target_text": target_text,
        "command": command,
        "waypoints": waypoints,
        "horizon_sec": first_non_empty(gt_action.get("horizon_sec"), l4_action.get("horizon_sec"), 3.0),
        "critical_object_ids": data.get("critical_object_ids", []),
        "infra_only_critical_ids": data.get("infra_only_critical_ids", []),
    }
    return item, []


def build_index(
    l4_dir: Path,
    output: Path,
    v2x_root: Path,
    limit: Optional[int] = None,
    fail_on_error: bool = False,
    target_parts: int = 4,
) -> Dict[str, int]:
    if not l4_dir.exists():
        raise FileNotFoundError(f"L4 directory does not exist: {l4_dir}")
    if target_parts not in {3, 4}:
        raise ValueError(f"target_parts must be 3 or 4, got {target_parts}")
    output.parent.mkdir(parents=True, exist_ok=True)
    coop_lookup = load_cooperative_lookup(v2x_root)

    stats = {"written": 0, "skipped": 0, "train": 0, "val": 0}
    with output.open("w", encoding="utf-8") as f:
        for path in iter_json_files(l4_dir, limit=limit):
            try:
                item, missing = parse_one_file(
                    path,
                    coop_lookup,
                    v2x_root,
                    target_parts=target_parts,
                )
            except Exception as exc:
                item, missing = None, [f"parse exception: {exc}"]
            if item is None:
                stats["skipped"] += 1
                msg = f"[WARN] Skip {path}: missing/invalid {', '.join(missing)}"
                print(msg, file=sys.stderr)
                if fail_on_error:
                    raise RuntimeError(msg)
                continue
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            stats["written"] += 1
            if item["split"] in stats:
                stats[item["split"]] += 1
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CoVLM-Bench L4 JSONL index for the Qwen-VL baseline.")
    parser.add_argument("--l4-dir", type=Path, default=DEFAULT_L4_DIR, help="Directory containing L4 CoT JSON files.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL index path.")
    parser.add_argument("--v2x-root", type=Path, default=DEFAULT_V2X_ROOT, help="V2X-Seq-SPD-New root for fallback image paths.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of JSON files to parse.")
    parser.add_argument("--fail-on-error", action="store_true", help="Raise if any sample cannot be parsed.")
    parser.add_argument(
        "--target-parts",
        type=int,
        choices=(3, 4),
        default=4,
        help="Number of CoT parts in target_text; action metadata remains structured either way.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_index(
        args.l4_dir,
        args.output,
        args.v2x_root,
        args.limit,
        args.fail_on_error,
        args.target_parts,
    )
    print(json.dumps({"output": str(args.output), **stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
