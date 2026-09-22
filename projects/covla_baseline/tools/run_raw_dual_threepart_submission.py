#!/usr/bin/env python3
"""Run the raw dual-view three-part submission matrix for CoVLM-Bench."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import yaml

from projects.covla_baseline.data.build_index import build_index
from projects.covla_baseline.three_part_cot import parse_three_part_cot
from projects.covla_baseline.tools import cot_diagnostics
from projects.covla_baseline.tools import evaluate_cot_quality
from projects.covla_baseline.tools.audit_ego_status import (
    build_prev_lookup,
    compute_status,
    summarize_status,
    vehicle_records,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_SPLITS = {"train": 1475, "val": 654}
EXPECTED_ROWS = sum(EXPECTED_SPLITS.values())
EXPECTED_VAL_ROWS = EXPECTED_SPLITS["val"]
EXPECTED_SOURCE_ANCHORS = 513
COMMANDS = ("GO_STRAIGHT", "TURN_LEFT", "TURN_RIGHT", "LATERAL_SHIFT", "STOP")
VARIANTS = ("standard", "ego_status")
CONDITIONS = ("normal", "ego_only", "blank", "shuffled")
CONTROL_CONDITIONS = ("ego_only", "blank", "shuffled")
PAPER_HORIZONS_S = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
CANONICAL_COMMANDS = {
    "GO_STRAIGHT",
    "TURN_LEFT",
    "TURN_RIGHT",
    "LATERAL_SHIFT",
    "STOP",
    "SLOW_DOWN",
}
PIPELINE_CONTRACT = "raw_dual_threepart_submission_v1"
PART_RE = re.compile(
    r"(?i)\bpart\s*([1-3])\s*(?:[-:\u2013\u2014]|[.)]|"
    r"\b(?:scene|overview|v2x|critical|decision|reasoning)\b)[^\n]*"
)
NO_V2X_SCHEMA_RE = re.compile(
    r"\b(?:no|zero)\s+"
    r"(?:(?:reliable|useful|additional|decision-relevant|hidden)\s+){0,2}"
    r"(?:infrastructure|v2x|cooperative)(?:[- ]only)?"
    r"(?:\s+(?:critical|hidden|decision-relevant))?\s+"
    r"(?:objects?|evidence|constraints?|information|increment|contribution|risks?)\b"
    r"|\b(?:infrastructure|v2x|cooperative)(?:[- ]only)?\s+"
    r"(?:objects?|evidence|view)\s+(?:is|are|was|were)?\s*"
    r"(?:not\s+available|unavailable|unreliable|absent|none)\b",
    re.I,
)
SOURCE_CONTEXT_RE = re.compile(
    r"\b(?:infrastructure|v2x|cooperative|roadside|second image|cross-view)\b"
    r".{0,120}\b(?:context|confirm(?:s|ed|ing)?|corroborat(?:es|ed|ing)?|confidence)\b",
    re.I,
)
NO_ADDITIONAL_CONSTRAINT_RE = re.compile(
    r"\b(?:no|without)\s+(?:new\s+|additional\s+|further\s+)?"
    r"(?:infrastructure|v2x|cooperative)?(?:[- ]only)?\s*"
    r"(?:object\s+)?(?:evidence|constraint|risk|information|object)?\b"
    r".{0,80}\b(?:trajectory-relevant|decision-relevant|critical|hidden|constraint|risk)\b"
    r"|\b(?:infrastructure|v2x|cooperative)\b.{0,80}\badds?\s+no\s+"
    r"(?:new\s+|additional\s+|further\s+)?(?:critical\s+)?(?:evidence|constraint|risk)",
    re.I,
)
SOURCE_UNAVAILABLE_RE = re.compile(
    r"\b(?:infrastructure|v2x|roadside|second)\s+(?:image|view|input|source)?\s*"
    r"(?:is|was|appears)?\s*(?:unavailable|missing|absent|blank|empty|not provided)\b",
    re.I,
)
SOURCE_MISMATCH_RE = re.compile(
    r"\b(?:infrastructure|v2x|roadside|second|cross-view)\b.{0,120}"
    r"\b(?:mismatch(?:ed)?|inconsistent|unrelated|different scene|"
    r"does not (?:match|correspond)|cannot be aligned)\b",
    re.I,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def ensure_dirs(run_dir: Path) -> None:
    for subdir in (
        "configs",
        "contracts",
        "indexes",
        "logs",
        "metrics",
        "reports",
        "results",
        "samples",
        "tables",
        "status",
    ):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)


def variant_dirs(run_dir: Path, variant: str) -> Dict[str, Path]:
    root = run_dir / variant
    return {
        "root": root,
        "model": root / "model",
        "epoch_checkpoints": root / "epoch_checkpoints",
        "best_checkpoint": root / "best_checkpoint",
        "metrics": root / "metrics",
        "predictions": root / "predictions",
        "samples": root / "samples",
        "logs": root / "logs",
        "status": root / "status",
    }


def sample_identity(row: Mapping[str, Any]) -> Tuple[str, str]:
    return (str(row.get("sample_id") or ""), str(row.get("token") or ""))


def sample_key(row: Mapping[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("token") or "")


def euclidean_xy(left: Sequence[Any], right: Sequence[Any]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def split_rationale_parts(text: str) -> Dict[int, str]:
    matches = list(PART_RE.finditer(text or ""))
    parts: Dict[int, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        parts[int(match.group(1))] = text[start:end].strip()
    return parts


def explicit_no_roadside_increment(text: str) -> bool:
    return bool(
        re.search(
            r"\b(no reliable|no additional|does not provide|unavailable|not reliable|no useful)"
            r".{0,80}\b(infrastructure|v2x|cooperative|second image|constraint|evidence)",
            text,
            re.I,
        )
    )


def positive_roadside_claim(text: str) -> bool:
    if explicit_no_roadside_increment(text):
        return False
    return bool(
        re.search(
            r"\b(infrastructure|v2x|cooperative|second image|overhead|roadside|cross-view)"
            r".{0,100}\b(reveal|adds?|provides?|shows?|confirms?|detects?|identifies?|"
            r"helps?|useful|critical|hidden|occluded|constraint|risk|traffic)",
            text,
            re.I,
        )
    )


def parse_source_role(text: str) -> str:
    parts = split_rationale_parts(text)
    source_text = "\n".join(parts.get(part, "") for part in (2, 3)).strip() or text
    if SOURCE_MISMATCH_RE.search(source_text):
        return "scene_mismatch"
    if SOURCE_UNAVAILABLE_RE.search(source_text):
        return "unavailable"
    no_additional = bool(NO_ADDITIONAL_CONSTRAINT_RE.search(source_text)) or bool(
        NO_V2X_SCHEMA_RE.search(source_text)
    )
    context = bool(SOURCE_CONTEXT_RE.search(source_text))
    decision_claim = not no_additional and positive_roadside_claim(source_text)
    if decision_claim:
        return "decision_relevant"
    if context:
        return "context_only"
    if no_additional or explicit_no_roadside_increment(source_text):
        return "no_additional_evidence"
    return "uncommitted"


def label_split(payload: Mapping[str, Any], path: Path) -> str:
    return str(payload.get("split") or path.stem.split("_", 1)[0])


def audit_labels(label_dir: Path) -> Dict[str, Any]:
    files = sorted(path for path in label_dir.glob("*.json") if path.is_file())
    failed: List[str] = []
    split_counts: Counter[str] = Counter()
    for path in files:
        payload = read_json(path)
        split = label_split(payload, path)
        split_counts[split] += 1
        validation = payload.get("validation")
        if isinstance(validation, Mapping) and validation.get("success") is False:
            failed.append(path.stem)
    passed = len(files) == EXPECTED_ROWS and dict(split_counts) == EXPECTED_SPLITS and not failed
    return {
        "passed": passed,
        "label_dir": str(label_dir),
        "rows": len(files),
        "split_counts": dict(split_counts),
        "validation_failures": failed,
    }


def load_index(path: Path) -> List[Dict[str, Any]]:
    return read_jsonl(path)


def audit_three_part_index(index_path: Path) -> Dict[str, Any]:
    rows = load_index(index_path)
    seen: set[Tuple[str, str]] = set()
    split_counts: Counter[str] = Counter()
    failures: List[Dict[str, Any]] = []
    future_target_pattern_counts: Counter[str] = Counter()
    future_target_leak_count = 0
    for row in rows:
        sample_id, token = sample_identity(row)
        reasons: List[str] = []
        if not sample_id or not token or (sample_id, token) in seen:
            reasons.append("missing_or_duplicate_identity_pair")
        seen.add((sample_id, token))
        split_counts[str(row.get("split"))] += 1
        parsed = parse_three_part_cot(str(row.get("target_text") or ""))
        if not parsed["format_parse_ok"]:
            reasons.extend(str(error) for error in parsed["errors"])
        future_target_leak_count += int(bool(parsed["future_target_leak"]))
        future_target_pattern_counts.update(parsed["future_target_leak_matches"].keys())
        command = row.get("command")
        waypoints = row.get("waypoints")
        valid_waypoints = (
            isinstance(waypoints, list)
            and len(waypoints) == 6
            and all(
                isinstance(point, list)
                and len(point) == 2
                and all(
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(float(value))
                    for value in point
                )
                for point in waypoints
            )
        )
        if command not in CANONICAL_COMMANDS or not valid_waypoints:
            reasons.append("incomplete_structured_action")
        if reasons:
            failures.append({"sample_id": sample_id, "reasons": reasons})
    passed = len(rows) == EXPECTED_ROWS and dict(split_counts) == EXPECTED_SPLITS and not failures
    return {
        "passed": passed,
        "index_path": str(index_path),
        "rows": len(rows),
        "split_counts": dict(split_counts),
        "future_target_leak_count": future_target_leak_count,
        "future_target_leak_pattern_counts": dict(sorted(future_target_pattern_counts.items())),
        "failures": failures,
    }


def structured_signature(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    waypoints = tuple(
        tuple(float(value) for value in step[:2]) for step in row.get("waypoints", [])
    )
    return (
        str(row.get("sample_id") or ""),
        str(row.get("token") or ""),
        str(row.get("split") or ""),
        str(row.get("ego_image") or ""),
        str(row.get("infra_image") or ""),
        str(row.get("command") or ""),
        waypoints,
    )


def audit_structured_action_parity(reference_index: Path, candidate_index: Path) -> Dict[str, Any]:
    reference = load_index(reference_index)
    candidate = load_index(candidate_index)
    mismatches: List[str] = []
    if len(reference) != len(candidate):
        mismatches.append(f"row_count:{len(reference)}!={len(candidate)}")
    for position, (left, right) in enumerate(zip(reference, candidate)):
        if structured_signature(left) != structured_signature(right):
            mismatches.append(
                f"row_{position}:{sample_key(left)}!={sample_key(right)}"
            )
            if len(mismatches) >= 20:
                break
    return {
        "passed": not mismatches,
        "reference_index": str(reference_index),
        "candidate_index": str(candidate_index),
        "reference_rows": len(reference),
        "candidate_rows": len(candidate),
        "mismatches": mismatches,
    }


def enrich_index_with_ego_status(
    *,
    index_path: Path,
    v2x_root: Path,
    enriched_index_path: Path,
    summary_json_path: Path,
) -> Dict[str, Any]:
    rows = load_index(index_path)
    records = vehicle_records(v2x_root)
    by_frame, prev_lookup = build_prev_lookup(records)
    enriched_rows: List[Dict[str, Any]] = []
    missing: Counter[str] = Counter()
    for row in rows:
        new_row = dict(row)
        status, reason = compute_status(
            row=new_row,
            by_frame=by_frame,
            prev_lookup=prev_lookup,
            v2x_root=v2x_root,
        )
        new_row["ego_status"] = status
        enriched_rows.append(new_row)
        if reason:
            missing[reason] += 1
    write_jsonl(enriched_index_path, enriched_rows)
    summary = summarize_status(rows, enriched_rows, missing)
    payload = {
        "created_at_utc": utc_now(),
        "index_path": str(index_path),
        "enriched_index_path": str(enriched_index_path),
        "v2x_root": str(v2x_root),
        "summary": summary,
        "leakage_assessment": {
            "leak_free": True,
            "inputs_used": [
                "vehicle-side current pointcloud timestamp",
                "vehicle-side previous pointcloud timestamp",
                "vehicle-side current novatel_to_world pose",
                "vehicle-side previous novatel_to_world pose",
            ],
            "inputs_not_used": [
                "future waypoints",
                "future command",
                "target_text",
                "ground-truth action text",
            ],
            "risk_note": (
                "Ego status is derived only from current and previous ego poses and timestamps."
            ),
        },
        "safe_minimal_signal": {
            "definition": "availability flag, speed, heading delta, and yaw rate from current/previous ego pose only"
        },
    }
    write_json(summary_json_path, payload)
    return payload


def default_prompt_assets(label_dir: Path) -> str:
    return str(label_dir)


def common_train_config(
    *,
    model_path: Path,
    label_dir: Path,
    l3_metadata_dir: Path,
    v2x_root: Path,
    index_path: Path,
    variant_dir: Path,
    seed: int,
) -> Dict[str, Any]:
    return {
        "model_name_or_path": str(model_path),
        "seed": int(seed),
        "deterministic": False,
        "train_lm_targets": True,
        "lambda_lm": 1.0,
        "no_cot": False,
        "index_path": str(index_path),
        "index_must_exist": True,
        "index_target_parts": 3,
        "l4_dir": str(label_dir),
        "l3_metadata_dir": str(l3_metadata_dir),
        "v2x_root": str(v2x_root),
        "train_split": "train",
        "init_checkpoint_dir": None,
        "bf16": True,
        "device_map": None,
        "use_lora": True,
        "use_qlora": False,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "gradient_checkpointing": True,
        "batch_size": 1,
        "eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 5.0e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 1.0,
        "num_train_epochs": 10,
        "max_steps": 0,
        "log_every": 50,
        "save_every": 0,
        "save_epoch_checkpoints": True,
        "eval_at_epoch_end": True,
        "eval_every_n_epochs": 2,
        "save_best_checkpoint": True,
        "early_stop_if_single_class_prediction": True,
        "epoch_eval_split": "val",
        "epoch_eval_max_batches": 0,
        "image_max_pixels": 262144,
        "part1_object_limit": 0,
        "part1_sentence_limit": 0,
        "compact_target_part1": False,
        "lambda_wp": 2.0,
        "lambda_cmd": 1.0,
        "lambda_fde": 1.0,
        "command_class_weighting": "none",
        "command_class_weight_cap": 5.0,
        "eval_max_new_tokens": 512,
        "eval_generate_all_text": False,
        "eval_examples_every_n": 50,
        "eval_examples_limit": 64,
        "lm_target_policy": "parts1_3_only",
        "cot_diagnostic_format": "three_part",
        "mode": "v2x_image",
        "head_pooling": "mean",
        "fusion_mode": "standard",
        "command_head_mode": "standard",
        "causal_ego_anchor": False,
        "vision_encode_per_view": False,
        "view_order": "ego_infra",
        "infra_condition": "normal",
        "neutral_ego_only_prompt": False,
        "output_dir": str(variant_dir / "model"),
        "epoch_checkpoint_root": str(variant_dir / "epoch_checkpoints"),
        "best_checkpoint_dir": str(variant_dir / "best_checkpoint"),
        "best_checkpoint_json": str(variant_dir / "best_checkpoint.json"),
        "accepted_l4_run": default_prompt_assets(label_dir),
    }


def build_train_config(
    *,
    variant: str,
    model_path: Path,
    label_dir: Path,
    l3_metadata_dir: Path,
    v2x_root: Path,
    base_index_path: Path,
    ego_status_index_path: Path,
    variant_dir: Path,
    seed: int,
) -> Dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    index_path = ego_status_index_path if variant == "ego_status" else base_index_path
    config = common_train_config(
        model_path=model_path,
        label_dir=label_dir,
        l3_metadata_dir=l3_metadata_dir,
        v2x_root=v2x_root,
        index_path=index_path,
        variant_dir=variant_dir,
        seed=seed,
    )
    config["use_ego_status_prompt"] = variant == "ego_status"
    return config


def build_condition_eval_config(
    train_cfg: Mapping[str, Any],
    *,
    condition: str,
    shuffled_map_path: Path,
) -> Dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    config = dict(train_cfg)
    config["index_must_exist"] = True
    config["index_target_parts"] = 3
    config["lm_target_policy"] = "parts1_3_only"
    config["cot_diagnostic_format"] = "three_part"
    if condition == "ego_only":
        config["mode"] = "ego_only"
        config["neutral_ego_only_prompt"] = True
        config["infra_condition"] = "normal"
    elif condition == "blank":
        config["mode"] = "v2x_image"
        config["neutral_ego_only_prompt"] = False
        config["infra_condition"] = "blank"
    elif condition == "shuffled":
        config["mode"] = "v2x_image"
        config["neutral_ego_only_prompt"] = False
        config["infra_condition"] = "shuffled"
        config["shuffled_infra_map_path"] = str(shuffled_map_path)
    else:
        config["mode"] = "v2x_image"
        config["neutral_ego_only_prompt"] = False
        config["infra_condition"] = "normal"
    return config


def build_language_eval_config(
    eval_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    config = dict(eval_cfg)
    config["eval_batch_size"] = 16
    config["eval_max_new_tokens"] = 1536
    config["eval_generate_all_text"] = True
    config["eval_examples_every_n"] = 1
    config["eval_examples_limit"] = 0
    config["cot_diagnostic_format"] = "three_part"
    return config


def val_rows_from_index(index_path: Path) -> List[Dict[str, Any]]:
    rows = [row for row in load_index(index_path) if str(row.get("split")) == "val"]
    identities = [sample_identity(row) for row in rows]
    if len(rows) != EXPECTED_VAL_ROWS:
        raise ValueError(f"Expected {EXPECTED_VAL_ROWS} val rows, found {len(rows)}")
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate validation sample identities in index")
    return rows


def assert_shuffled_map(index_path: Path, shuffled_map_path: Path) -> Dict[str, Any]:
    index_rows = load_index(index_path)
    map_rows = load_index(shuffled_map_path)
    source_by_id = {sample_key(row): row for row in index_rows}
    seen_keys = [sample_key(row) for row in map_rows]
    donor_keys = [
        str(row.get("shuffled_source_sample_id") or row.get("shuffled_source_token") or "")
        for row in map_rows
    ]
    payload = {
        "index_count": len(index_rows),
        "map_count": len(map_rows),
        "same_keys": set(seen_keys) == set(source_by_id),
        "duplicate_source_count": len(seen_keys) - len(set(seen_keys)),
        "duplicate_donor_count": len(donor_keys) - len(set(donor_keys)),
        "same_scene_pair_count": 0,
        "cross_split_pair_count": 0,
        "self_pair_count": 0,
        "missing_image_count": 0,
        "missing_donor_count": 0,
    }
    for row, donor_key in zip(map_rows, donor_keys):
        source = source_by_id.get(sample_key(row))
        donor = source_by_id.get(donor_key)
        if source is None or donor is None:
            payload["missing_donor_count"] += 1
            continue
        payload["same_scene_pair_count"] += int(
            str(source.get("scene_token")) == str(donor.get("scene_token"))
        )
        payload["cross_split_pair_count"] += int(
            str(source.get("split")) != str(donor.get("split"))
        )
        payload["self_pair_count"] += int(sample_key(source) == sample_key(donor))
        payload["missing_image_count"] += int(
            not Path(str(row.get("infra_image") or "")).is_file()
        )
    payload["passed"] = all(
        [
            payload["index_count"] == EXPECTED_ROWS,
            payload["map_count"] == EXPECTED_ROWS,
            payload["same_keys"],
            payload["duplicate_source_count"] == 0,
            payload["duplicate_donor_count"] == 0,
            payload["same_scene_pair_count"] == 0,
            payload["cross_split_pair_count"] == 0,
            payload["self_pair_count"] == 0,
            payload["missing_image_count"] == 0,
            payload["missing_donor_count"] == 0,
        ]
    )
    if not payload["passed"]:
        raise ValueError(f"strict shuffled map failed audit: {payload}")
    return payload


def planning_row_path(run_dir: Path, variant: str, condition: str) -> Path:
    return variant_dirs(run_dir, variant)["predictions"] / f"structured_val654_{condition}.jsonl"


def planning_metrics_path(run_dir: Path, variant: str, condition: str) -> Path:
    return variant_dirs(run_dir, variant)["metrics"] / f"eval_val654_{condition}.json"


def planning_profile_path(run_dir: Path, variant: str) -> Path:
    return variant_dirs(run_dir, variant)["metrics"] / "runtime_profile_normal_common100.json"


def language_sample_path(run_dir: Path, condition: str) -> Path:
    return run_dir / "samples" / "language" / f"standard_{condition}_val654.jsonl"


def language_metrics_path(run_dir: Path, condition: str) -> Path:
    return run_dir / "metrics" / "language" / f"standard_{condition}_eval.json"


def language_diagnostics_json_path(run_dir: Path, condition: str) -> Path:
    return run_dir / "metrics" / "language" / f"standard_{condition}_diagnostics.json"


def language_diagnostics_rows_path(run_dir: Path, condition: str) -> Path:
    return run_dir / "samples" / "language" / f"standard_{condition}_diagnostics_rows.jsonl"


def language_diagnostics_report_path(run_dir: Path, condition: str) -> Path:
    return run_dir / "reports" / f"standard_{condition}_language_diagnostics.md"


def manifest_path(run_dir: Path) -> Path:
    return run_dir / "contracts" / "submission_manifest.json"


def table_contract_path(run_dir: Path) -> Path:
    return run_dir / "contracts" / "table_contract.json"


def pipeline_status_path(run_dir: Path) -> Path:
    return run_dir / "status" / "pipeline_status.json"


def write_pipeline_status(run_dir: Path, stage: str, state: str, **extra: Any) -> None:
    payload = {
        "updated_at_utc": utc_now(),
        "contract": PIPELINE_CONTRACT,
        "stage": stage,
        "state": state,
        "run_dir": str(run_dir),
    }
    payload.update(extra)
    write_json(pipeline_status_path(run_dir), payload)


def build_table_contract(
    *,
    run_dir: Path,
    anchor_index: Path,
    base_index: Path,
    shuffled_map_path: Path,
    model_path: Path,
    seed: int,
) -> Dict[str, Any]:
    return {
        "contract": PIPELINE_CONTRACT,
        "created_at_utc": utc_now(),
        "research_question": (
            "Does the matched roadside view improve QA, source attribution, and planning "
            "in the same paired V2I scenes, and do those signals move together?"
        ),
        "core_hypothesis": (
            "The matched roadside image helps structured planning and grounded language, "
            "but source attribution and planning response need not change in the same way."
        ),
        "paper_claim": (
            "CoVLM-Bench provides sample-matched roadside interventions for language and "
            "planning, with a reproducible raw dual-view VLM planner as reference."
        ),
        "baseline_control": {
            "planner_rows": ["CoVLM-Drive", "CoVLM-Drive*"],
            "input_controls": list(CONDITIONS),
            "best_control_definition": "stronger of ego_only and blank for QA only",
            "shuffled_role": "wrong-view diagnostic, not absent-view control",
        },
        "table_contract": {
            "table2_rows": {
                "CoVLM-Drive": {
                    "variant": "standard",
                    "l2_m": ["--"] * 6,
                    "command_accuracy": "--",
                    "balanced_command_accuracy": "--",
                    "latency_ms": "--",
                    "fps": "--",
                    "peak_memory_gib": "--",
                },
                "CoVLM-Drive*": {
                    "variant": "ego_status",
                    "l2_m": ["--"] * 6,
                    "command_accuracy": "--",
                    "balanced_command_accuracy": "--",
                    "latency_ms": "--",
                    "fps": "--",
                    "peak_memory_gib": "--",
                    "star_marker": (
                        "additional prompt-visible ego availability, speed, heading delta, "
                        "and yaw rate from current/previous ego poses only"
                    ),
                },
            },
            "table4": {
                "planning_count": EXPECTED_VAL_ROWS,
                "source_anchor_count": EXPECTED_SOURCE_ANCHORS,
                "conditions": list(CONDITIONS),
                "planning_fde_m": {condition: "--" for condition in CONDITIONS},
                "roadside_attribution_percent": {condition: "--" for condition in CONDITIONS},
            },
        },
        "metric_definitions": {
            "l2_m": {
                "definition": "mean Euclidean waypoint error in meters at each horizon",
                "horizons_s": PAPER_HORIZONS_S,
                "direction": "lower_is_better",
            },
            "command_accuracy": {
                "definition": "fraction of exact command matches on validation samples",
                "direction": "higher_is_better",
            },
            "balanced_command_accuracy": {
                "definition": "mean command accuracy over present validation command classes",
                "direction": "higher_is_better",
            },
            "latency_ms": {
                "definition": (
                    "profiled common token-device-model-return latency per sample with "
                    "warmup 20 and measured 100"
                ),
                "direction": "lower_is_better",
            },
            "fps": {
                "definition": "throughput computed from profiled latency scope",
                "direction": "higher_is_better",
            },
            "peak_memory_gib": {
                "definition": "CUDA peak allocated memory in GiB under the same profile scope",
                "direction": "lower_is_better",
            },
            "roadside_attribution_percent": {
                "definition": (
                    "percentage of the same 513 valid-source anchors whose generated Parts 2/3 "
                    "attribute evidence to the roadside view"
                ),
                "direction": "response_diagnostic_not_accuracy",
            },
        },
        "compute_resource_assumptions": {
            "fixed_seed": int(seed),
            "training_variants": list(VARIANTS),
            "input_controls": list(CONDITIONS),
            "model_path": str(model_path),
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "base_index": str(base_index),
            "anchor_index": str(anchor_index),
            "shuffled_map_path": str(shuffled_map_path),
        },
        "claim_gate": {
            "table2": "replace placeholders only after canonical variant metrics exist",
            "table4": "replace placeholders only after standard four-condition planning and language runs exist",
        },
    }


def build_manifest(
    *,
    run_dir: Path,
    label_dir: Path,
    reference_index: Path,
    anchor_index: Path,
    shuffled_map_path: Path,
    model_path: Path,
    l3_metadata_dir: Path,
    v2x_root: Path,
    python: Path,
    seed: int,
    base_index_path: Path,
    ego_status_index_path: Path,
    variant_train_configs: Mapping[str, Path],
    eval_configs: Mapping[str, Mapping[str, Path]],
    language_configs: Mapping[str, Path],
) -> Dict[str, Any]:
    return {
        "contract": PIPELINE_CONTRACT,
        "created_at_utc": utc_now(),
        "run_dir": str(run_dir),
        "label_dir": str(label_dir),
        "reference_index": str(reference_index),
        "anchor_index": str(anchor_index),
        "shuffled_map_path": str(shuffled_map_path),
        "model_path": str(model_path),
        "l3_metadata_dir": str(l3_metadata_dir),
        "v2x_root": str(v2x_root),
        "python": str(python),
        "seed": int(seed),
        "indexes": {
            "standard": str(base_index_path),
            "ego_status": str(ego_status_index_path),
        },
        "variants": {
            variant: {
                "train_config": str(variant_train_configs[variant]),
                "eval_configs": {
                    condition: str(eval_configs[variant][condition])
                    for condition in CONDITIONS
                },
                "best_checkpoint": str(variant_dirs(run_dir, variant)["best_checkpoint"]),
            }
            for variant in VARIANTS
        },
        "language": {
            "standard_configs": {
                condition: str(language_configs[condition])
                for condition in CONDITIONS
            }
        },
    }


def load_manifest(run_dir: Path) -> Dict[str, Any]:
    manifest = read_json(manifest_path(run_dir))
    if manifest.get("contract") != PIPELINE_CONTRACT:
        raise ValueError(f"Unexpected submission contract in {manifest_path(run_dir)}")
    return manifest


def configured_python(manifest: Mapping[str, Any]) -> Path:
    return Path(str(manifest["python"]))


def configured_v2x_root(manifest: Mapping[str, Any]) -> Path:
    return Path(str(manifest["v2x_root"]))


def configured_seed(manifest: Mapping[str, Any]) -> int:
    return int(manifest["seed"])


def configured_index(manifest: Mapping[str, Any], variant: str) -> Path:
    return Path(str(manifest["indexes"][variant]))


def configured_best_checkpoint(run_dir: Path, variant: str) -> Path:
    return variant_dirs(run_dir, variant)["best_checkpoint"]


def configured_train_config(manifest: Mapping[str, Any], variant: str) -> Path:
    return Path(str(manifest["variants"][variant]["train_config"]))


def configured_eval_config(manifest: Mapping[str, Any], variant: str, condition: str) -> Path:
    return Path(str(manifest["variants"][variant]["eval_configs"][condition]))


def configured_language_config(manifest: Mapping[str, Any], condition: str) -> Path:
    return Path(str(manifest["language"]["standard_configs"][condition]))


def clean_env(gpu: str) -> Dict[str, str]:
    env = dict(os.environ)
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def gpu_state(gpu: str) -> Dict[str, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    free_mib, utilization = [int(value.strip()) for value in result.stdout.split(",")]
    return {"free_mib": free_mib, "utilization": utilization}


def wait_for_gpu(gpu: str, min_free_mib: int, max_utilization: int, poll_seconds: int) -> None:
    while True:
        state = gpu_state(gpu)
        if state["free_mib"] >= min_free_mib and state["utilization"] <= max_utilization:
            logging.info("GPU %s accepted: %s", gpu, state)
            return
        logging.info("GPU %s busy (%s); waiting", gpu, state)
        time.sleep(poll_seconds)


def run_logged(command: Sequence[str], log_path: Path, env: Mapping[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(command) + "\n")
        log_file.flush()
        subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            env=dict(env),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )


def load_expected_val_rows(index_path: Path) -> List[Dict[str, Any]]:
    return val_rows_from_index(index_path)


def validate_structured_predictions(
    *,
    path: Path,
    expected_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    rows = read_jsonl(path)
    expected = [sample_identity(row) for row in expected_rows]
    observed = [sample_identity(row) for row in rows]
    if len(rows) != len(expected_rows):
        raise ValueError(f"{path}: expected {len(expected_rows)} rows, found {len(rows)}")
    if observed != expected:
        raise ValueError(f"{path}: validation identities drifted from the canonical val index")
    for row in rows:
        waypoints = row.get("pred_waypoints")
        if not (
            isinstance(waypoints, list)
            and len(waypoints) == 6
            and all(isinstance(point, list) and len(point) >= 2 for point in waypoints)
        ):
            raise ValueError(f"{path}: invalid pred_waypoints for {sample_key(row)}")
    return {
        "path": str(path),
        "count": len(rows),
        "same_identity_order": True,
    }


def validate_generated_language_samples(
    *,
    path: Path,
    expected_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    rows = read_jsonl(path)
    expected = [sample_identity(row) for row in expected_rows]
    observed = [sample_identity(row) for row in rows]
    if len(rows) != len(expected_rows):
        raise ValueError(f"{path}: expected {len(expected_rows)} generated rows, found {len(rows)}")
    if observed != expected:
        raise ValueError(f"{path}: generated language identities drifted from canonical val order")
    bad = [sample_key(row) for row in rows if not str(row.get("raw_generated_cot") or row.get("generated_text") or "")]
    if bad:
        raise ValueError(f"{path}: empty generated text for {bad[:5]}")
    return {
        "path": str(path),
        "count": len(rows),
        "same_identity_order": True,
        "empty_text_count": 0,
    }


def train_metadata_path(run_dir: Path, variant: str) -> Path:
    return variant_dirs(run_dir, variant)["model"] / "run_metadata_train.json"


def training_complete(run_dir: Path, variant: str) -> bool:
    metadata_path = train_metadata_path(run_dir, variant)
    checkpoint_dir = configured_best_checkpoint(run_dir, variant)
    if not metadata_path.exists() or not checkpoint_dir.exists():
        return False
    metadata = read_json(metadata_path)
    return metadata.get("status") in {"completed", "stopped_early"}


def planning_eval_complete(run_dir: Path, variant: str, condition: str, expected_rows: Sequence[Mapping[str, Any]]) -> bool:
    metrics_path = planning_metrics_path(run_dir, variant, condition)
    rows_path = planning_row_path(run_dir, variant, condition)
    if not metrics_path.exists() or not rows_path.exists():
        return False
    metrics = read_json(metrics_path)
    if int(metrics.get("count", -1)) != len(expected_rows):
        return False
    try:
        validate_structured_predictions(path=rows_path, expected_rows=expected_rows)
    except Exception:
        return False
    return True


def run_structured_eval(
    *,
    run_dir: Path,
    manifest: Mapping[str, Any],
    variant: str,
    condition: str,
    gpu: str,
) -> Dict[str, Any]:
    expected_rows = load_expected_val_rows(configured_index(manifest, variant))
    if planning_eval_complete(run_dir, variant, condition, expected_rows):
        return {
            "variant": variant,
            "condition": condition,
            "status": "reused_complete_artifact",
            "metrics_path": str(planning_metrics_path(run_dir, variant, condition)),
            "predictions_path": str(planning_row_path(run_dir, variant, condition)),
        }
    config_path = configured_eval_config(manifest, variant, condition)
    checkpoint_dir = configured_best_checkpoint(run_dir, variant)
    metrics_path = planning_metrics_path(run_dir, variant, condition)
    rows_path = planning_row_path(run_dir, variant, condition)
    log_path = variant_dirs(run_dir, variant)["logs"] / f"eval_{condition}.log"
    command = [
        str(configured_python(manifest)),
        "-m",
        "projects.covla_baseline.evaluate",
        "--config",
        str(config_path),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--split",
        "val",
        "--output",
        str(metrics_path),
        "--structured-output",
        str(rows_path),
        "--cot-diagnostic-format",
        "three_part",
    ]
    if condition == "normal":
        command.extend(
            [
                "--profile-eval",
                "--profile-warmup-samples",
                "20",
                "--profile-max-measured-samples",
                "100",
                "--profile-output",
                str(planning_profile_path(run_dir, variant)),
            ]
        )
    run_logged(command, log_path, clean_env(gpu))
    validation = validate_structured_predictions(path=rows_path, expected_rows=expected_rows)
    metrics = read_json(metrics_path)
    if int(metrics.get("count", -1)) != EXPECTED_VAL_ROWS:
        raise RuntimeError(f"{metrics_path}: expected count={EXPECTED_VAL_ROWS}")
    return {
        "variant": variant,
        "condition": condition,
        "status": "complete",
        "metrics_path": str(metrics_path),
        "predictions_path": str(rows_path),
        "validation": validation,
    }


def run_variant(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    ensure_dirs(run_dir)
    manifest = load_manifest(run_dir)
    if args.variant not in VARIANTS:
        raise ValueError(f"unknown variant: {args.variant}")
    wait_for_gpu(args.gpu, args.min_free_mib, args.max_utilization, args.poll_seconds)
    write_pipeline_status(run_dir, "run_variant", "starting", variant=args.variant, gpu=args.gpu)
    train_cfg = configured_train_config(manifest, args.variant)
    best_checkpoint = configured_best_checkpoint(run_dir, args.variant)
    metadata_path = train_metadata_path(run_dir, args.variant)
    partial_markers = [
        variant_dirs(run_dir, args.variant)["model"] / "train_log.jsonl",
        variant_dirs(run_dir, args.variant)["epoch_checkpoints"],
    ]
    if not training_complete(run_dir, args.variant):
        if metadata_path.exists():
            metadata = read_json(metadata_path)
            if metadata.get("status") not in {"completed", "stopped_early"}:
                raise RuntimeError(
                    f"Refusing to resume or overwrite partial run for {args.variant}; "
                    "use a new run directory"
                )
        elif any(path.exists() for path in partial_markers):
            raise RuntimeError(
                f"Refusing to overwrite partial run for {args.variant}; use a new run directory"
            )
        command = [
            str(configured_python(manifest)),
            "-m",
            "projects.covla_baseline.train",
            "--config",
            str(train_cfg),
        ]
        run_logged(
            command,
            variant_dirs(run_dir, args.variant)["logs"] / "train.log",
            clean_env(args.gpu),
        )
    if not best_checkpoint.exists():
        raise FileNotFoundError(f"Missing best checkpoint: {best_checkpoint}")
    results = []
    for condition in CONDITIONS:
        write_pipeline_status(
            run_dir,
            "run_variant",
            "evaluating_condition",
            variant=args.variant,
            condition=condition,
            gpu=args.gpu,
        )
        results.append(
            run_structured_eval(
                run_dir=run_dir,
                manifest=manifest,
                variant=args.variant,
                condition=condition,
                gpu=args.gpu,
            )
        )
    payload = {
        "contract": PIPELINE_CONTRACT,
        "variant": args.variant,
        "completed_at_utc": utc_now(),
        "best_checkpoint": str(best_checkpoint),
        "results": results,
    }
    write_json(variant_dirs(run_dir, args.variant)["status"] / "run_variant_complete.json", payload)
    write_pipeline_status(run_dir, "run_variant", "complete", variant=args.variant, gpu=args.gpu)


def run_language(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    ensure_dirs(run_dir)
    manifest = load_manifest(run_dir)
    if args.condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {args.condition}")
    if not training_complete(run_dir, "standard"):
        raise RuntimeError("standard variant is not trained yet")
    expected_rows = load_expected_val_rows(configured_index(manifest, "standard"))
    checkpoint_dir = configured_best_checkpoint(run_dir, "standard")
    config_path = configured_language_config(manifest, args.condition)
    metrics_path = language_metrics_path(run_dir, args.condition)
    samples_path = language_sample_path(run_dir, args.condition)
    rows_path = language_diagnostics_rows_path(run_dir, args.condition)
    diagnostics_json = language_diagnostics_json_path(run_dir, args.condition)
    diagnostics_md = language_diagnostics_report_path(run_dir, args.condition)
    log_path = run_dir / "logs" / "language" / f"standard_{args.condition}.log"
    if metrics_path.exists() and samples_path.exists():
        validate_generated_language_samples(path=samples_path, expected_rows=expected_rows)
    else:
        wait_for_gpu(args.gpu, args.min_free_mib, args.max_utilization, args.poll_seconds)
        write_pipeline_status(
            run_dir,
            "run_language",
            "generating",
            condition=args.condition,
            gpu=args.gpu,
        )
        command = [
            str(configured_python(manifest)),
            "-m",
            "projects.covla_baseline.evaluate",
            "--config",
            str(config_path),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--split",
            "val",
            "--output",
            str(metrics_path),
            "--examples-output",
            str(samples_path),
            "--examples-every-n",
            "1",
            "--generate-all-text",
            "--action-first-report",
            "--cot-diagnostic-format",
            "three_part",
        ]
        run_logged(command, log_path, clean_env(args.gpu))
        validate_generated_language_samples(path=samples_path, expected_rows=expected_rows)
    metrics = read_json(metrics_path)
    if int(metrics.get("count", -1)) != EXPECTED_VAL_ROWS:
        raise RuntimeError(f"{metrics_path}: expected count={EXPECTED_VAL_ROWS}")
    diagnostics = cot_diagnostics.build_diagnostics(
        metrics_path,
        samples_path,
        configured_index(manifest, "standard"),
        rows_path,
        contract="three_part",
    )
    write_json(diagnostics_json, diagnostics)
    cot_diagnostics.write_report(diagnostics_md, diagnostics, contract="three_part")
    quality_summary, per_sample, judge_source = evaluate_cot_quality.build_summary(
        argparse.Namespace(
            samples=samples_path,
            index=configured_index(manifest, "standard"),
            contract="three_part",
        )
    )
    quality_json = diagnostics_json.with_name(f"{diagnostics_json.stem}_quality_summary.json")
    quality_rows = rows_path.with_name(f"{rows_path.stem}_quality_summary.jsonl")
    quality_md = diagnostics_md.with_name(f"{diagnostics_md.stem}_quality_summary.md")
    write_json(quality_json, quality_summary)
    write_jsonl(quality_rows, (evaluate_cot_quality.compact_three_part_diagnostic(row) for row in per_sample))
    evaluate_cot_quality.write_three_part_table(quality_md, quality_summary)
    write_json(
        run_dir / "status" / f"language_{args.condition}.json",
        {
            "contract": PIPELINE_CONTRACT,
            "condition": args.condition,
            "completed_at_utc": utc_now(),
            "metrics_path": str(metrics_path),
            "samples_path": str(samples_path),
            "diagnostics_path": str(diagnostics_json),
            "quality_summary_path": str(quality_json),
            "quality_sample_rows": len(per_sample),
            "judge_source_rows": len(judge_source),
        },
    )
    write_pipeline_status(run_dir, "run_language", "complete", condition=args.condition, gpu=args.gpu)


def command_of(row: Mapping[str, Any], key: str) -> str:
    value = str(row.get(key) or "").strip().upper()
    return value


def aggregate_planning_rows(
    rows: Sequence[Mapping[str, Any]],
    subset_ids: Sequence[str],
) -> Dict[str, Any]:
    by_id = {sample_key(row): row for row in rows}
    missing = [sample_id for sample_id in subset_ids if sample_id not in by_id]
    if missing:
        raise ValueError(f"missing {len(missing)} predictions; first missing={missing[:5]}")
    selected = [by_id[sample_id] for sample_id in subset_ids]
    step_values: List[List[float]] = []
    fdes: List[float] = []
    correct: List[float] = []
    per_class_total: Dict[str, int] = defaultdict(int)
    per_class_correct: Dict[str, int] = defaultdict(int)
    for row in selected:
        gt_wp = row.get("gt_waypoints") or []
        pred_wp = row.get("pred_waypoints") or []
        if len(gt_wp) != len(pred_wp):
            raise ValueError(f"waypoint length mismatch for {sample_key(row)}")
        distances = [euclidean_xy(pred_step, gt_step) for pred_step, gt_step in zip(pred_wp, gt_wp)]
        for idx, distance in enumerate(distances):
            while len(step_values) <= idx:
                step_values.append([])
            step_values[idx].append(distance)
        if distances:
            fdes.append(distances[-1])
        gt_command = command_of(row, "gt_command")
        pred_command = command_of(row, "pred_command")
        if gt_command:
            correct.append(1.0 if pred_command == gt_command else 0.0)
            per_class_total[gt_command] += 1
            per_class_correct[gt_command] += int(pred_command == gt_command)
    macro_values = [
        per_class_correct[command] / per_class_total[command]
        for command in COMMANDS
        if per_class_total.get(command, 0) > 0
    ]
    return {
        "N": len(selected),
        "waypoint_ADE_each_step": [mean(values) for values in step_values],
        "final_displacement_error": mean(fdes),
        "command_accuracy": mean(correct),
        "macro_command_accuracy_present_classes": mean(macro_values),
        "present_command_classes": [command for command in COMMANDS if per_class_total.get(command, 0) > 0],
    }


def section_source_presence_claim(section_text: str) -> bool:
    explicit_no_v2x = explicit_no_roadside_increment(section_text) or bool(
        NO_V2X_SCHEMA_RE.search(section_text)
    )
    return (not explicit_no_v2x) and positive_roadside_claim(section_text)


def summarize_language_condition(
    rows: Sequence[Mapping[str, Any]],
    anchor_ids: Sequence[str],
) -> Dict[str, Any]:
    by_id = {sample_key(row): row for row in rows}
    selected = [by_id[sample_id] for sample_id in anchor_ids]
    parsed_roles: Counter[str] = Counter()
    combined_claims = 0
    part2_claims = 0
    part3_claims = 0
    valid_parse = 0
    forbidden_action = 0
    future_leak = 0
    for row in selected:
        text = str(row.get("raw_generated_cot") or row.get("generated_text") or "")
        parsed = parse_three_part_cot(text)
        sections = parsed["sections"]
        role = parse_source_role(text)
        parsed_roles[role] += 1
        valid_parse += int(bool(parsed["format_parse_ok"]))
        forbidden_action += int(
            any(bool(value) for value in parsed["forbidden_text_action_fields"].values())
        )
        future_leak += int(bool(parsed["future_target_leak"]))
        combined_claims += int(role in {"decision_relevant", "context_only"})
        part2_claims += int(section_source_presence_claim(str(sections["critical_evidence"])))
        part3_claims += int(section_source_presence_claim(str(sections["decision_reasoning"])))
    denom = max(len(selected), 1)
    return {
        "N": len(selected),
        "roadside_attribution_rate_percent": 100.0 * combined_claims / denom,
        "part2_roadside_claim_rate_percent": 100.0 * part2_claims / denom,
        "part3_roadside_claim_rate_percent": 100.0 * part3_claims / denom,
        "exact_three_part_parse_rate": valid_parse / denom,
        "forbidden_text_action_field_rate": forbidden_action / denom,
        "future_target_leak_rate": future_leak / denom,
        "parsed_source_role_counts": dict(parsed_roles),
    }


def anchor_ids_from_legacy_index(path: Path) -> List[str]:
    rows = [
        row
        for row in load_index(path)
        if str(row.get("split")) == "val" and list(row.get("infra_only_critical_ids") or [])
    ]
    ids = [sample_key(row) for row in rows]
    if len(ids) != EXPECTED_SOURCE_ANCHORS:
        raise ValueError(
            f"expected {EXPECTED_SOURCE_ANCHORS} source anchors from {path}, found {len(ids)}"
        )
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate source-anchor sample ids")
    return ids


def submission_artifacts_ready(run_dir: Path) -> Tuple[bool, List[str]]:
    required = [
        variant_dirs(run_dir, variant)["status"] / "run_variant_complete.json"
        for variant in VARIANTS
    ]
    required.extend(run_dir / "status" / f"language_{condition}.json" for condition in CONDITIONS)
    missing = [str(path) for path in required if not path.exists()]
    return not missing, missing


def wait_for_submission_artifacts(
    run_dir: Path,
    *,
    wait_seconds: int,
    poll_seconds: int,
) -> None:
    deadline = time.monotonic() + max(wait_seconds, 0)
    while True:
        ready, missing = submission_artifacts_ready(run_dir)
        if ready:
            return
        if wait_seconds <= 0 or time.monotonic() >= deadline:
            raise RuntimeError(
                "submission artifacts are incomplete; missing: " + ", ".join(missing)
            )
        time.sleep(max(poll_seconds, 1))


def summarize(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    ensure_dirs(run_dir)
    wait_for_submission_artifacts(
        run_dir,
        wait_seconds=int(args.wait_seconds),
        poll_seconds=int(args.poll_seconds),
    )
    manifest = load_manifest(run_dir)
    planning_summary: Dict[str, Any] = {}
    for variant in VARIANTS:
        index_rows = load_expected_val_rows(configured_index(manifest, variant))
        all_ids = [sample_key(row) for row in index_rows]
        variant_payload: Dict[str, Any] = {"conditions": {}}
        for condition in CONDITIONS:
            rows = read_jsonl(planning_row_path(run_dir, variant, condition))
            validate_structured_predictions(
                path=planning_row_path(run_dir, variant, condition),
                expected_rows=index_rows,
            )
            variant_payload["conditions"][condition] = aggregate_planning_rows(rows, all_ids)
        normal = variant_payload["conditions"]["normal"]
        for condition in CONTROL_CONDITIONS:
            current = variant_payload["conditions"][condition]
            current["fde_delta_vs_normal_m"] = (
                float(current["final_displacement_error"]) - float(normal["final_displacement_error"])
            )
            current["command_delta_vs_normal"] = (
                float(current["command_accuracy"]) - float(normal["command_accuracy"])
            )
            current["macro_delta_vs_normal"] = (
                float(current["macro_command_accuracy_present_classes"])
                - float(normal["macro_command_accuracy_present_classes"])
            )
        profile_path = planning_profile_path(run_dir, variant)
        if profile_path.exists():
            variant_payload["runtime_profile"] = read_json(profile_path)
        planning_summary[variant] = variant_payload
    anchor_ids = anchor_ids_from_legacy_index(Path(str(manifest["anchor_index"])))
    subset_summary: Dict[str, Any] = {}
    for variant in VARIANTS:
        index_rows = load_expected_val_rows(configured_index(manifest, variant))
        expected_ids = {sample_key(row) for row in index_rows}
        missing = [sample_id for sample_id in anchor_ids if sample_id not in expected_ids]
        if missing:
            raise ValueError(f"{variant}: missing anchor ids in current val index: {missing[:5]}")
        variant_payload: Dict[str, Any] = {"conditions": {}}
        for condition in CONDITIONS:
            rows = read_jsonl(planning_row_path(run_dir, variant, condition))
            variant_payload["conditions"][condition] = aggregate_planning_rows(rows, anchor_ids)
        normal = variant_payload["conditions"]["normal"]
        for condition in CONTROL_CONDITIONS:
            current = variant_payload["conditions"][condition]
            current["fde_delta_vs_normal_m"] = (
                float(current["final_displacement_error"]) - float(normal["final_displacement_error"])
            )
            current["command_delta_vs_normal"] = (
                float(current["command_accuracy"]) - float(normal["command_accuracy"])
            )
            current["macro_delta_vs_normal"] = (
                float(current["macro_command_accuracy_present_classes"])
                - float(normal["macro_command_accuracy_present_classes"])
            )
        subset_summary[variant] = variant_payload
    language_summary: Dict[str, Any] = {}
    valid_rate = None
    standard_index_rows = load_expected_val_rows(configured_index(manifest, "standard"))
    for condition in CONDITIONS:
        sample_path = language_sample_path(run_dir, condition)
        validate_generated_language_samples(path=sample_path, expected_rows=standard_index_rows)
        rows = read_jsonl(sample_path)
        payload = summarize_language_condition(rows, anchor_ids)
        if condition == "normal":
            valid_rate = payload["roadside_attribution_rate_percent"]
            payload["valid_minus_condition_pp"] = None
        else:
            payload["valid_minus_condition_pp"] = float(valid_rate) - float(
                payload["roadside_attribution_rate_percent"]
            )
        language_summary[condition] = payload
    summary_payload = {
        "contract": PIPELINE_CONTRACT,
        "completed_at_utc": utc_now(),
        "run_dir": str(run_dir),
        "planning_all_654": planning_summary,
        "planning_source_anchor_513": subset_summary,
        "language_source_anchor_513": language_summary,
        "definitions": {
            "planning_delta": "condition minus normal for each variant; positive FDE delta is worse",
            "roadside_attribution": (
                "combined Parts 2/3 source-presence claim on the same 513 valid-source anchors"
            ),
            "part2_part3_claim": (
                "section-local roadside claim rate using the same deterministic lexical parser"
            ),
        },
    }
    write_json(run_dir / "results" / "submission_summary.json", summary_payload)
    lines = [
        "# Submission Summary",
        "",
        f"- Contract: `{PIPELINE_CONTRACT}`",
        f"- Validation planning rows: {EXPECTED_VAL_ROWS}",
        f"- Source anchors: {EXPECTED_SOURCE_ANCHORS}",
        "",
        "## Planning (all 654)",
        "",
    ]
    for variant, payload in planning_summary.items():
        normal = payload["conditions"]["normal"]
        lines.append(
            f"- {variant}: FDE={normal['final_displacement_error']:.4f}, "
            f"Cmd={normal['command_accuracy']:.4f}, "
            f"Macro={normal['macro_command_accuracy_present_classes']:.4f}"
        )
    lines.extend(["", "## Language (513 anchors)", ""])
    for condition, payload in language_summary.items():
        gap = payload["valid_minus_condition_pp"]
        gap_text = "Ref." if gap is None else f"{gap:+.1f} pp"
        lines.append(
            f"- {condition}: attr={payload['roadside_attribution_rate_percent']:.1f}%, "
            f"part2={payload['part2_roadside_claim_rate_percent']:.1f}%, "
            f"part3={payload['part3_roadside_claim_rate_percent']:.1f}%, gap={gap_text}"
        )
    (run_dir / "reports" / "submission_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    write_pipeline_status(run_dir, "summarize", "complete")


def prepare(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    ensure_dirs(run_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "pipeline.log"),
        ],
    )
    write_pipeline_status(run_dir, "prepare", "starting")
    label_dir = args.label_dir.resolve()
    reference_index = args.reference_index.resolve()
    anchor_index = args.anchor_index.resolve()
    shuffled_map_path = args.shuffled_map_path.resolve()
    model_path = args.model_path.resolve()
    l3_metadata_dir = args.l3_metadata_dir.resolve()
    v2x_root = args.v2x_root.resolve()
    python = args.python.resolve()
    label_audit = audit_labels(label_dir)
    write_json(run_dir / "contracts" / "label_audit.json", label_audit)
    if not label_audit["passed"]:
        raise RuntimeError("cleaned label audit failed")
    base_index_path = run_dir / "indexes" / "three_part_2129.jsonl"
    build_index(
        l4_dir=label_dir,
        output=base_index_path,
        v2x_root=v2x_root,
        fail_on_error=True,
        target_parts=3,
    )
    index_audit = audit_three_part_index(base_index_path)
    action_parity = audit_structured_action_parity(reference_index, base_index_path)
    shuffled_audit = assert_shuffled_map(base_index_path, shuffled_map_path)
    anchor_ids = anchor_ids_from_legacy_index(anchor_index)
    write_json(run_dir / "contracts" / "index_audit.json", index_audit)
    write_json(run_dir / "contracts" / "structured_action_parity.json", action_parity)
    write_json(run_dir / "contracts" / "strict_shuffled_map_audit.json", shuffled_audit)
    write_json(
        run_dir / "contracts" / "source_anchor_audit.json",
        {
            "anchor_index": str(anchor_index),
            "anchor_count": len(anchor_ids),
            "expected_anchor_count": EXPECTED_SOURCE_ANCHORS,
            "passed": len(anchor_ids) == EXPECTED_SOURCE_ANCHORS,
        },
    )
    if not index_audit["passed"] or not action_parity["passed"]:
        raise RuntimeError("three-part index contract failed")
    ego_status_index_path = run_dir / "indexes" / "three_part_2129_ego_status.jsonl"
    ego_status_summary = enrich_index_with_ego_status(
        index_path=base_index_path,
        v2x_root=v2x_root,
        enriched_index_path=ego_status_index_path,
        summary_json_path=run_dir / "contracts" / "ego_status_audit.json",
    )
    train_config_paths: Dict[str, Path] = {}
    eval_config_paths: Dict[str, Dict[str, Path]] = {variant: {} for variant in VARIANTS}
    language_config_paths: Dict[str, Path] = {}
    for variant in VARIANTS:
        dirs = variant_dirs(run_dir, variant)
        train_config = build_train_config(
            variant=variant,
            model_path=model_path,
            label_dir=label_dir,
            l3_metadata_dir=l3_metadata_dir,
            v2x_root=v2x_root,
            base_index_path=base_index_path,
            ego_status_index_path=ego_status_index_path,
            variant_dir=dirs["root"],
            seed=args.seed,
        )
        train_path = run_dir / "configs" / variant / "train_10ep.yaml"
        write_yaml(train_path, train_config)
        train_config_paths[variant] = train_path
        for condition in CONDITIONS:
            eval_cfg = build_condition_eval_config(
                train_config,
                condition=condition,
                shuffled_map_path=shuffled_map_path,
            )
            eval_path = run_dir / "configs" / variant / f"eval_{condition}.yaml"
            write_yaml(eval_path, eval_cfg)
            eval_config_paths[variant][condition] = eval_path
            if variant == "standard":
                language_cfg = build_language_eval_config(eval_cfg)
                language_path = run_dir / "configs" / "language" / f"standard_{condition}.yaml"
                write_yaml(language_path, language_cfg)
                language_config_paths[condition] = language_path
    table_contract = build_table_contract(
        run_dir=run_dir,
        anchor_index=anchor_index,
        base_index=base_index_path,
        shuffled_map_path=shuffled_map_path,
        model_path=model_path,
        seed=args.seed,
    )
    write_json(table_contract_path(run_dir), table_contract)
    manifest = build_manifest(
        run_dir=run_dir,
        label_dir=label_dir,
        reference_index=reference_index,
        anchor_index=anchor_index,
        shuffled_map_path=shuffled_map_path,
        model_path=model_path,
        l3_metadata_dir=l3_metadata_dir,
        v2x_root=v2x_root,
        python=python,
        seed=args.seed,
        base_index_path=base_index_path,
        ego_status_index_path=ego_status_index_path,
        variant_train_configs=train_config_paths,
        eval_configs=eval_config_paths,
        language_configs=language_config_paths,
    )
    write_json(manifest_path(run_dir), manifest)
    write_json(
        run_dir / "status" / "prepare_complete.json",
        {
            "contract": PIPELINE_CONTRACT,
            "completed_at_utc": utc_now(),
            "base_index_rows": line_count(base_index_path),
            "ego_status_index_rows": line_count(ego_status_index_path),
            "ego_status_available_count": ego_status_summary["summary"]["available_count"],
            "variants": list(VARIANTS),
            "conditions": list(CONDITIONS),
        },
    )
    write_pipeline_status(
        run_dir,
        "prepare",
        "complete",
        variants=list(VARIANTS),
        conditions=list(CONDITIONS),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Build the canonical indexes, audits, and configs.")
    prepare_parser.add_argument("--run-dir", type=Path, required=True)
    prepare_parser.add_argument("--label-dir", type=Path, required=True)
    prepare_parser.add_argument("--reference-index", type=Path, required=True)
    prepare_parser.add_argument("--anchor-index", type=Path, required=True)
    prepare_parser.add_argument("--shuffled-map-path", type=Path, required=True)
    prepare_parser.add_argument("--model-path", type=Path, required=True)
    prepare_parser.add_argument("--l3-metadata-dir", type=Path, required=True)
    prepare_parser.add_argument("--v2x-root", type=Path, required=True)
    prepare_parser.add_argument("--python", type=Path, default=Path(sys.executable))
    prepare_parser.add_argument("--seed", type=int, default=20260524)

    variant_parser = subparsers.add_parser("run-variant", help="Train one variant and run four structured controls.")
    variant_parser.add_argument("--run-dir", type=Path, required=True)
    variant_parser.add_argument("--variant", choices=VARIANTS, required=True)
    variant_parser.add_argument("--gpu", required=True)
    variant_parser.add_argument("--min-free-mib", type=int, default=45000)
    variant_parser.add_argument("--max-utilization", type=int, default=10)
    variant_parser.add_argument("--poll-seconds", type=int, default=30)

    language_parser = subparsers.add_parser("run-language", help="Run standard-variant generated language for one condition.")
    language_parser.add_argument("--run-dir", type=Path, required=True)
    language_parser.add_argument("--condition", choices=CONDITIONS, required=True)
    language_parser.add_argument("--gpu", required=True)
    language_parser.add_argument("--min-free-mib", type=int, default=45000)
    language_parser.add_argument("--max-utilization", type=int, default=10)
    language_parser.add_argument("--poll-seconds", type=int, default=30)

    summarize_parser = subparsers.add_parser("summarize", help="Build the canonical submission summary.")
    summarize_parser.add_argument("--run-dir", type=Path, required=True)
    summarize_parser.add_argument("--wait-seconds", type=int, default=0)
    summarize_parser.add_argument("--poll-seconds", type=int, default=30)

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        prepare(args)
        return
    if args.command == "run-variant":
        run_variant(args)
        return
    if args.command == "run-language":
        run_language(args)
        return
    if args.command == "summarize":
        summarize(args)
        return
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
