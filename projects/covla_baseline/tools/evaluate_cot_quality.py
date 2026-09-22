#!/usr/bin/env python3
"""Deterministic CoT quality metrics for full validation generated samples."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from projects.covla_baseline.three_part_cot import (
    CONTRACT_NAME as THREE_PART_CONTRACT,
    SECTION_KEYS as THREE_PART_SECTION_KEYS,
    aggregate_three_part_parses,
    parse_three_part_cot,
)


COMMANDS = {"GO_STRAIGHT", "TURN_LEFT", "TURN_RIGHT", "LATERAL_SHIFT", "STOP", "SLOW_DOWN"}
TRAJECTORY_GROUNDED_INFRA_ROLES = {
    "context_only",
    "decision_relevant",
    "no_additional_evidence",
}
OBJECT_ID_RE = re.compile(r"\b[EI]\d+\b", re.I)
TOKEN_RE = re.compile(r"[a-z0-9]+")
PART_RE = re.compile(r"(?im)^\s*part\s*([1-4])\b[^\n]*")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
DISTANCE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*m\b", re.I)
SPEED_RE = re.compile(r"(\d+(?:\.\d+)?)\s*m\s*/\s*s\b", re.I)


def _fallback_canonical_command(command: Any) -> str:
    text = str(command or "UNKNOWN").strip().upper().replace(" ", "_").replace("-", "_")
    aliases = {
        "STRAIGHT": "GO_STRAIGHT",
        "FORWARD": "GO_STRAIGHT",
        "KEEP_STRAIGHT": "GO_STRAIGHT",
        "LEFT": "TURN_LEFT",
        "RIGHT": "TURN_RIGHT",
        "MERGE": "LATERAL_SHIFT",
        "LANE_CHANGE": "LATERAL_SHIFT",
        "CHANGE_LANE": "LATERAL_SHIFT",
        "BRAKE": "STOP",
        "DECELERATE": "SLOW_DOWN",
    }
    return aliases.get(text, text if text in COMMANDS else "UNKNOWN")


def _extract_part4(text: str) -> str:
    matches = list(PART_RE.finditer(text))
    part4 = [m for m in matches if m.group(1) == "4"]
    if not part4:
        return text
    return text[part4[-1].start() :]


def _fallback_parse_raw_command(text: str) -> Optional[str]:
    valid_command = None
    for line in str(text or "").splitlines():
        lowered = line.lower()
        if "command" not in lowered or ":" not in line:
            continue
        if "one of" in lowered or "<" in line or ">" in line:
            continue
        value = " ".join(line.split(":", 1)[1].strip().split()[0:3]).strip(" .,:;[]()")
        command = _fallback_canonical_command(value)
        if command != "UNKNOWN":
            valid_command = command
    if valid_command is not None:
        return valid_command
    return None


def _extract_first_bracket_payload(text: str, start_idx: int) -> Optional[str]:
    start = text.find("[", start_idx)
    if start < 0:
        return None
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _fallback_parse_raw_waypoints(text: str) -> Optional[List[List[float]]]:
    part4 = _extract_part4(text)
    idx = part4.lower().find("waypoints")
    if idx < 0:
        return None
    search_idx = idx
    while True:
        bracket_start = part4.find("[", search_idx)
        if bracket_start < 0:
            return None
        payload = _extract_first_bracket_payload(part4, search_idx)
        if payload is None:
            return None
        search_idx = bracket_start + 1
        try:
            parsed = json.loads(payload)
        except Exception:
            try:
                parsed = ast.literal_eval(payload)
            except Exception:
                continue
        if not isinstance(parsed, list) or len(parsed) != 6:
            continue
        waypoints: List[List[float]] = []
        for item in parsed:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                break
            try:
                waypoints.append([float(item[0]), float(item[1])])
            except Exception:
                break
        if len(waypoints) == 6:
            return waypoints


try:
    from projects.covla_baseline.data.dataset import canonical_command as _repo_canonical_command
    from projects.covla_baseline.evaluate import parse_raw_command as _repo_parse_raw_command
    from projects.covla_baseline.evaluate import parse_raw_waypoints as _repo_parse_raw_waypoints

    canonical_command: Callable[[Any], str] = _repo_canonical_command
    parse_raw_command: Callable[[str], Optional[str]] = _repo_parse_raw_command
    parse_raw_waypoints: Callable[[str], Optional[List[List[float]]]] = _repo_parse_raw_waypoints
    PARSER_SOURCE = "projects.covla_baseline.evaluate"
except Exception as exc:  # pragma: no cover - only used when optional runtime deps are missing.
    canonical_command = _fallback_canonical_command
    parse_raw_command = _fallback_parse_raw_command
    parse_raw_waypoints = _fallback_parse_raw_waypoints
    PARSER_SOURCE = f"local_fallback:{type(exc).__name__}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no}: expected object row")
            rows.append(item)
    return rows


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def optional_mean(values: Sequence[Optional[float]]) -> Optional[float]:
    valid = [float(v) for v in values if v is not None]
    return mean(valid) if valid else None


def rate(numer: int, denom: int) -> float:
    return float(numer) / float(denom) if denom else 0.0


def safe_f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def sample_key(row: Dict[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("token") or row.get("global_index") or "")


def sort_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (str(row.get("split") or ""), str(row.get("sample_id") or ""), str(row.get("token") or ""))


def load_index(path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    duplicate_keys: List[str] = []
    rows = read_jsonl(path)
    for row in rows:
        for key_name in ("sample_id", "token"):
            key = str(row.get(key_name) or "")
            if not key:
                continue
            if key in index and index[key] is not row:
                duplicate_keys.append(key)
            index[key] = row
    return index, {"row_count": len(rows), "duplicate_match_keys": sorted(set(duplicate_keys))}


def match_reference(sample: Dict[str, Any], index: Dict[str, Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], str]:
    for key_name in ("sample_id", "token"):
        key = str(sample.get(key_name) or "")
        if key and key in index:
            return index[key], key_name
    return None, "unmatched"


def split_parts(text: str) -> Dict[int, str]:
    matches = list(PART_RE.finditer(text or ""))
    if not matches:
        return {}
    parts: Dict[int, str] = {}
    for idx, match in enumerate(matches):
        part_no = int(match.group(1))
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        parts[part_no] = text[start:end].strip()
    return parts


def normalize_sentence(sentence: str) -> str:
    return " ".join(tokenize(sentence))


def split_sentences(text: str) -> List[str]:
    pieces = [piece.strip(" \t\r\n-:;") for piece in SENTENCE_RE.split(text or "")]
    sentences = [piece for piece in pieces if normalize_sentence(piece)]
    return sentences


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text or "").lower())


def jaccard(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> float:
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def sentence_alignment(generated: str, reference: str) -> Dict[str, Any]:
    gen_sentences = split_sentences(generated)
    ref_sentences = split_sentences(reference)
    gen_tokens = [tokenize(sentence) for sentence in gen_sentences]
    ref_tokens = [tokenize(sentence) for sentence in ref_sentences]

    best_scores: List[float] = []
    for ref_tok in ref_tokens:
        best_scores.append(max((jaccard(ref_tok, gen_tok) for gen_tok in gen_tokens), default=0.0))
    alignment_score = mean(best_scores)
    missing_step_penalty = rate(sum(score < 0.25 for score in best_scores), len(best_scores))

    normalized = [normalize_sentence(sentence) for sentence in gen_sentences]
    duplicate_exact = sum(count - 1 for count in Counter(normalized).values() if count > 1)
    near_duplicate = 0
    for idx, tokens_i in enumerate(gen_tokens):
        for tokens_j in gen_tokens[idx + 1 :]:
            if jaccard(tokens_i, tokens_j) >= 0.85:
                near_duplicate += 1
                break
    redundancy_penalty = min(1.0, (duplicate_exact + near_duplicate) / max(len(gen_sentences), 1))
    overall = max(0.0, alignment_score * (1.0 - 0.35 * redundancy_penalty) * (1.0 - 0.50 * missing_step_penalty))
    return {
        "generated_sentence_count": len(gen_sentences),
        "reference_sentence_count": len(ref_sentences),
        "token_jaccard_sentence_alignment": alignment_score,
        "redundancy_penalty": redundancy_penalty,
        "missing_step_penalty": missing_step_penalty,
        "overall_score": overall,
    }


def adr_lite_scores(generated_text: str, reference_text: str) -> Dict[str, Any]:
    gen_parts = split_parts(generated_text)
    ref_parts = split_parts(reference_text)
    part_scores: Dict[str, Dict[str, Any]] = {}
    for part_no in (1, 2, 3):
        part_scores[f"part{part_no}"] = sentence_alignment(gen_parts.get(part_no, ""), ref_parts.get(part_no, ""))
    return {
        "parts": part_scores,
        "macro_alignment": mean([part_scores[f"part{part_no}"]["token_jaccard_sentence_alignment"] for part_no in (1, 2, 3)]),
        "macro_redundancy_penalty": mean([part_scores[f"part{part_no}"]["redundancy_penalty"] for part_no in (1, 2, 3)]),
        "macro_missing_step_penalty": mean([part_scores[f"part{part_no}"]["missing_step_penalty"] for part_no in (1, 2, 3)]),
        "overall_score": mean([part_scores[f"part{part_no}"]["overall_score"] for part_no in (1, 2, 3)]),
    }


def ordered_object_ids(text: str) -> List[str]:
    seen = set()
    ids: List[str] = []
    for match in OBJECT_ID_RE.finditer(text or ""):
        object_id = match.group(0).upper()
        if object_id not in seen:
            seen.add(object_id)
            ids.append(object_id)
    return ids


def normalize_ids(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    ids = []
    for value in values:
        text = str(value or "").strip().upper()
        if OBJECT_ID_RE.fullmatch(text):
            ids.append(text)
    return ids


def valid_object_ids(reference: Dict[str, Any]) -> List[str]:
    ids = set(normalize_ids(reference.get("critical_object_ids")))
    ids.update(normalize_ids(reference.get("infra_only_critical_ids")))
    table = reference.get("critical_object_table")
    if isinstance(table, list):
        for item in table:
            if not isinstance(item, dict):
                continue
            for key in ("label", "id"):
                value = str(item.get(key) or "").strip().upper()
                if OBJECT_ID_RE.fullmatch(value):
                    ids.add(value)
            aliases = item.get("aliases")
            if isinstance(aliases, list):
                ids.update(normalize_ids(aliases))
    return sorted(ids)


def object_sentence_map(text: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for sentence in split_sentences(text):
        for object_id in ordered_object_ids(sentence):
            mapping.setdefault(object_id, sentence)
    return mapping


def normalize_motion(text: str) -> Optional[str]:
    lowered = str(text or "").lower().replace("_", " ")
    if re.search(r"\bnear[- ]?zero|stationary|stopped|stop\b", lowered):
        return "stationary"
    if re.search(r"\bturning|turn left|left turn|turn right|right turn\b", lowered):
        return "turning"
    if re.search(r"\bgoing straight|moving straight|straight\b", lowered):
        return "going_straight"
    if re.search(r"\bmoving|in motion\b", lowered):
        return "moving"
    return None


def reference_slot_map(reference: Dict[str, Any], reference_text: str) -> Dict[str, Dict[str, Any]]:
    slots: Dict[str, Dict[str, Any]] = {}
    table = reference.get("critical_object_table")
    if isinstance(table, list):
        for item in table:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or item.get("id") or "").strip().upper()
            if not OBJECT_ID_RE.fullmatch(label):
                continue
            speed = item.get("instant_speed_mps")
            if speed is None:
                speed = item.get("smoothed_speed_mps")
            slots[label] = {
                "distance_m": item.get("distance_m"),
                "speed_mps": speed,
                "motion": normalize_motion(item.get("motion_state_refined") or item.get("motion_state")),
            }
    sentence_map = object_sentence_map(reference_text)
    for label, sentence in sentence_map.items():
        entry = slots.setdefault(label, {})
        dist = DISTANCE_RE.search(sentence)
        speed = SPEED_RE.search(sentence)
        if entry.get("distance_m") is None and dist:
            entry["distance_m"] = float(dist.group(1))
        if entry.get("speed_mps") is None and speed:
            entry["speed_mps"] = float(speed.group(1))
        if entry.get("motion") is None:
            entry["motion"] = normalize_motion(sentence)
    return slots


def generated_slots(text: str) -> Dict[str, Dict[str, Any]]:
    slots: Dict[str, Dict[str, Any]] = {}
    for label, sentence in object_sentence_map(text).items():
        dist = DISTANCE_RE.search(sentence)
        speed = SPEED_RE.search(sentence)
        slots[label] = {
            "distance_m": float(dist.group(1)) if dist else None,
            "speed_mps": float(speed.group(1)) if speed else None,
            "motion": normalize_motion(sentence),
        }
    return slots


def object_grounding(generated_text: str, reference: Dict[str, Any], reference_text: str) -> Dict[str, Any]:
    pred_ids = ordered_object_ids(generated_text)
    pred_set = set(pred_ids)
    critical_ids = set(normalize_ids(reference.get("critical_object_ids")))
    infra_ids = set(normalize_ids(reference.get("infra_only_critical_ids")))
    valid_ids = set(valid_object_ids(reference)) or critical_ids | infra_ids
    matched = pred_set & critical_ids
    precision = len(matched) / len(pred_set) if pred_set else (1.0 if not critical_ids else 0.0)
    recall = len(matched) / len(critical_ids) if critical_ids else 1.0
    hallucinated = pred_set - valid_ids
    infra_recall = len(pred_set & infra_ids) / len(infra_ids) if infra_ids else None

    ref_slots = reference_slot_map(reference, reference_text)
    gen_slots = generated_slots(generated_text)
    slot_ids = sorted((set(gen_slots) & set(ref_slots)) & critical_ids)

    def slot_rates(slot: str, tolerance: Optional[float] = None) -> Tuple[int, int, int]:
        total = 0
        parseable = 0
        matched_slot = 0
        for object_id in slot_ids:
            ref_value = ref_slots[object_id].get(slot)
            if ref_value is None:
                continue
            total += 1
            gen_value = gen_slots[object_id].get(slot)
            if gen_value is None:
                continue
            parseable += 1
            if tolerance is None:
                matched_slot += int(gen_value == ref_value)
            else:
                matched_slot += int(abs(float(gen_value) - float(ref_value)) <= tolerance)
        return total, parseable, matched_slot

    dist_total, dist_parseable, dist_match = slot_rates("distance_m", tolerance=2.0)
    speed_total, speed_parseable, speed_match = slot_rates("speed_mps", tolerance=1.5)
    motion_total, motion_parseable, motion_match = slot_rates("motion", tolerance=None)
    return {
        "generated_object_ids": pred_ids,
        "critical_object_ids": sorted(critical_ids),
        "infra_only_critical_ids": sorted(infra_ids),
        "precision": precision,
        "recall": recall,
        "f1": safe_f1(precision, recall),
        "infra_only_recall": infra_recall,
        "hallucinated_object_ids": sorted(hallucinated),
        "hallucinated_object_id_rate": len(hallucinated) / len(pred_set) if pred_set else 0.0,
        "slot_eval_object_count": len(slot_ids),
        "distance_parse_rate": rate(dist_parseable, dist_total),
        "distance_match_rate": rate(dist_match, dist_total),
        "speed_parse_rate": rate(speed_parseable, speed_total),
        "speed_match_rate": rate(speed_match, speed_total),
        "motion_parse_rate": rate(motion_parseable, motion_total),
        "motion_match_rate": rate(motion_match, motion_total),
    }


def action_consistency(sample: Dict[str, Any], reference: Dict[str, Any], generated_text: str) -> Dict[str, Any]:
    raw_command = parse_raw_command(generated_text)
    raw_command = canonical_command(raw_command or "UNKNOWN") if raw_command else None
    structured_command = canonical_command(sample.get("structured_pred_command") or sample.get("pred_command") or "UNKNOWN")
    gt_command = canonical_command(reference.get("command") or sample.get("gt_command") or "UNKNOWN")
    raw_waypoints = parse_raw_waypoints(generated_text)
    return {
        "raw_part4_command": raw_command,
        "structured_head_command": structured_command,
        "gt_command": gt_command,
        "raw_part4_command_parse_ok": raw_command is not None,
        "raw_part4_vs_gt": bool(raw_command == gt_command) if raw_command is not None else False,
        "raw_part4_vs_structured_head": bool(raw_command == structured_command) if raw_command is not None else False,
        "structured_head_vs_gt": bool(structured_command == gt_command),
        "raw_waypoint_parse_ok": raw_waypoints is not None,
        "raw_waypoint_count": len(raw_waypoints) if raw_waypoints is not None else 0,
    }


def evaluate_samples(samples: List[Dict[str, Any]], index: Dict[str, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    diagnostics: List[Dict[str, Any]] = []
    judge_source: List[Dict[str, Any]] = []
    for sample in sorted(samples, key=sort_key):
        reference, match_key = match_reference(sample, index)
        generated_text = str(sample.get("raw_generated_cot") or sample.get("generated_text") or "")
        if reference is None:
            diagnostics.append(
                {
                    "sample_id": sample.get("sample_id"),
                    "token": sample.get("token"),
                    "matched_reference": False,
                    "match_key": match_key,
                    "flags": ["unmatched_reference"],
                }
            )
            continue
        reference_text = str(reference.get("target_text") or "")
        trajectory_role = str(
            reference.get("trajectory_grounded_infra_role") or "unavailable"
        )
        adr = adr_lite_scores(generated_text, reference_text)
        grounding = object_grounding(generated_text, reference, reference_text)
        action = action_consistency(sample, reference, generated_text)
        flags = []
        if adr["overall_score"] < 0.25:
            flags.append("low_adr_lite")
        if grounding["recall"] < 0.5:
            flags.append("low_object_recall")
        if grounding["hallucinated_object_id_rate"] > 0.0:
            flags.append("hallucinated_object_id")
        if not action["raw_part4_vs_gt"]:
            flags.append("raw_action_mismatch_gt")
        if not action["structured_head_vs_gt"]:
            flags.append("structured_action_mismatch_gt")
        if not action["raw_waypoint_parse_ok"]:
            flags.append("raw_waypoint_unparseable")
        diag = {
            "sample_id": sample.get("sample_id"),
            "token": sample.get("token"),
            "split": sample.get("split") or reference.get("split"),
            "matched_reference": True,
            "match_key": match_key,
            "gt_command": action["gt_command"],
            "raw_part4_command": action["raw_part4_command"],
            "structured_head_command": action["structured_head_command"],
            "adr_lite": adr,
            "object_grounding": grounding,
            "action_consistency": action,
            "flags": flags,
        }
        diagnostics.append(diag)
        judge_source.append(
            {
                **diag,
                "generated_cot": generated_text,
                "reference_cot": reference_text,
            }
        )
    return diagnostics, judge_source


def evaluate_three_part_samples(
    samples: List[Dict[str, Any]],
    index: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Evaluate only the three language sections; action heads stay out of scope."""
    diagnostics: List[Dict[str, Any]] = []
    judge_source: List[Dict[str, Any]] = []
    for sample in sorted(samples, key=sort_key):
        reference, match_key = match_reference(sample, index)
        generated_text = str(
            sample.get("raw_generated_cot") or sample.get("generated_text") or ""
        )
        if reference is None:
            diagnostics.append(
                {
                    "sample_id": sample.get("sample_id"),
                    "token": sample.get("token"),
                    "matched_reference": False,
                    "match_key": match_key,
                    "flags": ["unmatched_reference"],
                }
            )
            continue

        reference_text = str(reference.get("target_text") or "")
        generated_parse = parse_three_part_cot(generated_text)
        reference_parse = parse_three_part_cot(reference_text)
        adr = adr_lite_scores(generated_text, reference_text)
        grounding = object_grounding(generated_text, reference, reference_text)
        trajectory_role = str(
            reference.get("trajectory_grounded_infra_role")
            or sample.get("trajectory_grounded_infra_role")
            or ""
        )
        flags: List[str] = []
        if not generated_parse["format_parse_ok"]:
            flags.append("three_part_format_error")
        if not reference_parse["format_parse_ok"]:
            flags.append("reference_three_part_contract_error")
        if adr["overall_score"] < 0.25:
            flags.append("low_adr_lite")
        if grounding["recall"] < 0.5:
            flags.append("low_object_recall")
        if grounding["hallucinated_object_id_rate"] > 0.0:
            flags.append("hallucinated_object_id")
        diag = {
            "sample_id": sample.get("sample_id"),
            "token": sample.get("token"),
            "split": sample.get("split") or reference.get("split"),
            "matched_reference": True,
            "match_key": match_key,
            "language_contract": THREE_PART_CONTRACT,
            "trajectory_grounded_infra_role": trajectory_role,
            "three_part_parse": generated_parse,
            "reference_three_part_parse_ok": reference_parse["format_parse_ok"],
            "adr_lite": adr,
            "object_grounding": grounding,
            "flags": flags,
        }
        diagnostics.append(diag)
        judge_source.append(
            {
                **diag,
                "generated_cot": generated_text,
                "reference_cot": reference_text,
            }
        )
    return diagnostics, judge_source


def aggregate_three_part_diagnostics(
    diagnostics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    matched = [row for row in diagnostics if row.get("matched_reference")]
    unmatched = [row for row in diagnostics if not row.get("matched_reference")]
    grounding = [row["object_grounding"] for row in matched]
    adr = [row["adr_lite"] for row in matched]
    infra_recalls = [
        item["infra_only_recall"]
        for item in grounding
        if item["infra_only_recall"] is not None
    ]

    pred_id_total = sum(len(item["generated_object_ids"]) for item in grounding)
    hallucinated_total = sum(
        len(item["hallucinated_object_ids"]) for item in grounding
    )
    critical_total = sum(len(item["critical_object_ids"]) for item in grounding)
    critical_hit_total = sum(
        len(set(item["generated_object_ids"]) & set(item["critical_object_ids"]))
        for item in grounding
    )
    pred_total = sum(len(set(item["generated_object_ids"])) for item in grounding)
    micro_precision = critical_hit_total / pred_total if pred_total else 0.0
    micro_recall = critical_hit_total / critical_total if critical_total else 0.0

    format_metrics = aggregate_three_part_parses(
        row["three_part_parse"] for row in matched
    )
    format_metrics["reference_contract_valid_rate"] = rate(
        sum(bool(row["reference_three_part_parse_ok"]) for row in matched),
        len(matched),
    )
    by_trajectory_role: Dict[str, Dict[str, Any]] = {}
    for trajectory_role in sorted(
        {str(row.get("trajectory_grounded_infra_role") or "unavailable") for row in matched}
    ):
        subset = [
            row
            for row in matched
            if str(row.get("trajectory_grounded_infra_role") or "unavailable")
            == trajectory_role
        ]
        subset_grounding = [row["object_grounding"] for row in subset]
        infra_only_recalls = [
            item["infra_only_recall"]
            for item in subset_grounding
            if item["infra_only_recall"] is not None
        ]
        generated_ids = sum(
            len(item["generated_object_ids"]) for item in subset_grounding
        )
        hallucinated_ids = sum(
            len(item["hallucinated_object_ids"]) for item in subset_grounding
        )
        by_trajectory_role[trajectory_role] = {
            "N": len(subset),
            "exact_three_part_parse_rate": rate(
                sum(bool(row["three_part_parse"]["format_parse_ok"]) for row in subset),
                len(subset),
            ),
            "future_target_leak_rate": rate(
                sum(bool(row["three_part_parse"].get("future_target_leak")) for row in subset),
                len(subset),
            ),
            "critical_evidence_alignment": mean(
                [row["adr_lite"]["parts"]["part2"]["overall_score"] for row in subset]
            ),
            "object_id_recall_macro": mean(
                [item["recall"] for item in subset_grounding]
            ),
            "infra_only_recall_macro": (
                mean(infra_only_recalls) if infra_only_recalls else None
            ),
            "infra_only_recall_available_count": len(infra_only_recalls),
            "hallucinated_object_id_rate_micro": (
                hallucinated_ids / generated_ids if generated_ids else 0.0
            ),
        }
    return {
        "counts": {
            "input_samples": len(diagnostics),
            "matched_samples": len(matched),
            "unmatched_samples": len(unmatched),
            "samples_with_infra_only_critical_ids": sum(
                bool(item["infra_only_critical_ids"]) for item in grounding
            ),
        },
        "language_contract": format_metrics,
        "by_trajectory_grounded_infra_role": by_trajectory_role,
        "adr_lite_alignment_three_parts": {
            "overall_score": mean([item["overall_score"] for item in adr]),
            "token_jaccard_sentence_alignment": mean(
                [item["macro_alignment"] for item in adr]
            ),
            "redundancy_penalty": mean(
                [item["macro_redundancy_penalty"] for item in adr]
            ),
            "missing_step_penalty": mean(
                [item["macro_missing_step_penalty"] for item in adr]
            ),
            "scene_understanding": mean(
                [item["parts"]["part1"]["overall_score"] for item in adr]
            ),
            "critical_evidence": mean(
                [item["parts"]["part2"]["overall_score"] for item in adr]
            ),
            "decision_reasoning": mean(
                [item["parts"]["part3"]["overall_score"] for item in adr]
            ),
        },
        "grounding": {
            "object_id_precision_macro": mean(
                [item["precision"] for item in grounding]
            ),
            "object_id_recall_macro": mean([item["recall"] for item in grounding]),
            "object_id_f1_macro": mean([item["f1"] for item in grounding]),
            "object_id_precision_micro": micro_precision,
            "object_id_recall_micro": micro_recall,
            "object_id_f1_micro": safe_f1(micro_precision, micro_recall),
            "infra_only_recall_macro": mean(infra_recalls),
            "infra_only_recall_available_count": len(infra_recalls),
            "hallucinated_object_id_rate_micro": (
                hallucinated_total / pred_id_total if pred_id_total else 0.0
            ),
            "distance_parse_rate_macro": mean(
                [item["distance_parse_rate"] for item in grounding]
            ),
            "distance_match_rate_macro": mean(
                [item["distance_match_rate"] for item in grounding]
            ),
            "speed_parse_rate_macro": mean(
                [item["speed_parse_rate"] for item in grounding]
            ),
            "speed_match_rate_macro": mean(
                [item["speed_match_rate"] for item in grounding]
            ),
            "motion_parse_rate_macro": mean(
                [item["motion_parse_rate"] for item in grounding]
            ),
            "motion_match_rate_macro": mean(
                [item["motion_match_rate"] for item in grounding]
            ),
        },
    }


def aggregate_diagnostics(diagnostics: List[Dict[str, Any]]) -> Dict[str, Any]:
    matched = [row for row in diagnostics if row.get("matched_reference")]
    unmatched = [row for row in diagnostics if not row.get("matched_reference")]
    grounding = [row["object_grounding"] for row in matched]
    action = [row["action_consistency"] for row in matched]
    adr = [row["adr_lite"] for row in matched]
    infra_recalls = [g["infra_only_recall"] for g in grounding if g["infra_only_recall"] is not None]

    pred_id_total = sum(len(g["generated_object_ids"]) for g in grounding)
    hallucinated_total = sum(len(g["hallucinated_object_ids"]) for g in grounding)
    critical_total = sum(len(g["critical_object_ids"]) for g in grounding)
    critical_hit_total = sum(len(set(g["generated_object_ids"]) & set(g["critical_object_ids"])) for g in grounding)
    pred_critical_total = sum(len(set(g["generated_object_ids"])) for g in grounding)
    micro_precision = critical_hit_total / pred_critical_total if pred_critical_total else 0.0
    micro_recall = critical_hit_total / critical_total if critical_total else 0.0

    by_command: Dict[str, Dict[str, Any]] = {}
    for command, rows in sorted(group_by(matched, lambda r: r["gt_command"]).items()):
        by_command[command] = {
            "count": len(rows),
            "adr_lite_overall_score": mean([r["adr_lite"]["overall_score"] for r in rows]),
            "object_f1": mean([r["object_grounding"]["f1"] for r in rows]),
            "raw_part4_vs_gt_rate": rate(sum(r["action_consistency"]["raw_part4_vs_gt"] for r in rows), len(rows)),
            "structured_head_vs_gt_rate": rate(sum(r["action_consistency"]["structured_head_vs_gt"] for r in rows), len(rows)),
        }

    return {
        "counts": {
            "input_samples": len(diagnostics),
            "matched_samples": len(matched),
            "unmatched_samples": len(unmatched),
            "samples_with_infra_only_critical_ids": sum(bool(g["infra_only_critical_ids"]) for g in grounding),
        },
        "adr_lite_alignment_parts1_3": {
            "overall_score": mean([a["overall_score"] for a in adr]),
            "token_jaccard_sentence_alignment": mean([a["macro_alignment"] for a in adr]),
            "redundancy_penalty": mean([a["macro_redundancy_penalty"] for a in adr]),
            "missing_step_penalty": mean([a["macro_missing_step_penalty"] for a in adr]),
            "part1_overall_score": mean([a["parts"]["part1"]["overall_score"] for a in adr]),
            "part2_overall_score": mean([a["parts"]["part2"]["overall_score"] for a in adr]),
            "part3_overall_score": mean([a["parts"]["part3"]["overall_score"] for a in adr]),
        },
        "grounding": {
            "object_id_precision_macro": mean([g["precision"] for g in grounding]),
            "object_id_recall_macro": mean([g["recall"] for g in grounding]),
            "object_id_f1_macro": mean([g["f1"] for g in grounding]),
            "object_id_precision_micro": micro_precision,
            "object_id_recall_micro": micro_recall,
            "object_id_f1_micro": safe_f1(micro_precision, micro_recall),
            "infra_only_recall_macro": mean(infra_recalls),
            "infra_only_recall_available_count": len(infra_recalls),
            "hallucinated_object_id_rate_micro": hallucinated_total / pred_id_total if pred_id_total else 0.0,
            "distance_parse_rate_macro": mean([g["distance_parse_rate"] for g in grounding]),
            "distance_match_rate_macro": mean([g["distance_match_rate"] for g in grounding]),
            "speed_parse_rate_macro": mean([g["speed_parse_rate"] for g in grounding]),
            "speed_match_rate_macro": mean([g["speed_match_rate"] for g in grounding]),
            "motion_parse_rate_macro": mean([g["motion_parse_rate"] for g in grounding]),
            "motion_match_rate_macro": mean([g["motion_match_rate"] for g in grounding]),
        },
        "action_consistency": {
            "raw_part4_command_parse_rate": rate(sum(a["raw_part4_command_parse_ok"] for a in action), len(action)),
            "raw_part4_command_vs_gt_rate": rate(sum(a["raw_part4_vs_gt"] for a in action), len(action)),
            "raw_part4_command_vs_structured_head_rate": rate(sum(a["raw_part4_vs_structured_head"] for a in action), len(action)),
            "structured_head_vs_gt_rate": rate(sum(a["structured_head_vs_gt"] for a in action), len(action)),
            "raw_waypoint_parse_rate": rate(sum(a["raw_waypoint_parse_ok"] for a in action), len(action)),
        },
        "by_gt_command": by_command,
    }


def group_by(rows: Sequence[Dict[str, Any]], key_fn: Callable[[Dict[str, Any]], str]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(key_fn(row))].append(row)
    return grouped


def validation_flags(samples: List[Dict[str, Any]], index_meta: Dict[str, Any], diagnostics: List[Dict[str, Any]]) -> Dict[str, Any]:
    sample_ids = [sample_key(row) for row in samples]
    duplicate_samples = sorted(key for key, count in Counter(sample_ids).items() if key and count > 1)
    matched_count = sum(bool(row.get("matched_reference")) for row in diagnostics)
    return {
        "deterministic_no_external_ml_dependencies": True,
        "judge_model_run": False,
        "full_raw_cot_omitted_from_main_jsonl": True,
        "matched_all_samples": matched_count == len(samples),
        "duplicate_sample_keys": duplicate_samples,
        "duplicate_index_match_keys": index_meta.get("duplicate_match_keys", []),
        "table_does_not_promote_shuffled_condition": True,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_table(path: Path, summary: Dict[str, Any]) -> None:
    metrics = summary["metrics"]
    counts = metrics["counts"]
    adr = metrics["adr_lite_alignment_parts1_3"]
    grounding = metrics["grounding"]
    action = metrics["action_consistency"]
    rows = [
        ("Samples matched", counts["matched_samples"]),
        ("ADR-lite Parts 1-3 overall", adr["overall_score"]),
        ("ADR-lite sentence Jaccard", adr["token_jaccard_sentence_alignment"]),
        ("ADR-lite missing-step penalty", adr["missing_step_penalty"]),
        ("Object ID F1 macro", grounding["object_id_f1_macro"]),
        ("Object ID recall macro", grounding["object_id_recall_macro"]),
        ("Infra-only recall macro", grounding["infra_only_recall_macro"]),
        ("Hallucinated object-ID rate", grounding["hallucinated_object_id_rate_micro"]),
        ("Raw Part 4 command parse rate", action["raw_part4_command_parse_rate"]),
        ("Raw Part 4 command vs GT", action["raw_part4_command_vs_gt_rate"]),
        ("Raw Part 4 command vs structured head", action["raw_part4_command_vs_structured_head_rate"]),
        ("Structured head vs GT", action["structured_head_vs_gt_rate"]),
        ("Raw waypoint parse rate", action["raw_waypoint_parse_rate"]),
    ]
    lines = [
        "# CoT Quality Summary",
        "",
        "Deterministic text-only metrics over the provided generated CoT samples and reference index.",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for name, value in rows:
        lines.append(f"| {name} | {fmt(value)} |")
    lines.extend(
        [
            "",
            "## By GT Command",
            "",
            "| GT command | Count | ADR-lite | Object F1 | Raw Part 4 vs GT | Structured vs GT |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for command, item in metrics["by_gt_command"].items():
        lines.append(
            "| "
            f"{command} | {item['count']} | {fmt(item['adr_lite_overall_score'])} | "
            f"{fmt(item['object_f1'])} | {fmt(item['raw_part4_vs_gt_rate'])} | "
            f"{fmt(item['structured_head_vs_gt_rate'])} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_three_part_table(path: Path, summary: Dict[str, Any]) -> None:
    metrics = summary["metrics"]
    counts = metrics["counts"]
    contract = metrics["language_contract"]
    adr = metrics["adr_lite_alignment_three_parts"]
    grounding = metrics["grounding"]
    rows = [
        ("Samples matched", counts["matched_samples"]),
        ("Exact three-part parse rate", contract["exact_three_part_parse_rate"]),
        ("Forbidden text action-field rate", contract["forbidden_text_action_field_rate"]),
        ("Scene understanding alignment", adr["scene_understanding"]),
        ("Critical evidence alignment", adr["critical_evidence"]),
        ("Decision reasoning alignment", adr["decision_reasoning"]),
        ("ADR-lite three-part overall", adr["overall_score"]),
        ("Object ID F1 macro", grounding["object_id_f1_macro"]),
        ("Object ID recall macro", grounding["object_id_recall_macro"]),
        ("Infra-only recall macro", grounding["infra_only_recall_macro"]),
        (
            "Hallucinated object-ID rate",
            grounding["hallucinated_object_id_rate_micro"],
        ),
    ]
    lines = [
        "# Three-Part Language Quality Summary",
        "",
        (
            "Deterministic diagnostics for Scene understanding, Critical evidence, "
            "and Decision reasoning. Commands and waypoints are produced by separate "
            "structured heads and are not compared with language text."
        ),
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {fmt(value)} |" for name, value in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def compact_diagnostic(row: Dict[str, Any]) -> Dict[str, Any]:
    if not row.get("matched_reference"):
        return row
    adr = row["adr_lite"]
    grounding = row["object_grounding"]
    action = row["action_consistency"]
    return {
        "sample_id": row.get("sample_id"),
        "token": row.get("token"),
        "split": row.get("split"),
        "matched_reference": True,
        "match_key": row.get("match_key"),
        "gt_command": action["gt_command"],
        "raw_part4_command": action["raw_part4_command"],
        "structured_head_command": action["structured_head_command"],
        "adr_lite_overall": adr["overall_score"],
        "part_scores": {
            name: {
                "alignment": part["token_jaccard_sentence_alignment"],
                "missing_step_penalty": part["missing_step_penalty"],
                "redundancy_penalty": part["redundancy_penalty"],
                "overall": part["overall_score"],
            }
            for name, part in adr["parts"].items()
        },
        "object_precision": grounding["precision"],
        "object_recall": grounding["recall"],
        "object_f1": grounding["f1"],
        "infra_only_recall": grounding["infra_only_recall"],
        "hallucinated_object_ids": grounding["hallucinated_object_ids"],
        "hallucinated_object_id_rate": grounding["hallucinated_object_id_rate"],
        "slot_match_rates": {
            "distance_parse": grounding["distance_parse_rate"],
            "distance_match": grounding["distance_match_rate"],
            "speed_parse": grounding["speed_parse_rate"],
            "speed_match": grounding["speed_match_rate"],
            "motion_parse": grounding["motion_parse_rate"],
            "motion_match": grounding["motion_match_rate"],
        },
        "raw_part4_command_parse_ok": action["raw_part4_command_parse_ok"],
        "raw_part4_vs_gt": action["raw_part4_vs_gt"],
        "raw_part4_vs_structured_head": action["raw_part4_vs_structured_head"],
        "structured_head_vs_gt": action["structured_head_vs_gt"],
        "raw_waypoint_parse_ok": action["raw_waypoint_parse_ok"],
        "flags": row["flags"],
    }


def compact_three_part_diagnostic(row: Dict[str, Any]) -> Dict[str, Any]:
    if not row.get("matched_reference"):
        return row
    adr = row["adr_lite"]
    grounding = row["object_grounding"]
    parsed = row["three_part_parse"]
    return {
        "sample_id": row.get("sample_id"),
        "token": row.get("token"),
        "split": row.get("split"),
        "matched_reference": True,
        "match_key": row.get("match_key"),
        "language_contract": row["language_contract"],
        "trajectory_grounded_infra_role": row.get(
            "trajectory_grounded_infra_role", "unavailable"
        ),
        "format_parse_ok": parsed["format_parse_ok"],
        "section_nonempty": parsed["section_nonempty"],
        "parse_errors": parsed["errors"],
        "future_target_leak": bool(parsed.get("future_target_leak")),
        "future_target_leak_matches": parsed.get("future_target_leak_matches", {}),
        "adr_lite_overall": adr["overall_score"],
        "section_scores": {
            section_name: {
                "alignment": adr["parts"][f"part{index}"][
                    "token_jaccard_sentence_alignment"
                ],
                "missing_step_penalty": adr["parts"][f"part{index}"][
                    "missing_step_penalty"
                ],
                "redundancy_penalty": adr["parts"][f"part{index}"][
                    "redundancy_penalty"
                ],
                "overall": adr["parts"][f"part{index}"]["overall_score"],
            }
            for index, section_name in enumerate(THREE_PART_SECTION_KEYS, start=1)
        },
        "object_precision": grounding["precision"],
        "object_recall": grounding["recall"],
        "object_f1": grounding["f1"],
        "infra_only_recall": grounding["infra_only_recall"],
        "hallucinated_object_ids": grounding["hallucinated_object_ids"],
        "hallucinated_object_id_rate": grounding["hallucinated_object_id_rate"],
        "slot_match_rates": {
            "distance_parse": grounding["distance_parse_rate"],
            "distance_match": grounding["distance_match_rate"],
            "speed_parse": grounding["speed_parse_rate"],
            "speed_match": grounding["speed_match_rate"],
            "motion_parse": grounding["motion_parse_rate"],
            "motion_match": grounding["motion_match_rate"],
        },
        "flags": row["flags"],
    }


def select_judge_rows(rows: Sequence[Dict[str, Any]], limit: int = 100) -> List[Dict[str, Any]]:
    strata = group_by(
        [row for row in rows if row.get("matched_reference")],
        lambda row: (
            f"{row['gt_command']}|infra={bool(row['object_grounding']['infra_only_critical_ids'])}|"
            f"action_error={not row['action_consistency']['raw_part4_vs_gt']}"
        ),
    )
    ordered_strata = {key: sorted(value, key=sort_key) for key, value in sorted(strata.items())}
    selected: List[Dict[str, Any]] = []
    used = set()
    while len(selected) < limit:
        added = False
        for key in sorted(ordered_strata):
            bucket = ordered_strata[key]
            while bucket and sample_key(bucket[0]) in used:
                bucket.pop(0)
            if not bucket:
                continue
            row = bucket.pop(0)
            selected.append(row)
            used.add(sample_key(row))
            added = True
            if len(selected) >= limit:
                break
        if not added:
            break
    return selected


def judge_package_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rubric = [
        "scene correctness",
        "object grounding",
        "V2X grounding",
        "decision causality",
        "action consistency",
        "hallucination control",
        "overall score",
    ]
    packaged = []
    for row in select_judge_rows(rows, limit=100):
        packaged.append(
            {
                "sample_id": row.get("sample_id"),
                "token": row.get("token"),
                "split": row.get("split"),
                "generated_cot": row["generated_cot"],
                "reference_cot": row["reference_cot"],
                "raw_part4_command": row["action_consistency"]["raw_part4_command"],
                "structured_head_command": row["action_consistency"]["structured_head_command"],
                "gt_command": row["action_consistency"]["gt_command"],
                "critical_object_ids": row["object_grounding"]["critical_object_ids"],
                "infra_only_critical_ids": row["object_grounding"]["infra_only_critical_ids"],
                "rubric_dimensions": rubric,
                "judge_instruction": "Score each rubric dimension deterministically from 1 to 5; do not infer from hidden metadata.",
            }
        )
    return packaged


def select_three_part_judge_rows(
    rows: Sequence[Dict[str, Any]], limit: int = 100
) -> List[Dict[str, Any]]:
    strata = group_by(
        [row for row in rows if row.get("matched_reference")],
        lambda row: (
            f"infra={bool(row['object_grounding']['infra_only_critical_ids'])}|"
            f"format_ok={bool(row['three_part_parse']['format_parse_ok'])}"
        ),
    )
    buckets = {key: sorted(value, key=sort_key) for key, value in sorted(strata.items())}
    selected: List[Dict[str, Any]] = []
    while len(selected) < limit:
        added = False
        for key in sorted(buckets):
            if not buckets[key]:
                continue
            selected.append(buckets[key].pop(0))
            added = True
            if len(selected) >= limit:
                break
        if not added:
            break
    return selected


def three_part_judge_package_rows(
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rubric = [
        "scene understanding correctness",
        "critical evidence grounding",
        "V2X grounding",
        "decision causality",
        "hallucination control",
        "overall score",
    ]
    packaged: List[Dict[str, Any]] = []
    for row in select_three_part_judge_rows(rows, limit=100):
        packaged.append(
            {
                "sample_id": row.get("sample_id"),
                "token": row.get("token"),
                "split": row.get("split"),
                "language_contract": THREE_PART_CONTRACT,
                "generated_cot": row["generated_cot"],
                "reference_cot": row["reference_cot"],
                "critical_object_ids": row["object_grounding"][
                    "critical_object_ids"
                ],
                "infra_only_critical_ids": row["object_grounding"][
                    "infra_only_critical_ids"
                ],
                "rubric_dimensions": rubric,
                "judge_instruction": (
                    "Score each language rubric dimension from 1 to 5. Do not assess "
                    "or infer commands or numeric waypoints from the language text."
                ),
            }
        )
    return packaged


def build_summary(args: argparse.Namespace) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    samples = read_jsonl(args.samples)
    index, index_meta = load_index(args.index)
    contract = str(getattr(args, "contract", "legacy_four_part"))
    if contract == "three_part":
        diagnostics, judge_source = evaluate_three_part_samples(samples, index)
        metrics = aggregate_three_part_diagnostics(diagnostics)
    elif contract == "legacy_four_part":
        diagnostics, judge_source = evaluate_samples(samples, index)
        metrics = aggregate_diagnostics(diagnostics)
    else:
        raise ValueError(f"Unknown language contract: {contract}")
    flags = validation_flags(samples, index_meta, diagnostics)
    if contract == "three_part":
        flags["language_action_agreement_computed"] = False
    summary = {
        "run_metadata": {
            "created_at_utc": utc_now(),
            "script": str(Path(__file__).resolve()),
            "samples": str(args.samples),
            "index": str(args.index),
            "parser_source": (
                "projects.covla_baseline.three_part_cot"
                if contract == "three_part"
                else PARSER_SOURCE
            ),
            "language_contract": (
                THREE_PART_CONTRACT if contract == "three_part" else "legacy_four_part"
            ),
            "judge_model_run": False,
        },
        "counts": {
            "sample_rows": len(samples),
            "index_rows": index_meta["row_count"],
        },
        "validation_flags": flags,
        "metrics": metrics,
    }
    return summary, diagnostics, judge_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate deterministic CoT quality metrics for generated full-val JSONL.")
    parser.add_argument("--samples", type=Path, required=True, help="Generated CoT JSONL.")
    parser.add_argument("--index", type=Path, required=True, help="Usable index JSONL with target_text and GT metadata.")
    parser.add_argument("--output-json", type=Path, required=True, help="Summary JSON output.")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Per-sample diagnostic JSONL output.")
    parser.add_argument("--output-table", type=Path, required=True, help="Markdown table summary output.")
    parser.add_argument("--judge-sample-jsonl", type=Path, default=None, help="Optional deterministic stratified 100-row judge package.")
    parser.add_argument(
        "--contract",
        choices=("legacy_four_part", "three_part"),
        default="legacy_four_part",
        help="Language contract used by deterministic parsers and reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, diagnostics, judge_source = build_summary(args)
    write_json(args.output_json, summary)
    if args.contract == "three_part":
        write_jsonl(
            args.output_jsonl,
            (compact_three_part_diagnostic(row) for row in diagnostics),
        )
        write_three_part_table(args.output_table, summary)
    else:
        write_jsonl(args.output_jsonl, (compact_diagnostic(row) for row in diagnostics))
        write_table(args.output_table, summary)
    judge_count = 0
    if args.judge_sample_jsonl is not None:
        judge_rows = (
            three_part_judge_package_rows(judge_source)
            if args.contract == "three_part"
            else judge_package_rows(judge_source)
        )
        judge_count = len(judge_rows)
        write_jsonl(args.judge_sample_jsonl, judge_rows)
    counts = summary["metrics"]["counts"]
    print(
        "[DONE] CoT quality metrics written; "
        f"matched={counts['matched_samples']}/{counts['input_samples']} "
        f"judge_rows={judge_count}"
    )


if __name__ == "__main__":
    main()
