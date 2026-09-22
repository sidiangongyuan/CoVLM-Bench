"""Strict parsing and aggregation for the three-part CoVLM reasoning contract.

The language target contains scene understanding, critical evidence, and
decision reasoning only. Command and waypoint outputs belong to independent
structured heads and are deliberately outside this module.
"""
from __future__ import annotations

from collections import Counter
import re
from typing import Any, Dict, Iterable, Mapping


CONTRACT_NAME = "three_part_reasoning_v1"
SECTION_KEYS = (
    "scene_understanding",
    "critical_evidence",
    "decision_reasoning",
)
SECTION_DISPLAY_NAMES = {
    "scene_understanding": "Scene understanding",
    "critical_evidence": "Critical evidence",
    "decision_reasoning": "Decision reasoning",
}

# The stored training target uses ``Scene overview`` and ``V2X-aware critical
# objects``. They are the concrete labels for the paper's semantic sections
# ``Scene understanding`` and ``Critical evidence``.
_ALLOWED_HEADER_LABELS = {
    1: {"scene overview", "scene understanding"},
    2: {
        "v2x-aware critical objects",
        "v2x aware critical objects",
        "v2x-aware critical evidence",
        "v2x aware critical evidence",
        "critical evidence",
        "critical objects visible from the ego view",
    },
    3: {"decision reasoning"},
}
_EXACT_HEADER_RE = re.compile(
    r"(?im)^\s*Part\s*([1-3])\s*-\s*([^:\n]+?)\s*:\s*$"
)
_ANY_PART_HEADER_RE = re.compile(r"(?im)^\s*Part\s*(\d+)\b[^\n]*$")
_ACTION_FIELD_RE = re.compile(r"(?im)^\s*Action\s*:")
_COMMAND_FIELD_RE = re.compile(r"(?im)^\s*Command\s*:")
_WAYPOINT_FIELD_RE = re.compile(r"(?im)^\s*Waypoints?\s*:")

# Decision reasoning may legitimately describe a planned path or a semantic
# action. What must not appear is supervision provenance (for example, "GT
# command") or a serialized future target that could duplicate the structured
# heads. Keep this audit separate from the three-section format contract.
_FUTURE_TARGET_LEAK_PATTERNS = {
    "gt_or_ground_truth_metadata": re.compile(
        r"(?i)(?<![A-Za-z0-9])GT(?![A-Za-z0-9])"
        r"|\bground[\s-]?truth\b"
    ),
    "target_action_metadata": re.compile(
        r"(?i)\btarget\s+(?:command|action|trajectory|waypoints?)\b"
    ),
    "future_waypoint_metadata": re.compile(r"(?i)\bfuture\s+waypoints?\b"),
    "planned_trajectory_provenance": re.compile(
        r"(?i)\b(?:confirmed|indicated|specified|given)\s+by\s+"
        r"(?:the\s+)?planned\s+trajectory\b"
    ),
    "serialized_waypoint_coordinates": re.compile(
        r"\[\s*\[\s*[+-]?\d+(?:\.\d+)?\s*,\s*[+-]?\d+(?:\.\d+)?\s*\]"
        r"\s*,\s*\[\s*[+-]?\d+(?:\.\d+)?\s*,\s*[+-]?\d+(?:\.\d+)?"
    ),
}


def _normalized_label(value: str) -> str:
    return " ".join(str(value or "").strip().lower().replace("\u2013", "-").split())


def parse_three_part_cot(text: str) -> Dict[str, Any]:
    """Parse exactly one ordered, non-empty three-section reasoning response."""
    raw = str(text or "")
    exact_matches = list(_EXACT_HEADER_RE.finditer(raw))
    any_part_matches = list(_ANY_PART_HEADER_RE.finditer(raw))
    errors: list[str] = []

    if exact_matches and raw[: exact_matches[0].start()].strip():
        errors.append("unexpected_preamble")

    part_numbers = [int(match.group(1)) for match in exact_matches]
    all_part_numbers = [int(match.group(1)) for match in any_part_matches]
    if part_numbers != [1, 2, 3]:
        errors.append("expected_exactly_one_ordered_part_1_2_3_sequence")
    if len(any_part_matches) != 3 or all_part_numbers != [1, 2, 3]:
        errors.append("unexpected_or_duplicate_part_header")

    sections = {key: "" for key in SECTION_KEYS}
    header_labels: Dict[str, str] = {}
    if part_numbers == [1, 2, 3]:
        for index, match in enumerate(exact_matches):
            part_number = int(match.group(1))
            key = SECTION_KEYS[part_number - 1]
            label = _normalized_label(match.group(2))
            header_labels[key] = label
            if label not in _ALLOWED_HEADER_LABELS[part_number]:
                errors.append(f"unexpected_part_{part_number}_label")
            start = match.end()
            end = exact_matches[index + 1].start() if index + 1 < 3 else len(raw)
            sections[key] = raw[start:end].strip()

    section_nonempty = {key: bool(value.strip()) for key, value in sections.items()}
    for key, nonempty in section_nonempty.items():
        if not nonempty:
            errors.append(f"empty_{key}")

    forbidden = {
        "part4_header": any(number == 4 for number in all_part_numbers),
        "action_field": bool(_ACTION_FIELD_RE.search(raw)),
        "command_field": bool(_COMMAND_FIELD_RE.search(raw)),
        "waypoint_field": bool(_WAYPOINT_FIELD_RE.search(raw)),
    }
    for name, present in forbidden.items():
        if present:
            errors.append(f"forbidden_{name}")

    future_target_leak_matches = {
        name: match.group(0)
        for name, pattern in _FUTURE_TARGET_LEAK_PATTERNS.items()
        if (match := pattern.search(raw)) is not None
    }

    # Preserve order while avoiding repeated error labels.
    errors = list(dict.fromkeys(errors))
    return {
        "contract": CONTRACT_NAME,
        "format_parse_ok": not errors,
        "canonical_sections": list(SECTION_KEYS),
        "canonical_display_names": dict(SECTION_DISPLAY_NAMES),
        "part_numbers": part_numbers,
        "all_part_numbers": all_part_numbers,
        "header_labels": header_labels,
        "sections": sections,
        "section_nonempty": section_nonempty,
        "section_word_counts": {
            key: len(value.split()) for key, value in sections.items()
        },
        "forbidden_text_action_fields": forbidden,
        "future_target_leak": bool(future_target_leak_matches),
        "future_target_leak_matches": future_target_leak_matches,
        "errors": errors,
    }


def aggregate_three_part_parses(
    parses: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    rows = [dict(item) for item in parses]
    count = len(rows)
    denom = max(count, 1)
    error_counts = Counter(
        str(error) for row in rows for error in row.get("errors", [])
    )
    section_nonempty_rate = {
        key: sum(bool(row.get("section_nonempty", {}).get(key)) for row in rows)
        / denom
        for key in SECTION_KEYS
    }
    average_section_words = {
        key: sum(
            int(row.get("section_word_counts", {}).get(key, 0)) for row in rows
        )
        / denom
        for key in SECTION_KEYS
    }
    forbidden_count = sum(
        any(bool(value) for value in row.get("forbidden_text_action_fields", {}).values())
        for row in rows
    )
    future_target_leak_count = sum(
        bool(row.get("future_target_leak")) for row in rows
    )
    future_target_pattern_counts = Counter(
        str(name)
        for row in rows
        for name in row.get("future_target_leak_matches", {})
    )
    return {
        "contract": CONTRACT_NAME,
        "generated_count": count,
        "exact_three_part_parse_rate": sum(
            bool(row.get("format_parse_ok")) for row in rows
        )
        / denom,
        "section_nonempty_rate": section_nonempty_rate,
        "average_section_words": average_section_words,
        "forbidden_text_action_field_rate": forbidden_count / denom,
        "future_target_leak_rate": future_target_leak_count / denom,
        "future_target_leak_count": future_target_leak_count,
        "future_target_leak_pattern_counts": dict(
            sorted(future_target_pattern_counts.items())
        ),
        "parse_error_counts": dict(sorted(error_counts.items())),
        "text_command_agreement": "not_applicable_not_computed",
        "text_waypoint_agreement": "not_applicable_not_computed",
    }


def is_three_part_cot(text: str) -> bool:
    return bool(parse_three_part_cot(text)["format_parse_ok"])
