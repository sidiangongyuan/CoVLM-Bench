#!/usr/bin/env python3
"""Post-process action-first CoT samples into appendix-ready diagnostics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from projects.covla_baseline.data.dataset import canonical_command
from projects.covla_baseline.evaluate import parse_rate, parse_raw_command, parse_raw_waypoints
from projects.covla_baseline.three_part_cot import (
    CONTRACT_NAME as THREE_PART_CONTRACT,
    aggregate_three_part_parses,
    parse_three_part_cot,
)


INFRA_KEYWORDS = ("infra", "infrastructure", "v2x", "occluded", "hidden", "blind spot", "i1", "i2", "i3")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_index_metadata(path: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return records
    for item in read_jsonl(path):
        sample_id = str(item.get("sample_id", ""))
        token = str(item.get("token", ""))
        if sample_id:
            records[sample_id] = item
        if token:
            records[token] = item
    return records


def command_waypoint_metrics(raw_wp: Optional[List[List[float]]], structured_wp: Any) -> Optional[Dict[str, float]]:
    if raw_wp is None or not isinstance(structured_wp, list) or len(structured_wp) != 6:
        return None
    dists: List[float] = []
    for raw, pred in zip(raw_wp, structured_wp):
        if not isinstance(pred, (list, tuple)) or len(pred) < 2:
            return None
        dists.append(math.hypot(float(raw[0]) - float(pred[0]), float(raw[1]) - float(pred[1])))
    return {"ade": sum(dists) / max(len(dists), 1), "fde": dists[-1] if dists else 0.0}


def mention_any(text: str, ids: Iterable[str]) -> bool:
    for object_id in ids:
        if not object_id:
            continue
        if re.search(rf"\b{re.escape(str(object_id))}\b", text):
            return True
    return False


def infra_reference(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in INFRA_KEYWORDS)


def short_excerpt(text: str, marker: str, limit: int = 420) -> str:
    idx = text.find(marker)
    if idx < 0:
        idx = 0
    excerpt = " ".join(text[idx : idx + limit].split())
    if len(text) > idx + limit:
        excerpt += " ..."
    return excerpt


def choose_examples(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    wanted: List[Tuple[str, Any]] = [
        ("raw CoT agrees with structured action", lambda r: r["raw_command"] and r["raw_command"] == r["structured_command"]),
        (
            "raw CoT disagrees with structured action but structured matches GT",
            lambda r: r["raw_command"]
            and r["raw_command"] != r["structured_command"]
            and r["structured_command"] == r["gt_command"],
        ),
        (
            "raw CoT/structured both wrong",
            lambda r: r["raw_command"]
            and r["raw_command"] != r["gt_command"]
            and r["structured_command"] != r["gt_command"],
        ),
        ("V2X/infra-critical example", lambda r: bool(r["infra_only_critical_ids"]) and r["infra_reference"]),
    ]
    selected: List[Dict[str, Any]] = []
    used = set()
    for label, predicate in wanted:
        match = next((r for r in records if r["sample_id"] not in used and predicate(r)), None)
        if match is None:
            continue
        used.add(match["sample_id"])
        selected.append(
            {
                "case": label,
                "sample_id": match["sample_id"],
                "token": match["token"],
                "gt_command": match["gt_command"],
                "structured_command": match["structured_command"],
                "raw_command": match["raw_command"],
                "format_parse_ok": match["format_parse_ok"],
                "critical_object_ids": match["critical_object_ids"],
                "infra_only_critical_ids": match["infra_only_critical_ids"],
                "excerpt": short_excerpt(match["raw_generated_cot"], "Part 4" if match["format_parse_ok"] else "Part 1"),
            }
        )
    return selected


def missing_example_cases(selected: List[Dict[str, Any]]) -> List[str]:
    required = {
        "raw CoT agrees with structured action",
        "raw CoT disagrees with structured action but structured matches GT",
        "raw CoT/structured both wrong",
        "V2X/infra-critical example",
    }
    present = {str(item.get("case")) for item in selected}
    return sorted(required - present)


def rate(numer: int, denom: int) -> float:
    return float(numer) / float(denom) if denom else 0.0


def build_legacy_diagnostics(
    metrics_path: Path,
    samples_path: Path,
    index_path: Path,
    output_jsonl: Optional[Path],
) -> Dict[str, Any]:
    base_metrics = read_json(metrics_path)
    rows = read_jsonl(samples_path)
    index = load_index_metadata(index_path)
    records: List[Dict[str, Any]] = []
    raw_wp_ades: List[float] = []
    raw_wp_fdes: List[float] = []

    for row in rows:
        text = str(row.get("raw_generated_cot", ""))
        structured_command = canonical_command(row.get("structured_pred_command", "UNKNOWN"))
        gt_command = canonical_command(row.get("gt_command", "UNKNOWN"))
        raw_command = parse_raw_command(text)
        raw_wp = parse_raw_waypoints(text)
        wp_metrics = command_waypoint_metrics(raw_wp, row.get("structured_pred_waypoints"))
        if wp_metrics is not None:
            raw_wp_ades.append(wp_metrics["ade"])
            raw_wp_fdes.append(wp_metrics["fde"])
        meta = index.get(str(row.get("sample_id", ""))) or index.get(str(row.get("token", ""))) or {}
        critical_ids = [str(x) for x in meta.get("critical_object_ids", []) if x]
        infra_ids = [str(x) for x in meta.get("infra_only_critical_ids", []) if x]
        records.append(
            {
                "global_index": row.get("global_index"),
                "sample_id": row.get("sample_id"),
                "token": row.get("token"),
                "split": row.get("split"),
                "raw_generated_cot": text,
                "structured_command": structured_command,
                "gt_command": gt_command,
                "raw_command": raw_command,
                "format_parse_ok": parse_rate(text),
                "raw_command_parse_ok": raw_command is not None,
                "raw_command_agrees_with_structured": raw_command == structured_command,
                "raw_command_agrees_with_gt": raw_command == gt_command,
                "structured_command_agrees_with_gt": structured_command == gt_command,
                "raw_waypoints_parse_ok": raw_wp is not None,
                "raw_waypoint_vs_structured_ade": None if wp_metrics is None else wp_metrics["ade"],
                "raw_waypoint_vs_structured_fde": None if wp_metrics is None else wp_metrics["fde"],
                "critical_object_ids": critical_ids,
                "infra_only_critical_ids": infra_ids,
                "critical_object_mentioned": mention_any(text, critical_ids) if critical_ids else None,
                "infra_only_critical_object_mentioned": mention_any(text, infra_ids) if infra_ids else None,
                "infra_reference": infra_reference(text),
                "ade": row.get("ade"),
                "fde": row.get("fde"),
            }
        )

    sample_count = len(records)
    critical_available = [r for r in records if r["critical_object_ids"]]
    infra_available = [r for r in records if r["infra_only_critical_ids"]]
    metrics = {
        "sample_count": sample_count,
        "metric_scope": "sparse 1024-token action-first examples",
        "source_validation_count": base_metrics.get("count"),
        "source_examples_every_n": base_metrics.get("examples_every_n"),
        "source_examples_global_indices": base_metrics.get("examples_global_indices"),
        "raw_generated_cot_format_parse_rate": rate(sum(r["format_parse_ok"] for r in records), sample_count),
        "raw_part4_command_parse_rate": rate(sum(r["raw_command_parse_ok"] for r in records), sample_count),
        "raw_part4_command_vs_structured_agreement_rate": rate(
            sum(r["raw_command_agrees_with_structured"] for r in records), sample_count
        ),
        "raw_part4_command_vs_gt_agreement_rate": rate(
            sum(r["raw_command_agrees_with_gt"] for r in records), sample_count
        ),
        "structured_command_vs_gt_accuracy_on_diagnostic_samples": rate(
            sum(r["structured_command_agrees_with_gt"] for r in records), sample_count
        ),
        "raw_part4_waypoint_parse_rate": rate(sum(r["raw_waypoints_parse_ok"] for r in records), sample_count),
        "raw_waypoint_vs_structured_ADE": (
            sum(raw_wp_ades) / len(raw_wp_ades) if raw_wp_ades else "not_available"
        ),
        "raw_waypoint_vs_structured_FDE": (
            sum(raw_wp_fdes) / len(raw_wp_fdes) if raw_wp_fdes else "not_available"
        ),
        "critical_object_mention_rate": (
            rate(sum(bool(r["critical_object_mentioned"]) for r in critical_available), len(critical_available))
            if critical_available
            else "not_available"
        ),
        "critical_object_metric_scope": (
            "sparse examples joined to index critical_object_ids"
            if critical_available
            else "not_available: samples lack critical_object_ids and index join failed"
        ),
        "infra_only_critical_object_mention_rate": (
            rate(
                sum(bool(r["infra_only_critical_object_mentioned"]) for r in infra_available),
                len(infra_available),
            )
            if infra_available
            else "not_available"
        ),
        "infra_only_critical_metric_scope": (
            "sparse examples joined to index infra_only_critical_ids"
            if infra_available
            else "not_available: no infra_only_critical_ids in matched examples"
        ),
        "infra_reference_rate": rate(sum(r["infra_reference"] for r in records), sample_count),
        "infra_reference_rule": "case-insensitive keyword/ID match over infra, infrastructure, v2x, occluded, hidden, blind spot, I1/I2/I3",
    }
    if output_jsonl is not None:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl.open("w", encoding="utf-8") as f:
            for record in records:
                out = dict(record)
                out.pop("raw_generated_cot", None)
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
    examples = choose_examples(records)
    return {
        "run_metadata": {
            "stage": "cot_diagnostics_postprocess",
            "created_at_utc": utc_now(),
            "metrics_path": str(metrics_path),
            "samples_path": str(samples_path),
            "index_path": str(index_path),
            "gpu_inference_started": False,
        },
        "metrics": metrics,
        "qualitative_examples": examples,
        "missing_qualitative_example_cases": missing_example_cases(examples),
    }


def choose_three_part_examples(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    wanted: List[Tuple[str, Any]] = [
        (
            "valid three-part reasoning",
            lambda row: row["format_parse_ok"],
        ),
        (
            "three-part format failure",
            lambda row: not row["format_parse_ok"],
        ),
        (
            "infra-critical evidence grounded",
            lambda row: bool(row["infra_only_critical_ids"])
            and bool(row["infra_only_critical_evidence_mentioned"]),
        ),
    ]
    selected: List[Dict[str, Any]] = []
    used = set()
    for label, predicate in wanted:
        match = next(
            (
                row
                for row in records
                if row["sample_id"] not in used and predicate(row)
            ),
            None,
        )
        if match is None:
            continue
        used.add(match["sample_id"])
        selected.append(
            {
                "case": label,
                "sample_id": match["sample_id"],
                "token": match["token"],
                "format_parse_ok": match["format_parse_ok"],
                "parse_errors": match["parse_errors"],
                "critical_object_ids": match["critical_object_ids"],
                "infra_only_critical_ids": match["infra_only_critical_ids"],
                "excerpt": short_excerpt(match["raw_generated_cot"], "Part 1"),
            }
        )
    return selected


def build_three_part_diagnostics(
    metrics_path: Path,
    samples_path: Path,
    index_path: Path,
    output_jsonl: Optional[Path],
) -> Dict[str, Any]:
    base_metrics = read_json(metrics_path)
    rows = read_jsonl(samples_path)
    index = load_index_metadata(index_path)
    records: List[Dict[str, Any]] = []
    parses: List[Dict[str, Any]] = []

    for row in rows:
        text = str(row.get("raw_generated_cot") or row.get("generated_text") or "")
        parsed = parse_three_part_cot(text)
        parses.append(parsed)
        sections = parsed["sections"]
        evidence_text = str(sections["critical_evidence"])
        reasoning_text = str(sections["decision_reasoning"])
        meta = (
            index.get(str(row.get("sample_id", "")))
            or index.get(str(row.get("token", "")))
            or {}
        )
        critical_ids = [str(value) for value in meta.get("critical_object_ids", []) if value]
        infra_ids = [
            str(value) for value in meta.get("infra_only_critical_ids", []) if value
        ]
        records.append(
            {
                "global_index": row.get("global_index"),
                "sample_id": row.get("sample_id"),
                "token": row.get("token"),
                "split": row.get("split"),
                "raw_generated_cot": text,
                "language_contract": THREE_PART_CONTRACT,
                "format_parse_ok": parsed["format_parse_ok"],
                "parse_errors": parsed["errors"],
                "section_nonempty": parsed["section_nonempty"],
                "section_word_counts": parsed["section_word_counts"],
                "forbidden_text_action_fields": parsed[
                    "forbidden_text_action_fields"
                ],
                "critical_object_ids": critical_ids,
                "infra_only_critical_ids": infra_ids,
                "critical_evidence_object_mentioned": (
                    mention_any(evidence_text, critical_ids) if critical_ids else None
                ),
                "infra_only_critical_evidence_mentioned": (
                    mention_any(evidence_text, infra_ids) if infra_ids else None
                ),
                "decision_reasoning_object_mentioned": (
                    mention_any(reasoning_text, critical_ids) if critical_ids else None
                ),
                "critical_evidence_infra_reference": infra_reference(evidence_text),
            }
        )

    sample_count = len(records)
    critical_available = [row for row in records if row["critical_object_ids"]]
    infra_available = [row for row in records if row["infra_only_critical_ids"]]
    contract_metrics = aggregate_three_part_parses(parses)
    contract_metrics.update(
        {
            "scope": "generated language samples",
            "reporting_note": (
                "Only Scene understanding, Critical evidence, and Decision reasoning "
                "are evaluated. Structured commands and waypoints are outside the "
                "language contract and were not compared with text."
            ),
        }
    )
    metrics = {
        "sample_count": sample_count,
        "metric_scope": "three-part generated language samples",
        "source_validation_count": base_metrics.get("count"),
        "source_examples_every_n": base_metrics.get("examples_every_n"),
        "source_examples_global_indices": base_metrics.get(
            "examples_global_indices"
        ),
        "language_contract": contract_metrics,
        "critical_evidence_object_mention_rate": (
            rate(
                sum(bool(row["critical_evidence_object_mentioned"]) for row in critical_available),
                len(critical_available),
            )
            if critical_available
            else "not_available"
        ),
        "decision_reasoning_object_mention_rate": (
            rate(
                sum(bool(row["decision_reasoning_object_mentioned"]) for row in critical_available),
                len(critical_available),
            )
            if critical_available
            else "not_available"
        ),
        "infra_only_critical_evidence_mention_rate": (
            rate(
                sum(
                    bool(row["infra_only_critical_evidence_mentioned"])
                    for row in infra_available
                ),
                len(infra_available),
            )
            if infra_available
            else "not_available"
        ),
        "critical_evidence_infra_reference_rate": rate(
            sum(row["critical_evidence_infra_reference"] for row in records),
            sample_count,
        ),
        "grounding_rule": (
            "Exact object-ID mention within the parsed Critical evidence section; "
            "infra reference uses a transparent keyword/ID rule."
        ),
    }
    if output_jsonl is not None:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl.open("w", encoding="utf-8") as file:
            for record in records:
                out = dict(record)
                out.pop("raw_generated_cot", None)
                file.write(json.dumps(out, ensure_ascii=False) + "\n")
    return {
        "run_metadata": {
            "stage": "three_part_cot_diagnostics_postprocess",
            "created_at_utc": utc_now(),
            "metrics_path": str(metrics_path),
            "samples_path": str(samples_path),
            "index_path": str(index_path),
            "language_contract": THREE_PART_CONTRACT,
            "gpu_inference_started": False,
        },
        "metrics": metrics,
        "qualitative_examples": choose_three_part_examples(records),
    }


def build_diagnostics(
    metrics_path: Path,
    samples_path: Path,
    index_path: Path,
    output_jsonl: Optional[Path],
    contract: str = "legacy_four_part",
) -> Dict[str, Any]:
    if contract == "three_part":
        return build_three_part_diagnostics(
            metrics_path, samples_path, index_path, output_jsonl
        )
    if contract != "legacy_four_part":
        raise ValueError(f"Unknown language contract: {contract}")
    return build_legacy_diagnostics(
        metrics_path, samples_path, index_path, output_jsonl
    )


def fmt_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_legacy_report(path: Path, diagnostics: Dict[str, Any]) -> None:
    m = diagnostics["metrics"]
    examples = diagnostics["qualitative_examples"]
    lines = [
        "# CoT Diagnostics for Structured-Head Benchmark",
        "",
        "Scope: sparse 1024-token action-first examples from the accepted original-val run.",
        "",
        "Policy note: structured Part 4 is action-consistent output from Qwen-conditioned command/waypoint heads. It is not relabeled as raw generated CoT; raw generated CoT is retained separately for audit.",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    metric_order = [
        "sample_count",
        "raw_generated_cot_format_parse_rate",
        "raw_part4_command_parse_rate",
        "raw_part4_command_vs_structured_agreement_rate",
        "raw_part4_command_vs_gt_agreement_rate",
        "structured_command_vs_gt_accuracy_on_diagnostic_samples",
        "raw_part4_waypoint_parse_rate",
        "raw_waypoint_vs_structured_ADE",
        "raw_waypoint_vs_structured_FDE",
        "critical_object_mention_rate",
        "infra_only_critical_object_mention_rate",
        "infra_reference_rate",
    ]
    for key in metric_order:
        lines.append(f"| `{key}` | {fmt_metric(m.get(key))} |")
    lines.extend(
        [
            "",
            "Grounding note: critical-object mention metrics are computed by joining sparse samples to the index metadata. Infra reference uses a transparent keyword/ID rule, not a semantic judge.",
            "",
            "## Qualitative Examples",
            "",
        ]
    )
    for ex in examples:
        lines.extend(
            [
                f"### {ex['case']}: `{ex['sample_id']}`",
                "",
                f"- GT command: `{ex['gt_command']}`; structured command: `{ex['structured_command']}`; raw Part 4 command: `{ex['raw_command']}`.",
                f"- Critical IDs: `{ex['critical_object_ids']}`; infra-only critical IDs: `{ex['infra_only_critical_ids']}`.",
                f"- Excerpt: {ex['excerpt']}",
                "",
            ]
        )
    missing_cases = diagnostics.get("missing_qualitative_example_cases", [])
    if missing_cases:
        lines.extend(
            [
                "Missing requested example categories in this sparse sample:",
                "",
            ]
        )
        for case in missing_cases:
            lines.append(f"- {case}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_three_part_report(path: Path, diagnostics: Dict[str, Any]) -> None:
    metrics = diagnostics["metrics"]
    contract = metrics["language_contract"]
    rows = [
        ("sample_count", metrics["sample_count"]),
        ("exact_three_part_parse_rate", contract["exact_three_part_parse_rate"]),
        (
            "forbidden_text_action_field_rate",
            contract["forbidden_text_action_field_rate"],
        ),
        (
            "critical_evidence_object_mention_rate",
            metrics["critical_evidence_object_mention_rate"],
        ),
        (
            "decision_reasoning_object_mention_rate",
            metrics["decision_reasoning_object_mention_rate"],
        ),
        (
            "infra_only_critical_evidence_mention_rate",
            metrics["infra_only_critical_evidence_mention_rate"],
        ),
        (
            "critical_evidence_infra_reference_rate",
            metrics["critical_evidence_infra_reference_rate"],
        ),
    ]
    lines = [
        "# Three-Part Language Diagnostics",
        "",
        (
            "Scope: generated Scene understanding, Critical evidence, and Decision "
            "reasoning. Commands and waypoints remain outputs of separate structured "
            "heads; no language-action agreement is computed."
        ),
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    lines.extend(f"| `{name}` | {fmt_metric(value)} |" for name, value in rows)
    lines.extend(["", "## Qualitative Examples", ""])
    for example in diagnostics["qualitative_examples"]:
        lines.extend(
            [
                f"### {example['case']}: `{example['sample_id']}`",
                "",
                f"- Format valid: `{example['format_parse_ok']}`; errors: `{example['parse_errors']}`.",
                (
                    f"- Critical IDs: `{example['critical_object_ids']}`; infra-only "
                    f"critical IDs: `{example['infra_only_critical_ids']}`."
                ),
                f"- Excerpt: {example['excerpt']}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_report(
    path: Path,
    diagnostics: Dict[str, Any],
    contract: str = "legacy_four_part",
) -> None:
    if contract == "three_part":
        write_three_part_report(path, diagnostics)
        return
    if contract != "legacy_four_part":
        raise ValueError(f"Unknown language contract: {contract}")
    write_legacy_report(path, diagnostics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CoT diagnostics from action-first eval artifacts.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--contract",
        choices=("legacy_four_part", "three_part"),
        default="legacy_four_part",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnostics = build_diagnostics(
        args.metrics,
        args.samples,
        args.index,
        args.output_jsonl,
        contract=args.contract,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(args.output_md, diagnostics, contract=args.contract)
    print(
        "[DONE] CoT diagnostics saved to "
        f"{args.output_json}; report saved to {args.output_md}; "
        f"sample_count={diagnostics['metrics']['sample_count']}"
    )


if __name__ == "__main__":
    main()
