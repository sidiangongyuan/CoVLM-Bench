#!/usr/bin/env python3
"""Build CoVLM-protocol planning comparison tables from structured exports."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


COMMANDS = ["GO_STRAIGHT", "TURN_LEFT", "TURN_RIGHT", "LATERAL_SHIFT", "STOP", "SLOW_DOWN", "UNKNOWN"]
HORIZONS = ["0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s"]


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_val_index(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = {}
    for row in read_jsonl(path):
        if row.get("split") == "val":
            rows[str(row["token"])] = row
    return rows


def l2_steps(pred: Sequence[Sequence[float]], gt: Sequence[Sequence[float]]) -> List[float]:
    return [
        math.hypot(float(p[0]) - float(g[0]), float(p[1]) - float(g[1]))
        for p, g in zip(pred, gt)
    ]


def mean(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    if not vals:
        return None
    return sum(vals) / len(vals)


def metric_summary(rows: List[Dict[str, Any]], has_command: bool) -> Dict[str, Any]:
    count = len(rows)
    if count == 0:
        return {
            "count": 0,
            "l2": {h: None for h in HORIZONS},
            "fde": None,
            "command_accuracy": None,
            "macro_command_accuracy_present_classes": None,
            "macro_command_accuracy_all_classes": None,
        }
    step_vals: List[List[float]] = [[] for _ in HORIZONS]
    command_correct = 0
    by_command: Dict[str, List[int]] = defaultdict(list)
    for row in rows:
        distances = l2_steps(row["pred_waypoints"], row["gt_waypoints"])
        for idx, value in enumerate(distances[: len(HORIZONS)]):
            step_vals[idx].append(value)
        if has_command:
            gt_command = str(row.get("gt_command", "UNKNOWN"))
            correct = int(str(row.get("pred_command", "UNKNOWN")) == gt_command)
            command_correct += correct
            by_command[gt_command].append(correct)
    summary: Dict[str, Any] = {
        "count": count,
        "l2": {h: mean(vals) for h, vals in zip(HORIZONS, step_vals)},
        "fde": mean(step_vals[-1]),
        "command_accuracy": None,
        "macro_command_accuracy_present_classes": None,
        "macro_command_accuracy_all_classes": None,
    }
    if has_command:
        present = [mean(vals) for vals in by_command.values() if vals]
        all_classes = [mean(by_command[cmd]) if by_command.get(cmd) else 0.0 for cmd in COMMANDS]
        summary.update(
            {
                "command_accuracy": command_correct / max(count, 1),
                "macro_command_accuracy_present_classes": mean(v for v in present if v is not None),
                "macro_command_accuracy_all_classes": mean(all_classes),
                "gt_command_distribution": dict(Counter(str(row.get("gt_command", "UNKNOWN")) for row in rows)),
                "pred_command_distribution": dict(Counter(str(row.get("pred_command", "UNKNOWN")) for row in rows)),
            }
        )
    return summary


def load_structured_export(path: Path, val_tokens: set[str]) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    seen = {str(row["token"]) for row in rows}
    if len(rows) != 654:
        raise ValueError(f"{path} expected 654 rows, got {len(rows)}")
    if seen != val_tokens:
        missing = sorted(val_tokens - seen)[:10]
        extra = sorted(seen - val_tokens)[:10]
        raise ValueError(f"{path} token mismatch; missing={missing}, extra={extra}")
    return rows


def load_univ2x_bridge(path: Path, val_tokens: set[str]) -> List[Dict[str, Any]]:
    rows = []
    for row in read_jsonl(path):
        rows.append(
            {
                "token": str(row["token"]),
                "sample_id": row.get("sample_id"),
                "pred_waypoints": row["univ2x_pred_as_covlm_6"],
                "gt_waypoints": row["covlm_gt_covlm_6"],
                "pred_command": None,
                "gt_command": None,
            }
        )
    seen = {row["token"] for row in rows}
    if len(rows) != 654 or seen != val_tokens:
        raise ValueError(f"{path} bridge rows do not match 654 CoVLM val tokens")
    return rows


def subset_rows(rows: List[Dict[str, Any]], tokens: set[str]) -> List[Dict[str, Any]]:
    return [row for row in rows if str(row["token"]) in tokens]


def fmt_float(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def fmt_acc(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{100.0 * value:.1f}"


def row_to_markdown(label: str, summary: Dict[str, Any], note: str) -> str:
    l2 = summary["l2"]
    return (
        f"| {label} | {summary['count']} | "
        + " | ".join(fmt_float(l2[h]) for h in HORIZONS)
        + f" | {fmt_float(summary['fde'])} | {fmt_acc(summary['command_accuracy'])} | "
        + f"{fmt_acc(summary['macro_command_accuracy_present_classes'])} | {note} |"
    )


def write_main_table(path: Path, rows: List[Dict[str, Any]], appendix_rows: List[Dict[str, Any]]) -> None:
    header = (
        "| Method | N | L2@0.5 | L2@1.0 | L2@1.5 | L2@2.0 | L2@2.5 | L2@3.0/FDE | "
        "FDE | Cmd Acc (%) | Macro Cmd Acc (%) | Note |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    )
    lines = [
        "# Main CoVLM-Protocol Planning Table",
        "",
        "All rows use the corrected six-step CoVLM GT. UniV2X is reported only through the accepted bridge under CoVLM protocol.",
        "",
        header,
    ]
    lines.extend(row_to_markdown(row["label"], row["summary"], row["note"]) for row in rows)
    if appendix_rows:
        lines.extend(["", "## Appendix/Internal", "", header])
        lines.extend(row_to_markdown(row["label"], row["summary"], row["note"]) for row in appendix_rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_subset_table(path: Path, title: str, rows: List[Dict[str, Any]], appendix_rows: List[Dict[str, Any]]) -> None:
    header = (
        "| Method | N | L2@0.5 | L2@1.0 | L2@1.5 | L2@2.0 | L2@2.5 | L2@3.0/FDE | "
        "FDE | Cmd Acc (%) | Macro Cmd Acc (%) | Note |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    )
    lines = [f"# {title}", "", header]
    lines.extend(row_to_markdown(row["label"], row["summary"], row["note"]) for row in rows)
    if appendix_rows:
        lines.extend(["", "## Appendix/Internal", "", header])
        lines.extend(row_to_markdown(row["label"], row["summary"], row["note"]) for row in appendix_rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_maneuver_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    header = (
        "| GT Command | Method | N | L2@3.0/FDE | Cmd Acc (%) | Note |\n"
        "|---|---|---:|---:|---:|---|"
    )
    lines = ["# Maneuver Subset Table", "", header]
    for command in COMMANDS:
        command_rows = [row for row in rows if row["command"] == command and row["summary"]["count"] > 0]
        if not command_rows:
            continue
        for row in command_rows:
            lines.append(
                f"| {command} | {row['label']} | {row['summary']['count']} | "
                f"{fmt_float(row['summary']['fde'])} | {fmt_acc(row['summary']['command_accuracy'])} | {row['note']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--bridge-metrics", type=Path, required=True)
    parser.add_argument("--bridge-per-sample", type=Path, required=True)
    parser.add_argument("--ego-export", type=Path, required=True)
    parser.add_argument("--dual-export", type=Path, required=True)
    parser.add_argument("--blank-export", type=Path, required=True)
    parser.add_argument("--shuffled-export", type=Path, required=True)
    parser.add_argument("--cot-export", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir
    for sub in ["metrics", "reports", "tables"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    val_index = load_val_index(args.index_path)
    val_tokens = set(val_index)
    if len(val_tokens) != 654:
        raise ValueError(f"Expected 654 val tokens, got {len(val_tokens)}")
    critical_tokens = {token for token, row in val_index.items() if row.get("infra_only_critical_ids")}
    command_tokens = {cmd: {token for token, row in val_index.items() if row.get("command") == cmd} for cmd in COMMANDS}

    bridge_rows = load_univ2x_bridge(args.bridge_per_sample, val_tokens)
    bridge_metrics = read_json(args.bridge_metrics)
    univ2x_main_l2 = bridge_metrics["overlap654_covlm_protocol"]["univ2x_l2"]
    univ2x_main_summary = {
        "count": 654,
        "l2": univ2x_main_l2,
        "fde": bridge_metrics["overlap654_covlm_protocol"]["univ2x_fde"],
        "command_accuracy": None,
        "macro_command_accuracy_present_classes": None,
        "macro_command_accuracy_all_classes": None,
    }

    row_specs = [
        {
            "key": "univ2x_bridge",
            "label": "UniV2X (checkpoint, bridge under CoVLM protocol)",
            "rows": bridge_rows,
            "has_command": False,
            "note": "bridge under CoVLM protocol; no command output",
            "main_summary_override": univ2x_main_summary,
            "main": True,
        },
        {
            "key": "ego_no_cot",
            "label": "Qwen3-VL-8B Ego-only, no-CoT-loss",
            "rows": load_structured_export(args.ego_export, val_tokens),
            "has_command": True,
            "note": "structured head; no CoT loss",
            "main": True,
        },
        {
            "key": "v2x_no_cot",
            "label": "Qwen3-VL-8B V2X-VLM, no-CoT-loss",
            "rows": load_structured_export(args.dual_export, val_tokens),
            "has_command": True,
            "note": "structured head; no CoT loss",
            "main": True,
        },
        {
            "key": "v2x_blank_infra",
            "label": "Qwen3-VL-8B V2X-VLM, blank infra",
            "rows": load_structured_export(args.blank_export, val_tokens),
            "has_command": True,
            "note": "input ablation",
            "main": True,
        },
        {
            "key": "v2x_shuffled_infra",
            "label": "Qwen3-VL-8B V2X-VLM, shuffled infra",
            "rows": load_structured_export(args.shuffled_export, val_tokens),
            "has_command": True,
            "note": "input ablation",
            "main": True,
        },
        {
            "key": "v2x_cot_supervised",
            "label": "Qwen3-VL-8B V2X-VLM, CoT-supervised",
            "rows": load_structured_export(args.cot_export, val_tokens),
            "has_command": True,
            "note": "appendix/internal; auxiliary LM supervision",
            "main": False,
        },
    ]

    main_rows: List[Dict[str, Any]] = []
    appendix_rows: List[Dict[str, Any]] = []
    critical_rows: List[Dict[str, Any]] = []
    critical_appendix_rows: List[Dict[str, Any]] = []
    maneuver_rows: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {
        "protocol": "CoVLM corrected 6-step GT",
        "validation": {
            "val_count": len(val_tokens),
            "v2x_critical_definition": "non-empty infra_only_critical_ids in the shared 654-row val index",
            "v2x_critical_count": len(critical_tokens),
            "gt_command_distribution": dict(Counter(row.get("command", "UNKNOWN") for row in val_index.values())),
            "univ2x_label": "bridge under CoVLM protocol",
            "cot_supervised_scope": "appendix/internal only",
        },
        "rows": {},
    }

    for spec in row_specs:
        all_summary = spec.get("main_summary_override") or metric_summary(spec["rows"], spec["has_command"])
        critical_summary = metric_summary(subset_rows(spec["rows"], critical_tokens), spec["has_command"])
        row_payload = {
            "label": spec["label"],
            "note": spec["note"],
            "main": bool(spec["main"]),
            "all_val": all_summary,
            "v2x_critical": critical_summary,
            "maneuver_subsets": {},
        }
        table_row = {"label": spec["label"], "summary": all_summary, "note": spec["note"]}
        critical_row = {"label": spec["label"], "summary": critical_summary, "note": spec["note"]}
        if spec["main"]:
            main_rows.append(table_row)
            critical_rows.append(critical_row)
        else:
            appendix_rows.append(table_row)
            critical_appendix_rows.append(critical_row)
        for command, tokens in command_tokens.items():
            command_summary = metric_summary(subset_rows(spec["rows"], tokens), spec["has_command"])
            row_payload["maneuver_subsets"][command] = command_summary
            maneuver_rows.append(
                {
                    "command": command,
                    "label": spec["label"],
                    "summary": command_summary,
                    "note": spec["note"] if spec["main"] else "appendix/internal",
                }
            )
        metrics["rows"][spec["key"]] = row_payload

    metrics_path = out / "metrics" / "unified_planning_comparison.json"
    write_json(metrics_path, metrics)
    write_main_table(out / "tables" / "main_covlm_protocol_table.md", main_rows, appendix_rows)
    write_subset_table(
        out / "tables" / "v2x_critical_subset_table.md",
        "V2X-Critical Subset Table",
        critical_rows,
        critical_appendix_rows,
    )
    write_maneuver_table(out / "tables" / "maneuver_subset_table.md", maneuver_rows)

    report_lines = [
        "# Unified CoVLM Planning Comparison",
        "",
        f"- Protocol: {metrics['protocol']}",
        f"- Validation rows: {len(val_tokens)}",
        f"- V2X-critical subset: {len(critical_tokens)}",
        "- UniV2X row is bridge-only under CoVLM protocol and has no command accuracy.",
        "- CoT-supervised V2X row is appendix/internal only.",
        "",
        "## Artifacts",
        "",
        f"- Metrics: `{metrics_path}`",
        "- Main table: `tables/main_covlm_protocol_table.md`",
        "- V2X-critical table: `tables/v2x_critical_subset_table.md`",
        "- Maneuver table: `tables/maneuver_subset_table.md`",
    ]
    (out / "reports" / "unified_planning_comparison.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"metrics": str(metrics_path), "rows": list(metrics["rows"].keys())}, indent=2))


if __name__ == "__main__":
    main()
