#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch


COMMANDS = ["GO_STRAIGHT", "TURN_LEFT", "TURN_RIGHT", "LATERAL_SHIFT", "STOP", "SLOW_DOWN", "UNKNOWN"]
ID2COMMAND = {idx: name for idx, name in enumerate(COMMANDS)}
HORIZONS = ["0.5", "1.0", "1.5", "2.0", "2.5", "3.0"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tensor_cpu(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().cpu().float()
    if isinstance(value, list) and value and torch.is_tensor(value[0]):
        return torch.stack([v.detach().cpu().float().reshape(-1)[0] for v in value])
    return torch.tensor(value, dtype=torch.float32)


def univ2x_to_covlm_xy(points: Any) -> np.ndarray:
    pts = tensor_cpu(points).numpy().astype(np.float64)
    if pts.ndim == 3 and pts.shape[0] == 1:
        pts = pts[0]
    pts = pts[:6, :2]
    # UniV2X [forward, lateral] -> CoVLM [x_lateral, y_forward] = [-lateral, forward].
    return np.stack([-pts[:, 1], pts[:, 0]], axis=-1)


def command_from_item(item: Dict[str, Any]) -> tuple[str, Optional[List[float]]]:
    logits = item.get("command_logits")
    if logits is not None:
        logits_tensor = tensor_cpu(logits)
        if logits_tensor.ndim > 1:
            logits_tensor = logits_tensor.reshape(-1, logits_tensor.shape[-1])[0]
        pred_id = int(torch.argmax(logits_tensor).item())
        return ID2COMMAND.get(pred_id, "UNKNOWN"), logits_tensor.tolist()
    pred = item.get("pred_command")
    if pred is not None:
        pred_tensor = tensor_cpu(pred).reshape(-1)
        return ID2COMMAND.get(int(pred_tensor[0].item()), "UNKNOWN"), None
    return "UNKNOWN", None


def load_results(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "bbox_results" in payload:
        rows = payload["bbox_results"]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise RuntimeError(f"Unsupported UniMM/UniV2X result pkl format: {type(payload)}")
    out = {}
    for item in rows:
        token = str(item["token"])
        out[token] = item
    return out


def checkpoint_profile(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {"checkpoint_path": None, "checkpoint_size_bytes": None, "checkpoint_tensor_params": None}
    exists = path.exists()
    size = path.stat().st_size if exists else None
    tensor_params = None
    if exists:
        payload = torch.load(str(path), map_location="cpu")
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else {}
        tensor_params = int(sum(v.numel() for v in state.values() if torch.is_tensor(v)))
    return {
        "checkpoint_path": str(path),
        "checkpoint_exists": exists,
        "checkpoint_size_bytes": size,
        "checkpoint_tensor_params": tensor_params,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export UniMM-V2X results under CoVLM-Bench metrics.")
    parser.add_argument("--results-pkl", type=Path, required=True)
    parser.add_argument("--covlm-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--condition", default="normal")
    parser.add_argument("--elapsed-sec", type=float)
    parser.add_argument("--cuda-peak-memory-mib", type=float)
    parser.add_argument("--cpu-rss-mib", type=float)
    args = parser.parse_args()

    covlm_rows = [row for row in read_jsonl(args.covlm_index) if row.get("split") == "val"]
    results = load_results(args.results_pkl)
    if len(covlm_rows) != 654:
        raise RuntimeError(f"Expected 654 CoVLM val rows, got {len(covlm_rows)}")

    structured = []
    dists = []
    cmd_correct = 0
    per_cmd = defaultdict(lambda: {"count": 0, "correct": 0})
    missing = []
    for row in covlm_rows:
        token = str(row["token"])
        if token not in results:
            missing.append(token)
            continue
        item = results[token]
        pred_wp = univ2x_to_covlm_xy(item["planning_traj"])
        gt_wp = np.asarray(row["waypoints"], dtype=np.float64)[:6, :2]
        step_dist = np.sqrt(((pred_wp - gt_wp) ** 2).sum(axis=-1))
        pred_command, logits = command_from_item(item)
        gt_command = str(row["command"])
        cmd_correct += int(pred_command == gt_command)
        per_cmd[gt_command]["count"] += 1
        per_cmd[gt_command]["correct"] += int(pred_command == gt_command)
        dists.append(step_dist)
        payload = {
            "sample_id": row.get("sample_id"),
            "token": token,
            "condition": args.condition,
            "pred_waypoints": pred_wp.tolist(),
            "gt_waypoints": gt_wp.tolist(),
            "pred_command": pred_command,
            "gt_command": gt_command,
        }
        if logits is not None:
            payload["command_logits"] = logits
        structured.append(payload)

    if missing:
        raise RuntimeError(f"Missing {len(missing)} CoVLM val tokens in result pkl, first={missing[:10]}")
    if len(structured) != 654:
        raise RuntimeError(f"Expected 654 structured rows, got {len(structured)}")

    dist_arr = np.stack(dists, axis=0)
    present_acc = [
        rec["correct"] / max(rec["count"], 1)
        for rec in per_cmd.values()
        if rec["count"] > 0
    ]
    all_acc = [
        per_cmd[cmd]["correct"] / per_cmd[cmd]["count"] if per_cmd[cmd]["count"] else 0.0
        for cmd in COMMANDS
    ]
    metrics = {
        "created_at_utc": utc_now(),
        "method": "UniMM-V2X CoVLM-Bench protocol port/adaptation",
        "condition": args.condition,
        "count": len(structured),
        "waypoint_ADE_each_step": [float(x) for x in dist_arr.mean(axis=0).tolist()],
        "final_displacement_error": float(dist_arr[:, -1].mean()),
        "ADE": float(dist_arr.mean()),
        "FDE": float(dist_arr[:, -1].mean()),
        "command_accuracy": cmd_correct / max(len(structured), 1),
        "macro_command_accuracy_present_classes": float(np.mean(present_acc)) if present_acc else 0.0,
        "macro_command_accuracy_all_classes": float(np.mean(all_acc)) if all_acc else 0.0,
        "pred_command_distribution": dict(Counter(row["pred_command"] for row in structured)),
        "gt_command_distribution": dict(Counter(row["gt_command"] for row in structured)),
        "command_labels": COMMANDS,
        "normal_only_causal_v2x_gain_claim": False,
    }

    out = args.output_dir
    structured_path = out / "samples" / f"unimmv2x_covlm_{args.condition}_structured_predictions.jsonl"
    metrics_path = out / "metrics" / f"unimmv2x_covlm_{args.condition}_metrics.json"
    profile_path = out / "metrics" / f"unimmv2x_covlm_{args.condition}_profile.json"
    report_path = out / "reports" / f"unimmv2x_covlm_{args.condition}_summary.md"
    write_jsonl(structured_path, structured)
    write_json(metrics_path, metrics)
    throughput = len(structured) / args.elapsed_sec if args.elapsed_sec and args.elapsed_sec > 0 else None
    profile = {
        "created_at_utc": utc_now(),
        "profile_eval_version": 2,
        "condition": args.condition,
        "count": len(structured),
        "elapsed_sec": args.elapsed_sec,
        "throughput_samples_per_sec": throughput,
        "fps": throughput,
        "latency_ms_per_sample": (1000.0 / throughput) if throughput else None,
        "cuda_peak_memory_mib": args.cuda_peak_memory_mib,
        "cpu_rss_mib": args.cpu_rss_mib,
        "flops": None,
        "flops_note": "not measured; not fabricated",
        **checkpoint_profile(args.checkpoint),
    }
    write_json(profile_path, profile)

    ade = metrics["waypoint_ADE_each_step"]
    table = [
        "| Method | Cond | N | ADE@0.5 | ADE@1.0 | ADE@1.5 | ADE@2.0 | ADE@2.5 | ADE@3.0 | FDE@3.0 | Cmd Acc | Macro Cmd | Lat ms | FPS | Peak MiB | Params | Ckpt bytes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| UniMM-V2X port | {args.condition} | {metrics['count']} | "
            f"{ade[0]:.6f} | {ade[1]:.6f} | {ade[2]:.6f} | {ade[3]:.6f} | {ade[4]:.6f} | {ade[5]:.6f} | "
            f"{metrics['final_displacement_error']:.6f} | {metrics['command_accuracy']:.6f} | "
            f"{metrics['macro_command_accuracy_present_classes']:.6f} | "
            f"{profile['latency_ms_per_sample'] if profile['latency_ms_per_sample'] is not None else 'null'} | "
            f"{profile['fps'] if profile['fps'] is not None else 'null'} | "
            f"{profile['cuda_peak_memory_mib'] if profile['cuda_peak_memory_mib'] is not None else 'null'} | "
            f"{profile['checkpoint_tensor_params']} | {profile['checkpoint_size_bytes']} |"
        ),
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(table) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": str(metrics_path), "structured": str(structured_path), "profile": str(profile_path), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
