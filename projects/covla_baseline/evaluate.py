#!/usr/bin/env python3
"""Evaluate command and waypoint heads for the CoVLM Qwen-VL baseline."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from torch.utils.data import DataLoader

from projects.covla_baseline.data.collator import CoVLACollator
from projects.covla_baseline.data.dataset import (
    COMMANDS,
    DEFAULT_PART1_OBJECT_LIMIT,
    DEFAULT_PART1_SENTENCE_LIMIT,
    ID2COMMAND,
    CoVLABaselineDataset,
    canonical_command,
)
from projects.covla_baseline.data.evidence import FEATURE_DIM as L1_EVIDENCE_FEATURE_DIM
from projects.covla_baseline.data.evidence import ROUTE_FEATURE_DIM as ROUTE_EVIDENCE_FEATURE_DIM
from projects.covla_baseline.models.qwen_vla import QwenVLABaseline
from projects.covla_baseline.three_part_cot import (
    CONTRACT_NAME as THREE_PART_CONTRACT,
    aggregate_three_part_parses,
    parse_three_part_cot,
)
from projects.covla_baseline.train import (
    cfg_int,
    import_processor,
    load_config,
    to_device,
    validate_causal_anchor_config,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def profile_file_artifact(path: Path) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def profile_checkpoint_provenance(checkpoint_dir: Path) -> Dict[str, Any]:
    root = checkpoint_dir.expanduser().resolve()
    candidates = (
        "action_heads.pt",
        "baseline_config.json",
        "adapter_config.json",
        "adapter_model.safetensors",
        "model.safetensors",
        "backbone/adapter_config.json",
        "backbone/adapter_model.safetensors",
        "backbone/model.safetensors",
    )
    files = []
    for relative in candidates:
        path = root / relative
        if path.is_file():
            files.append({"relative_path": relative, **profile_file_artifact(path)})
            files[-1].pop("path")
    if not files:
        raise FileNotFoundError(f"No runtime checkpoint payload files found in {root}")
    return {"resolved_path": str(root), "files": files}


def percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = min(len(ordered) - 1, max(0.0, (pct / 100.0) * (len(ordered) - 1)))
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def value_distribution(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"status": "empty", "count": 0}
    vals = [float(v) for v in values]
    return {
        "status": "ok",
        "count": len(vals),
        "mean": statistics.fmean(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "median": statistics.median(vals),
        "p50": percentile(vals, 50),
        "p90": percentile(vals, 90),
        "p95": percentile(vals, 95),
        "p99": percentile(vals, 99),
        "min": min(vals),
        "max": max(vals),
    }


def bytes_to_gib(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return float(value) / 1024**3


def cuda_device_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "selected_index": None,
        "selected_name": None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        info.update(
            {
                "selected_index": int(idx),
                "selected_name": torch.cuda.get_device_name(idx),
                "cuda_runtime_version": torch.version.cuda,
            }
        )
    return info


def query_nvidia_smi() -> Dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc)}
    rows = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        rows.append(
            {
                "index": parts[0],
                "name": parts[1],
                "memory_used_mib": parts[2],
                "memory_total_mib": parts[3],
                "utilization_gpu_percent": parts[4],
                "power_draw_w": parts[5],
            }
        )
    return {"status": "ok", "rows": rows}


def cpu_rss_bytes() -> Tuple[Optional[int], str]:
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss), "psutil"
    except Exception:
        try:
            import resource

            rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            return rss_kib * 1024, "resource_ru_maxrss"
        except Exception:
            return None, "unavailable"


def checkpoint_size_summary(checkpoint_dir: str) -> Dict[str, Any]:
    root = Path(checkpoint_dir)
    total = 0
    files: Dict[str, Optional[int]] = {
        "backbone/adapter_model.safetensors": None,
        "backbone/adapter_config.json": None,
        "adapter_model.safetensors": None,
        "adapter_config.json": None,
        "action_heads.pt": None,
        "baseline_config.json": None,
    }
    config_files: List[str] = []
    if root.exists():
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            size = path.stat().st_size
            total += size
            rel = str(path.relative_to(root))
            if rel in files:
                files[rel] = size
            if path.suffix.lower() in {".json", ".yaml", ".yml"}:
                config_files.append(rel)
    adapter_bytes = files.get("backbone/adapter_model.safetensors")
    if adapter_bytes is None:
        adapter_bytes = files.get("adapter_model.safetensors")
    return {
        "checkpoint_dir": str(root),
        "total_bytes": total if root.exists() else None,
        "total_gib": bytes_to_gib(total) if root.exists() else None,
        "files": files,
        "adapter_checkpoint_bytes": adapter_bytes,
        "action_heads_checkpoint_bytes": files.get("action_heads.pt"),
        "config_files": sorted(config_files),
        "status": "ok" if root.exists() else "missing",
    }


def parameter_group_summary(model: QwenVLABaseline) -> Dict[str, Any]:
    groups = {
        "backbone": 0,
        "lora_adapter_named": 0,
        "action_head": 0,
        "command_head": 0,
        "residual_gate_head": 0,
        "geov2x_residual_head": 0,
        "predicted_evidence_residual_head": 0,
        "predicted_evidence_token_residual_head": 0,
        "evidence_residual_head": 0,
        "evidence_directed_residual_head": 0,
        "task_query_pooler": 0,
        "other": 0,
    }
    trainable_groups = {name: 0 for name in groups}
    total = 0
    trainable = 0
    for name, param in model.named_parameters():
        n = int(param.numel())
        total += n
        if param.requires_grad:
            trainable += n
        if name.startswith("backbone."):
            group = "lora_adapter_named" if "lora_" in name.lower() else "backbone"
        elif name.startswith("action_head."):
            group = "action_head"
        elif name.startswith("command_head."):
            group = "command_head"
        elif name.startswith("residual_gate_head."):
            group = "residual_gate_head"
        elif name.startswith("geov2x_residual_head."):
            group = "geov2x_residual_head"
        elif name.startswith("predicted_evidence_residual_head."):
            group = "predicted_evidence_residual_head"
        elif name.startswith("predicted_evidence_token_residual_head."):
            group = "predicted_evidence_token_residual_head"
        elif name.startswith("evidence_residual_head."):
            group = "evidence_residual_head"
        elif name.startswith("evidence_directed_residual_head."):
            group = "evidence_directed_residual_head"
        elif name.startswith("task_query_pooler."):
            group = "task_query_pooler"
        else:
            group = "other"
        groups[group] += n
        if param.requires_grad:
            trainable_groups[group] += n
    head_and_residual_groups = [
        "action_head",
        "command_head",
        "residual_gate_head",
        "geov2x_residual_head",
        "predicted_evidence_residual_head",
        "predicted_evidence_token_residual_head",
        "evidence_residual_head",
        "evidence_directed_residual_head",
        "task_query_pooler",
    ]
    non_backbone_groups = head_and_residual_groups + ["other"]
    return {
        "status": "ok",
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "trainable_ratio": trainable / max(total, 1),
        "groups": groups,
        "trainable_groups": trainable_groups,
        "trainable_non_backbone_parameters": sum(trainable_groups[name] for name in non_backbone_groups),
        "head_and_residual_parameters": sum(groups[name] for name in head_and_residual_groups),
        "lora_parameter_group_caveat": (
            "LoRA adapter parameters are identified conservatively by parameter names containing "
            "'lora_' under backbone.*. During eval, PEFT adapters may be loaded with "
            "is_trainable=false, so trainable backbone/LoRA counts can be zero even when LoRA "
            "was trained."
        ),
        "caveat": "Grouped by module name prefixes; LoRA grouping uses lora_ name matching.",
    }


def tensor_shape(value: Any) -> Optional[List[int]]:
    if torch.is_tensor(value):
        return [int(x) for x in value.shape]
    return None


def collect_input_stats(batch: Dict[str, Any], samples: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    stats: Dict[str, List[float]] = defaultdict(list)
    input_ids = batch.get("input_ids")
    attention_mask = batch.get("attention_mask")
    if torch.is_tensor(input_ids):
        pad_lengths = [int(input_ids.shape[1])] * int(input_ids.shape[0])
        stats["input_ids_length"].extend(float(x) for x in pad_lengths)
    if torch.is_tensor(attention_mask):
        stats["attention_tokens"].extend(float(x) for x in attention_mask.detach().cpu().sum(dim=1).tolist())
    pixel_values = batch.get("pixel_values")
    if torch.is_tensor(pixel_values):
        per_sample = float(pixel_values.numel()) / max(len(samples), 1)
        stats["pixel_values_numel"].extend([per_sample] * len(samples))
    image_grid = batch.get("image_grid_thw")
    if torch.is_tensor(image_grid):
        grid = image_grid.detach().cpu().view(-1, image_grid.shape[-1])
        if grid.numel():
            products = grid.prod(dim=1).float().tolist()
            per_sample = len(products) / max(len(samples), 1)
            stats["image_grid_count_per_sample"].extend([float(per_sample)] * len(samples))
            stats["image_grid_tokens"].extend(float(x) for x in products)
    evidence_mask = batch.get("object_evidence_mask")
    if torch.is_tensor(evidence_mask):
        stats["object_evidence_mask_count"].extend(
            float(x) for x in evidence_mask.detach().cpu().sum(dim=1).tolist()
        )
    for sample in samples:
        prompt = str(sample.get("prompt", ""))
        stats["prompt_chars"].append(float(len(prompt)))
        stats["prompt_words"].append(float(len(prompt.split())))
        image_paths = sample.get("image_paths") or []
        stats["image_inputs_per_sample"].append(float(len(image_paths)))
    return stats


def merge_stat_lists(target: Dict[str, List[float]], source: Dict[str, List[float]]) -> None:
    for key, values in source.items():
        target.setdefault(key, []).extend(values)


def command_calibration_stats(logits: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> Dict[str, Any]:
    if logits.numel() == 0:
        return {"status": "empty"}
    probs = torch.softmax(logits.float(), dim=-1)
    conf, pred = probs.max(dim=-1)
    correct = pred.eq(labels)
    top2 = probs.topk(k=min(2, probs.shape[-1]), dim=-1).values
    margin = top2[:, 0] - (top2[:, 1] if top2.shape[-1] > 1 else 0.0)
    entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
    nll = torch.nn.functional.cross_entropy(logits.float(), labels.long(), reduction="mean")
    one_hot = torch.nn.functional.one_hot(labels.long(), num_classes=probs.shape[-1]).float()
    brier = torch.mean(torch.sum((probs - one_hot).pow(2), dim=-1))
    table = []
    ece = 0.0
    conf_cpu = conf.detach().cpu()
    correct_cpu = correct.detach().float().cpu()
    for idx in range(bins):
        lo = idx / bins
        hi = (idx + 1) / bins
        if idx == bins - 1:
            mask = (conf_cpu >= lo) & (conf_cpu <= hi)
        else:
            mask = (conf_cpu >= lo) & (conf_cpu < hi)
        count = int(mask.sum())
        if count:
            avg_conf = float(conf_cpu[mask].mean())
            acc = float(correct_cpu[mask].mean())
            ece += (count / max(len(conf_cpu), 1)) * abs(avg_conf - acc)
        else:
            avg_conf = None
            acc = None
        table.append(
            {
                "bin_start": lo,
                "bin_end": hi,
                "count": count,
                "avg_confidence": avg_conf,
                "accuracy": acc,
            }
        )
    correct_conf = conf[correct]
    wrong_conf = conf[~correct]
    return {
        "status": "ok",
        "top1_confidence_mean": float(conf.mean()),
        "top1_confidence_correct_mean": float(correct_conf.mean()) if correct_conf.numel() else None,
        "top1_confidence_wrong_mean": float(wrong_conf.mean()) if wrong_conf.numel() else None,
        "entropy_mean": float(entropy.mean()),
        "entropy_std": float(entropy.std(unbiased=False)) if entropy.numel() > 1 else 0.0,
        "top1_top2_margin_mean": float(margin.mean()),
        "top1_top2_margin_std": float(margin.std(unbiased=False)) if margin.numel() > 1 else 0.0,
        "negative_log_likelihood": float(nll),
        "brier_score": float(brier),
        "expected_calibration_error": float(ece),
        "num_bins": int(bins),
        "confidence_bins": table,
    }


def checkpoint_backbone_path(checkpoint_dir: str, cfg: Dict[str, Any]) -> Tuple[str, bool]:
    backbone = Path(checkpoint_dir) / "backbone"
    if backbone.exists():
        return str(backbone), True
    return str(cfg["model_name_or_path"]), False


def load_heads_if_available(model: QwenVLABaseline, checkpoint_dir: str) -> bool:
    path = Path(checkpoint_dir) / "action_heads.pt"
    if not path.exists():
        return False
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or not {"action_head", "command_head"} <= set(state):
        raise ValueError(
            f"Invalid action-head checkpoint; expected action_head and command_head: {path}"
        )
    model.action_head.load_state_dict(state["action_head"], strict=True)
    model.command_head.load_state_dict(state["command_head"], strict=True)
    shape_state = state.get("shape_command_head")
    if shape_state is not None and getattr(model, "shape_command_head", None) is not None:
        model.shape_command_head.load_state_dict(shape_state, strict=True)
    residual_state = state.get("residual_gate_head")
    if residual_state is not None and getattr(model, "residual_gate_head", None) is not None:
        model.residual_gate_head.load_state_dict(residual_state, strict=True)
    evidence_state = state.get("evidence_residual_head")
    if evidence_state is not None and getattr(model, "evidence_residual_head", None) is not None:
        model.evidence_residual_head.load_state_dict(evidence_state, strict=True)
    directed_state = state.get("evidence_directed_residual_head")
    if directed_state is not None and getattr(model, "evidence_directed_residual_head", None) is not None:
        model.evidence_directed_residual_head.load_state_dict(directed_state, strict=True)
    geov2x_state = state.get("geov2x_residual_head")
    if geov2x_state is not None and getattr(model, "geov2x_residual_head", None) is not None:
        model.geov2x_residual_head.load_state_dict(geov2x_state, strict=True)
    predicted_state = state.get("predicted_evidence_residual_head")
    if (
        predicted_state is not None
        and getattr(model, "predicted_evidence_residual_head", None) is not None
    ):
        model.predicted_evidence_residual_head.load_state_dict(predicted_state, strict=True)
    predicted_token_state = state.get("predicted_evidence_token_residual_head")
    if (
        predicted_token_state is not None
        and getattr(model, "predicted_evidence_token_residual_head", None) is not None
    ):
        model.predicted_evidence_token_residual_head.load_state_dict(predicted_token_state, strict=True)
    task_query_state = state.get("task_query_pooler")
    if task_query_state is not None and getattr(model, "task_query_pooler", None) is not None:
        model.task_query_pooler.load_state_dict(task_query_state, strict=True)
    return True


def load_baseline_config(checkpoint_dir: str) -> Dict[str, Any]:
    path = Path(checkpoint_dir) / "baseline_config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def validate_checkpoint_input_contract(
    cfg: Dict[str, Any],
    checkpoint_dir: str,
    *,
    resolved_head_pooling: str,
    resolved_vision_encode_per_view: bool,
) -> None:
    """Fail closed for the paper's canonical CEA evaluation interfaces."""
    contract = str(cfg.get("cea_input_contract") or "").strip()
    if not contract:
        return

    baseline = load_baseline_config(checkpoint_dir)
    if not baseline:
        raise ValueError(
            f"cea_input_contract={contract} requires checkpoint baseline_config.json"
        )
    checkpoint_is_anchor_readout = (
        str(baseline.get("head_pooling", "")).lower() == "causal_ego_anchor_mean"
        and baseline.get("vision_encode_per_view") is True
    )
    checkpoint_is_full_sequence_readout = (
        str(baseline.get("head_pooling", "")).lower() == "mean"
        and baseline.get("vision_encode_per_view") is True
    )

    common = {
        "use_trajectory_grounded_role": False,
        "part1_object_limit": 0,
        "part1_sentence_limit": 0,
        "compact_target_part1": False,
        "eval_batch_size": 1,
    }
    if contract == "causal_ego_anchor_compact_v1":
        if not checkpoint_is_anchor_readout:
            raise ValueError(
                f"cea_input_contract={contract} requires a CEA per-view checkpoint"
            )
        expected = {
            **common,
            "mode": "v2x_image",
            "causal_ego_anchor": True,
            "vision_encode_per_view": True,
            "view_order": "ego_infra",
            "image_max_pixels": 262144,
            "ego_image_max_pixels": 262144,
            "infra_image_max_pixels": 65536,
        }
        if str(resolved_head_pooling).lower() != "causal_ego_anchor_mean":
            raise ValueError("canonical CEA evaluation requires causal_ego_anchor_mean")
        if resolved_vision_encode_per_view is not True:
            raise ValueError("canonical CEA evaluation requires per-view vision encoding")
    elif contract == "causal_ego_anchor_full_sequence_mean_v1":
        if not checkpoint_is_full_sequence_readout:
            raise ValueError(
                f"cea_input_contract={contract} requires the matched full-sequence "
                "mean per-view checkpoint"
            )
        expected = {
            **common,
            "mode": "v2x_image",
            "causal_ego_anchor": True,
            "vision_encode_per_view": True,
            "view_order": "ego_infra",
            "image_max_pixels": 262144,
            "ego_image_max_pixels": 262144,
            "infra_image_max_pixels": 65536,
        }
        if str(resolved_head_pooling).lower() != "mean":
            raise ValueError("matched action-readout evaluation requires mean head pooling")
        if resolved_vision_encode_per_view is not True:
            raise ValueError(
                "matched action-readout evaluation requires per-view vision encoding"
            )
    elif contract == "declared_ego_control_v1":
        if not checkpoint_is_anchor_readout:
            raise ValueError(
                f"cea_input_contract={contract} requires a CEA per-view checkpoint"
            )
        expected = {
            **common,
            "mode": "ego_only",
            "neutral_ego_only_prompt": True,
            "causal_ego_anchor": False,
            "vision_encode_per_view": False,
            "image_max_pixels": 262144,
        }
        if str(resolved_head_pooling).lower() != "mean":
            raise ValueError("declared-ego control requires mean head pooling")
        if resolved_vision_encode_per_view is not False:
            raise ValueError("declared-ego control must disable per-view vision encoding")
    else:
        raise ValueError(f"Unknown cea_input_contract: {contract}")

    mismatches = {
        key: {"expected": value, "actual": cfg.get(key)}
        for key, value in expected.items()
        if cfg.get(key) != value
    }
    if mismatches:
        raise ValueError(f"CEA evaluation input contract mismatch: {mismatches}")


def model_loss_kwargs(cfg: Dict[str, Any], checkpoint_dir: str) -> Dict[str, Any]:
    baseline_cfg = load_baseline_config(checkpoint_dir)
    weights = baseline_cfg.get("command_class_weights", cfg.get("command_class_weights"))
    command_weights = torch.tensor(weights, dtype=torch.float32) if weights is not None else None
    fusion_mode = baseline_cfg.get("fusion_mode", cfg.get("fusion_mode"))
    default_evidence_dim = (
        ROUTE_EVIDENCE_FEATURE_DIM
        if str(fusion_mode or "").lower() == "evidence_directed_residual"
        else L1_EVIDENCE_FEATURE_DIM
    )
    return {
        "lambda_wp": float(baseline_cfg.get("lambda_wp", cfg.get("lambda_wp", 1.0))),
        "lambda_cmd": float(baseline_cfg.get("lambda_cmd", cfg.get("lambda_cmd", 0.2))),
        "lambda_fde": float(baseline_cfg.get("lambda_fde", cfg.get("lambda_fde", 0.0))),
        "lambda_residual_l2": float(
            baseline_cfg.get("lambda_residual_l2", cfg.get("lambda_residual_l2", 0.0))
        ),
        "lambda_gate_l1": float(baseline_cfg.get("lambda_gate_l1", cfg.get("lambda_gate_l1", 0.0))),
        "lambda_gate_prior": float(
            baseline_cfg.get("lambda_gate_prior", cfg.get("lambda_gate_prior", 0.0))
        ),
        "lambda_direct_residual": float(
            baseline_cfg.get("lambda_direct_residual", cfg.get("lambda_direct_residual", 0.0))
        ),
        "lambda_gate_benefit": float(
            baseline_cfg.get("lambda_gate_benefit", cfg.get("lambda_gate_benefit", 0.0))
        ),
        "command_class_weights": command_weights,
        "command_class_weight_reduction": str(
            baseline_cfg.get(
                "command_class_weight_reduction",
                cfg.get("command_class_weight_reduction", "mean"),
            )
        ),
        "command_class_weight_normalizer": float(
            baseline_cfg.get(
                "command_class_weight_normalizer",
                cfg.get("command_class_weight_normalizer", 1.0),
            )
        ),
        "command_head_mode": str(
            baseline_cfg.get("command_head_mode", cfg.get("command_head_mode", "standard"))
        ),
        "command_shape_hidden_size": int(
            baseline_cfg.get("command_shape_hidden_size", cfg.get("command_shape_hidden_size", 64))
        ),
        "head_pooling": str(
            cfg.get(
                "head_pooling_override",
                baseline_cfg.get("head_pooling", cfg.get("head_pooling", "last")),
            )
        ),
        "vision_encode_per_view": bool(
            cfg.get(
                "vision_encode_per_view_override",
                baseline_cfg.get(
                    "vision_encode_per_view",
                    cfg.get("vision_encode_per_view", False),
                ),
            )
        ),
        "fusion_mode": fusion_mode,
        "residual_gate_init_bias": float(
            baseline_cfg.get("residual_gate_init_bias", cfg.get("residual_gate_init_bias", -3.0))
        ),
        "evidence_feature_dim": int(
            baseline_cfg.get("evidence_feature_dim", cfg.get("evidence_feature_dim", default_evidence_dim))
        ),
        "evidence_hidden_size": int(
            baseline_cfg.get("evidence_hidden_size", cfg.get("evidence_hidden_size", 256))
        ),
        "evidence_layers": int(baseline_cfg.get("evidence_layers", cfg.get("evidence_layers", 2))),
        "evidence_heads": int(baseline_cfg.get("evidence_heads", cfg.get("evidence_heads", 4))),
        "evidence_gate_mode": str(
            baseline_cfg.get("evidence_gate_mode", cfg.get("evidence_gate_mode", "per_step"))
        ),
        "safety_mixer": baseline_cfg.get("safety_mixer", cfg.get("safety_mixer")),
        "base_shape_guard_alpha": float(
            baseline_cfg.get("base_shape_guard_alpha", cfg.get("base_shape_guard_alpha", 0.2))
        ),
        "base_shape_guard_waypoints_only": bool(
            baseline_cfg.get(
                "base_shape_guard_waypoints_only",
                cfg.get("base_shape_guard_waypoints_only", True),
            )
        ),
        "base_shape_guard_source": str(
            baseline_cfg.get(
                "base_shape_guard_source",
                cfg.get("base_shape_guard_source", "frozen_ego_base_waypoints_only"),
            )
        ),
        "turn_right_guard_alpha": float(
            baseline_cfg.get("turn_right_guard_alpha", cfg.get("turn_right_guard_alpha", 0.05))
        ),
        "turn_right_guard_prior_threshold": float(
            baseline_cfg.get(
                "turn_right_guard_prior_threshold",
                cfg.get("turn_right_guard_prior_threshold", 0.9),
            )
        ),
        "turn_right_guard_residual_threshold": float(
            baseline_cfg.get(
                "turn_right_guard_residual_threshold",
                cfg.get("turn_right_guard_residual_threshold", 0.1),
            )
        ),
        "right_turn_shape_guard_alpha": float(
            baseline_cfg.get(
                "right_turn_shape_guard_alpha",
                cfg.get("right_turn_shape_guard_alpha", 0.4),
            )
        ),
        "right_turn_shape_guard_prior_threshold": float(
            baseline_cfg.get(
                "right_turn_shape_guard_prior_threshold",
                cfg.get("right_turn_shape_guard_prior_threshold", 0.65),
            )
        ),
        "lambda_teacher_wp": float(
            baseline_cfg.get("lambda_teacher_wp", cfg.get("lambda_teacher_wp", 0.0))
        ),
        "lambda_teacher_cmd": float(
            baseline_cfg.get("lambda_teacher_cmd", cfg.get("lambda_teacher_cmd", 0.0))
        ),
        "geometry_dim": int(baseline_cfg.get("geometry_dim", cfg.get("geometry_dim", 11))),
        "geov2x_hidden_size": int(
            baseline_cfg.get("geov2x_hidden_size", cfg.get("geov2x_hidden_size", 256))
        ),
        "geov2x_heads": int(baseline_cfg.get("geov2x_heads", cfg.get("geov2x_heads", 4))),
        "lambda_counterfactual_base": float(
            baseline_cfg.get("lambda_counterfactual_base", cfg.get("lambda_counterfactual_base", 1.0))
        ),
        "lambda_counterfactual_gate": float(
            baseline_cfg.get("lambda_counterfactual_gate", cfg.get("lambda_counterfactual_gate", 0.1))
        ),
        "lambda_counterfactual_cmd": float(
            baseline_cfg.get("lambda_counterfactual_cmd", cfg.get("lambda_counterfactual_cmd", 0.2))
        ),
        "predicted_evidence_dim": int(
            baseline_cfg.get("predicted_evidence_dim", cfg.get("predicted_evidence_dim", 10))
        ),
        "predicted_evidence_hidden_size": int(
            baseline_cfg.get(
                "predicted_evidence_hidden_size",
                cfg.get("predicted_evidence_hidden_size", 256),
            )
        ),
        "lambda_predicted_evidence": float(
            baseline_cfg.get("lambda_predicted_evidence", cfg.get("lambda_predicted_evidence", 0.0))
        ),
        "task_query_dim": int(
            baseline_cfg.get("task_query_dim", cfg.get("task_query_dim", 256))
        ),
        "task_query_heads": int(
            baseline_cfg.get("task_query_heads", cfg.get("task_query_heads", 4))
        ),
        "task_query_dropout": float(
            baseline_cfg.get("task_query_dropout", cfg.get("task_query_dropout", 0.0))
        ),
        "lambda_task_query_ego_wp": float(
            baseline_cfg.get(
                "lambda_task_query_ego_wp",
                cfg.get("lambda_task_query_ego_wp", 0.0),
            )
        ),
        "lambda_task_query_ego_cmd": float(
            baseline_cfg.get(
                "lambda_task_query_ego_cmd",
                cfg.get("lambda_task_query_ego_cmd", 0.0),
            )
        ),
        "lambda_task_query_consistency_wp": float(
            baseline_cfg.get(
                "lambda_task_query_consistency_wp",
                cfg.get("lambda_task_query_consistency_wp", 0.0),
            )
        ),
        "lambda_task_query_consistency_cmd": float(
            baseline_cfg.get(
                "lambda_task_query_consistency_cmd",
                cfg.get("lambda_task_query_consistency_cmd", 0.0),
            )
        ),
        "lambda_task_query_infra_attention": float(
            baseline_cfg.get(
                "lambda_task_query_infra_attention",
                cfg.get("lambda_task_query_infra_attention", 0.0),
            )
        ),
    }


def residual_fusion_modes() -> set[str]:
    return {
        "ego_residual",
        "geov2x_residual",
        "predicted_evidence_residual",
        "predicted_evidence_token_residual",
        "evidence_residual",
        "evidence_directed_residual",
    }


def evidence_fusion_modes() -> set[str]:
    return {"evidence_residual", "evidence_directed_residual"}


def dataset_evidence_mode(cfg: Dict[str, Any]) -> str:
    if str(cfg.get("fusion_mode") or "").lower() == "evidence_directed_residual":
        return "route_conditioned"
    return "l1"


def normalize_cot_diagnostic_format(value: Any) -> str:
    normalized = str(value or "legacy_four_part").strip().lower().replace("-", "_")
    aliases = {
        "four_part": "legacy_four_part",
        "legacy": "legacy_four_part",
        "threepart": "three_part",
        "3part": "three_part",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"legacy_four_part", "three_part"}:
        raise ValueError(
            "cot_diagnostic_format must be legacy_four_part or three_part, "
            f"got {value!r}"
        )
    return normalized


def text_stats(
    texts: List[str], cot_diagnostic_format: str = "legacy_four_part"
) -> Dict[str, Any]:
    lengths = [len(t.split()) for t in texts]
    ref_count = sum(("I1" in t or "I2" in t or "infra" in t.lower()) for t in texts)
    n = max(len(texts), 1)
    stats = {
        "generated_count": len(texts),
        "avg_text_words": sum(lengths) / n,
        "infra_reference_rate": ref_count / n,
    }
    if normalize_cot_diagnostic_format(cot_diagnostic_format) == "three_part":
        three_part = aggregate_three_part_parses(
            parse_three_part_cot(text) for text in texts
        )
        stats.update(
            {
                "format_parse_rate": three_part["exact_three_part_parse_rate"],
                "three_part_language_diagnostic": three_part,
            }
        )
        return stats
    parseable = sum(("Part 1" in t and "Part 4" in t and "Command" in t) for t in texts)
    stats["format_parse_rate"] = parseable / n
    return stats


def parse_rate(text: str) -> bool:
    return "Part 1" in text and "Part 4" in text and "Command" in text


def cot_format_parse_ok(text: str, cot_diagnostic_format: str) -> bool:
    if normalize_cot_diagnostic_format(cot_diagnostic_format) == "three_part":
        return bool(parse_three_part_cot(text)["format_parse_ok"])
    return parse_rate(text)


PART_HEADER_RE = re.compile(
    r"(?i)\bPart\s*([1-4])\s*(?:[-:\u2013\u2014]|[.)]|\b(?:scene|overview|v2x|critical|decision|reasoning|action)\b)"
)
COMMAND_FIELD_RE = re.compile(r"(?i)\bCommand\s*:\s*(GO_STRAIGHT|TURN_LEFT|TURN_RIGHT|LATERAL_SHIFT|STOP)\b")


def trim_generated_cot_to_first_cycle(text: str) -> str:
    """Drop repeated CoT cycles after the first generated Part 4 block."""
    text = str(text or "")
    matches = list(PART_HEADER_RE.finditer(text))
    first_part4 = next((match for match in matches if match.group(1) == "4"), None)
    if first_part4 is None:
        return text
    command_match = COMMAND_FIELD_RE.search(text, first_part4.start())
    if command_match is not None:
        return text[: command_match.end()].rstrip()
    for match in matches:
        if match.start() > first_part4.start():
            return text[: match.start()].rstrip()
    return text.rstrip()


def extract_part4(text: str) -> str:
    lower = text.lower()
    idx = lower.rfind("part 4")
    return text[idx:] if idx >= 0 else text


def parse_raw_command(text: str) -> Optional[str]:
    valid_command = None
    for line in str(text or "").splitlines():
        lowered = line.lower()
        if "command" not in lowered or ":" not in line:
            continue
        if "one of" in lowered or "<" in line or ">" in line:
            continue
        value = line.split(":", 1)[1].strip().split()[0:3]
        command = canonical_command(" ".join(value).strip(" .,:;[]()"))
        if command != "UNKNOWN":
            valid_command = command
    return valid_command


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


def parse_raw_waypoints(text: str) -> Optional[List[List[float]]]:
    part4 = extract_part4(text)
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
        valid = True
        for item in parsed:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                valid = False
                break
            try:
                waypoints.append([float(item[0]), float(item[1])])
            except (TypeError, ValueError):
                valid = False
                break
        if valid:
            return waypoints


def format_waypoints(waypoints: List[List[float]]) -> str:
    return "[" + ", ".join(f"[{float(x):.2f}, {float(y):.2f}]" for x, y in waypoints) + "]"


def format_action_consistent_part4(command: str, waypoints: List[List[float]]) -> str:
    return (
        "Part 4 - Action:\n"
        f"The structured action head predicts the high-level command {command}.\n"
        f"Command: {command}"
    )


def waypoint_pair_metrics(raw_wp: List[List[float]], structured_wp: List[List[float]]) -> Dict[str, float]:
    dists = [
        math.hypot(float(raw[0]) - float(pred[0]), float(raw[1]) - float(pred[1]))
        for raw, pred in zip(raw_wp, structured_wp)
    ]
    return {"ade": sum(dists) / max(len(dists), 1), "fde": dists[-1] if dists else 0.0}


def generate_cot_texts(model: QwenVLABaseline, processor: Any, batch: Dict[str, Any], max_new_tokens: int = 256) -> List[str]:
    """Generate L4 CoT continuations from prompt-only processor inputs."""
    if not hasattr(model.backbone, "generate"):
        return [""] * int(batch["input_ids"].shape[0])
    gen_inputs = {
        k: v
        for k, v in batch.items()
        if torch.is_tensor(v)
        and k
        not in {
            "labels",
            "waypoints",
            "command_id",
            "head_attention_mask",
            "causal_anchor_attention_mask",
            "ego_view_mask",
            "infra_view_mask",
            "task_query_attention_mask",
        }
    }
    try:
        generated = model.backbone.generate(**gen_inputs, max_new_tokens=max_new_tokens, do_sample=False)
        prompt_len = int(gen_inputs["input_ids"].shape[1])
        new_tokens = generated[:, prompt_len:]
        if hasattr(processor, "batch_decode"):
            return processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        tokenizer = getattr(processor, "tokenizer", processor)
        return tokenizer.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    except Exception as exc:
        print(f"[WARN] CoT generation failed during evaluation: {exc}")
        return [""] * int(batch["input_ids"].shape[0])


def update_by_command(
    by_command: Dict[str, Dict[str, Any]],
    cmd_name: str,
    dist: torch.Tensor,
    pred_cmd: int,
    gt_cmd: int,
) -> None:
    rec = by_command[cmd_name]
    rec["count"] += 1
    rec["cmd_correct"] += int(pred_cmd == gt_cmd)
    rec["ade_sum"] += float(dist.mean())
    rec["fde_sum"] += float(dist[-1])


@torch.no_grad()
def evaluate(
    config_path: str,
    checkpoint_dir: str,
    split: str = "val",
    max_batches: int = 0,
    examples_output: Optional[str] = None,
    examples_every_n: int = 50,
    generate_all_text: Optional[bool] = None,
    action_first_report: bool = False,
    topk_bad_cases: int = 20,
    structured_output: Optional[str] = None,
    profile_eval: Optional[bool] = None,
    profile_warmup_samples: Optional[int] = None,
    profile_max_measured_samples: Optional[int] = None,
    profile_output: Optional[str] = None,
    profile_no_cot: bool = True,
    cot_diagnostic_format: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = load_config(config_path)
    configured_cot_format = cot_diagnostic_format or cfg.get("cot_diagnostic_format")
    if configured_cot_format is None:
        configured_cot_format = (
            "three_part"
            if str(cfg.get("lm_target_policy") or "") == "parts1_3_only"
            else "legacy_four_part"
        )
    resolved_cot_format = normalize_cot_diagnostic_format(configured_cot_format)
    three_part_diagnostic = resolved_cot_format == "three_part"
    profile_enabled = bool(cfg.get("profile_eval", False) if profile_eval is None else profile_eval)
    if generate_all_text is None:
        generate_all_text = bool(cfg.get("eval_generate_all_text", False))
    if profile_enabled and profile_no_cot and not bool(generate_all_text):
        examples_output = None
    warmup_samples = int(
        cfg.get("profile_warmup_samples", 20) if profile_warmup_samples is None else profile_warmup_samples
    )
    max_measured_samples = int(
        cfg.get("profile_max_measured_samples", 0)
        if profile_max_measured_samples is None
        else profile_max_measured_samples
    )
    model_path, loaded_checkpoint_backbone = checkpoint_backbone_path(checkpoint_dir, cfg)
    processor = import_processor(str(cfg["model_name_or_path"]))
    dataset_split = None if split == "trainval" else split
    dataset = CoVLABaselineDataset(
        cfg["index_path"],
        mode=cfg.get("mode", "v2x_image"),
        split=dataset_split,
        load_images=False,
        use_ego_status_prompt=bool(cfg.get("use_ego_status_prompt", False)),
        neutral_ego_only_prompt=bool(cfg.get("neutral_ego_only_prompt", False)),
        part1_object_limit=cfg_int(cfg, "part1_object_limit", DEFAULT_PART1_OBJECT_LIMIT),
        part1_sentence_limit=cfg_int(cfg, "part1_sentence_limit", DEFAULT_PART1_SENTENCE_LIMIT),
        compact_target_part1=bool(cfg.get("compact_target_part1", False)),
        base_prediction_path=(
            cfg.get("base_prediction_path")
            if str(cfg.get("fusion_mode") or "").lower() in residual_fusion_modes()
            else None
        ),
        use_geov2x_geometry=str(cfg.get("fusion_mode") or "").lower() in {
            "geov2x_residual",
            "predicted_evidence_residual",
            "predicted_evidence_token_residual",
        },
        geov2x_geometry_zero=bool(cfg.get("geov2x_geometry_zero", cfg.get("geometry_ablation_zero", False))),
        v2x_root=str(cfg.get("v2x_root", "data/V2X-Seq-SPD")),
        infra_condition=str(cfg.get("infra_condition", "normal")),
        infra_condition_mix=None,
        shuffled_infra_map_path=cfg.get("shuffled_infra_map_path"),
        use_l1_evidence_tokens=str(cfg.get("fusion_mode") or "").lower() in evidence_fusion_modes(),
        evidence_top_k=int(cfg.get("route_evidence_top_k", cfg.get("evidence_top_k", 12))),
        evidence_mode=dataset_evidence_mode(cfg),
        route_corridor_width_m=float(cfg.get("route_corridor_width_m", 6.0)),
        route_near_horizon_steps=int(cfg.get("route_near_horizon_steps", 6)),
        route_relevance_variant=str(cfg.get("route_relevance_variant", "strict")),
        # Teacher predictions supervise training only; evaluation must remain
        # independent of train-split teacher artifacts.
        teacher_prediction_path=None,
        teacher_exclude_decision_relevant=False,
        route_risk_supervision=bool(cfg.get("route_risk_supervision", False)),
        use_trajectory_grounded_role=bool(
            cfg.get("use_trajectory_grounded_role", False)
        ),
        trajectory_relevant_repeat=1,
        view_order=str(cfg.get("view_order", "ego_infra")),
        causal_ego_anchor=bool(cfg.get("causal_ego_anchor", False)),
    )
    collator = CoVLACollator(
        processor,
        mode=cfg.get("mode", "v2x_image"),
        image_max_pixels=cfg.get("image_max_pixels"),
        include_targets=False,
        sample_passthrough_keys=cfg.get("sample_passthrough_keys"),
        task_query_text=cfg.get("task_query_text"),
        task_query_tail_tokens=int(cfg.get("task_query_tail_tokens", 0)),
        view_order=str(cfg.get("view_order", "ego_infra")),
        ego_image_max_pixels=cfg.get("ego_image_max_pixels"),
        infra_image_max_pixels=cfg.get("infra_image_max_pixels"),
        causal_ego_anchor=bool(cfg.get("causal_ego_anchor", False)),
    )
    loader = DataLoader(dataset, batch_size=int(cfg.get("eval_batch_size", 1)), shuffle=False, collate_fn=collator, num_workers=0)
    if profile_enabled and int(cfg.get("eval_batch_size", 1)) != 1:
        raise ValueError("common evaluation profiler requires eval_batch_size=1")
    loss_kwargs = model_loss_kwargs(cfg, checkpoint_dir)
    validate_causal_anchor_config(
        cfg,
        resolved_head_pooling=str(loss_kwargs["head_pooling"]),
    )
    validate_checkpoint_input_contract(
        cfg,
        checkpoint_dir,
        resolved_head_pooling=str(loss_kwargs["head_pooling"]),
        resolved_vision_encode_per_view=bool(loss_kwargs["vision_encode_per_view"]),
    )
    model = QwenVLABaseline(
        model_name_or_path=model_path,
        **loss_kwargs,
        use_lora=False,
        gradient_checkpointing=False,
        bf16=bool(cfg.get("bf16", True)),
        device_map=cfg.get("device_map"),
    )
    if not load_heads_if_available(model, checkpoint_dir):
        raise FileNotFoundError(
            f"Evaluation requires checkpoint action heads: {Path(checkpoint_dir) / 'action_heads.pt'}"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not cfg.get("device_map"):
        model.to(device)
    model.eval()
    parameter_profile = parameter_group_summary(model) if profile_enabled else None
    checkpoint_profile = checkpoint_size_summary(checkpoint_dir) if profile_enabled else None
    profile_input_provenance = None
    if profile_enabled:
        profile_input_provenance = {
            "config": profile_file_artifact(Path(config_path)),
            "index": profile_file_artifact(Path(str(cfg["index_path"]))),
            "checkpoint": profile_checkpoint_provenance(Path(checkpoint_dir)),
        }
    profile_start_rss, profile_rss_source = cpu_rss_bytes()
    profile_smi_before = query_nvidia_smi() if profile_enabled else None
    if profile_enabled and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    warmup_done = 0
    warmup_tokens: List[str] = []
    if profile_enabled and warmup_samples > 0:
        warmup_iter = iter(loader)
        while warmup_done < warmup_samples:
            try:
                warmup_batch = next(warmup_iter)
            except StopIteration:
                break
            warmup_n = len(warmup_batch["samples"])
            warmup_tokens.extend(
                str(sample.get("token") or sample.get("sample_id") or "")
                for sample in warmup_batch["samples"]
            )
            warmup_batch = to_device(warmup_batch, device)
            _ = model(**warmup_batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            warmup_done += warmup_n
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    ade_sum = torch.zeros(6)
    fde_sum = 0.0
    cmd_correct = 0
    count = 0
    fde_values: List[float] = []
    ade_values: List[float] = []
    step_error_values: List[List[float]] = [[] for _ in range(6)]
    base_fde_delta_values: List[float] = []
    base_ade_delta_values: List[float] = []
    base_improved_count = 0
    base_degraded_count = 0
    base_tied_count = 0
    all_command_logits: List[torch.Tensor] = []
    all_gt_command_ids: List[torch.Tensor] = []
    confusion = torch.zeros((len(COMMANDS), len(COMMANDS)), dtype=torch.long)
    pred_command_counts = {cmd: 0 for cmd in COMMANDS}
    generated_texts: List[str] = []
    by_command: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "cmd_correct": 0, "ade_sum": 0.0, "fde_sum": 0.0})
    bad_cases: List[Dict[str, Any]] = []
    examples_f = None
    examples_written = 0
    examples_indices: List[int] = []
    examples_every_n = max(1, int(examples_every_n))
    action_stats: Dict[str, Any] = {
        "generated_count": 0,
        "raw_format_ok": 0,
        "raw_command_parse_ok": 0,
        "raw_command_structured_agree": 0,
        "raw_waypoint_parse_ok": 0,
        "raw_waypoint_structured_ade_sum": 0.0,
        "raw_waypoint_structured_fde_sum": 0.0,
        "action_consistent_part4_valid": 0,
    }
    gate_values: List[float] = []
    delta_norms: List[float] = []
    alpha_values: List[float] = []
    guard_groups: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "cmd_correct": 0, "ade_sum": 0.0, "fde_sum": 0.0}
    )
    profile_model_ms: List[float] = []
    profile_device_model_return_ms: List[float] = []
    profile_device_model_postprocess_ms: List[float] = []
    profile_end_to_end_ms: List[float] = []
    profile_to_device_ms: List[float] = []
    profile_postprocess_ms: List[float] = []
    profile_data_ms: List[float] = []
    profile_input_stats: Dict[str, List[float]] = {}
    profile_measured = 0
    profile_measured_tokens: List[str] = []
    profile_seen = 0
    profile_batch_shapes: Dict[str, Any] = {}
    profile_wall_start = time.perf_counter()
    if examples_output:
        Path(examples_output).parent.mkdir(parents=True, exist_ok=True)
        examples_f = Path(examples_output).open("w", encoding="utf-8")
    structured_f = None
    if structured_output:
        Path(structured_output).parent.mkdir(parents=True, exist_ok=True)
        structured_f = Path(structured_output).open("w", encoding="utf-8")
    loader_iter = iter(loader)
    bi = 0
    while True:
        load_start = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        load_elapsed = time.perf_counter() - load_start
        samples = batch["samples"]
        batch_size = len(samples)
        batch_profile_start = time.perf_counter()
        measured_batch = False
        if profile_enabled:
            if max_measured_samples <= 0 or profile_measured < max_measured_samples:
                measured_batch = True
            if measured_batch and batch_size > 0:
                merge_stat_lists(profile_input_stats, collect_input_stats(batch, samples))
                for key, value in batch.items():
                    shape = tensor_shape(value)
                    if shape is not None:
                        profile_batch_shapes.setdefault(key, shape)
        to_device_start = time.perf_counter()
        batch = to_device(batch, device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        to_device_elapsed = time.perf_counter() - to_device_start
        model_start_event = model_end_event = None
        model_wall_start = time.perf_counter()
        if measured_batch and torch.cuda.is_available():
            model_start_event = torch.cuda.Event(enable_timing=True)
            model_end_event = torch.cuda.Event(enable_timing=True)
            model_start_event.record()
        out = model(**batch)
        if model_end_event is not None:
            model_end_event.record()
            torch.cuda.synchronize()
            model_elapsed_ms = float(model_start_event.elapsed_time(model_end_event))  # type: ignore[union-attr]
        else:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_elapsed_ms = (time.perf_counter() - model_wall_start) * 1000.0
        device_model_return_elapsed_ms = (time.perf_counter() - to_device_start) * 1000.0
        postprocess_start = time.perf_counter()
        pred_wp = out["pred_waypoints"].detach().cpu()
        gt_wp = batch["waypoints"].detach().cpu()
        dist = torch.norm(pred_wp - gt_wp, dim=-1)
        ade_sum += dist.sum(dim=0)
        fde_sum += float(dist[:, -1].sum())
        pred_cmd = out["command_logits"].argmax(dim=-1).detach().cpu()
        command_logits = out["command_logits"].detach().float().cpu()
        gt_cmd = batch["command_id"].detach().cpu()
        all_command_logits.append(command_logits)
        all_gt_command_ids.append(gt_cmd)
        gate_tensor = out.get("gate")
        delta_tensor = out.get("delta_waypoints")
        predicted_evidence_cpu = out.get("predicted_evidence")
        if torch.is_tensor(predicted_evidence_cpu):
            predicted_evidence_cpu = predicted_evidence_cpu.detach().float().cpu()
        gate_cpu = gate_tensor.detach().float().cpu().view(-1) if gate_tensor is not None else None
        delta_cpu = delta_tensor.detach().float().cpu() if delta_tensor is not None else None
        base_wp_cpu = out.get("base_waypoints")
        if torch.is_tensor(base_wp_cpu):
            base_wp_cpu = base_wp_cpu.detach().float().cpu()
        base_logits_cpu = out.get("base_command_logits")
        if torch.is_tensor(base_logits_cpu):
            base_logits_cpu = base_logits_cpu.detach().float().cpu()
        risk_zone_wp_cpu = out.get("risk_zone_waypoints")
        if torch.is_tensor(risk_zone_wp_cpu):
            risk_zone_wp_cpu = risk_zone_wp_cpu.detach().float().cpu()
        alpha_cpu = out.get("alpha_shape")
        if torch.is_tensor(alpha_cpu):
            alpha_cpu = alpha_cpu.detach().float().cpu().view(-1)
            alpha_values.extend([float(x) for x in alpha_cpu])
        guard_cpu = out.get("base_shape_guard_applied")
        if torch.is_tensor(guard_cpu):
            guard_cpu = guard_cpu.detach().bool().cpu().view(-1)
        turn_guard_cpu = out.get("turn_right_guard_applied")
        if torch.is_tensor(turn_guard_cpu):
            turn_guard_cpu = turn_guard_cpu.detach().bool().cpu().view(-1)
        turn_guard_norm_cpu = out.get("turn_right_guard_residual_norm")
        if torch.is_tensor(turn_guard_norm_cpu):
            turn_guard_norm_cpu = turn_guard_norm_cpu.detach().float().cpu().view(-1)
        if gate_cpu is not None:
            gate_values.extend([float(x) for x in gate_cpu])
        if delta_cpu is not None:
            norms = delta_cpu.norm(dim=-1).mean(dim=1)
            delta_norms.extend([float(x) for x in norms])
        cmd_correct += int((pred_cmd == gt_cmd).sum())
        fde_values.extend(float(x) for x in dist[:, -1].tolist())
        ade_values.extend(float(x) for x in dist.mean(dim=1).tolist())
        for step_idx in range(min(6, dist.shape[1])):
            step_error_values[step_idx].extend(float(x) for x in dist[:, step_idx].tolist())
        if base_wp_cpu is not None:
            base_dist = torch.norm(base_wp_cpu - gt_wp, dim=-1)
            final_fde = dist[:, -1]
            base_fde = base_dist[:, -1]
            final_ade = dist.mean(dim=1)
            base_ade = base_dist.mean(dim=1)
            deltas = (base_fde - final_fde).tolist()
            base_fde_delta_values.extend(float(x) for x in deltas)
            base_ade_delta_values.extend(float(x) for x in (base_ade - final_ade).tolist())
            base_improved_count += int((final_fde < base_fde).sum())
            base_degraded_count += int((final_fde > base_fde).sum())
            base_tied_count += int((final_fde == base_fde).sum())
        should_generate_rows = [
            row
            for row in range(len(samples))
            if generate_all_text or (examples_f is not None and (count + row) % examples_every_n == 0)
        ]
        gen_texts = [""] * len(samples)
        if should_generate_rows:
            gen_texts = generate_cot_texts(model, processor, batch, max_new_tokens=int(cfg.get("eval_max_new_tokens", 256)))
            if bool(cfg.get("eval_trim_generated_cot_to_first_cycle", False)):
                gen_texts = [trim_generated_cot_to_first_cycle(text) for text in gen_texts]
        for row, sample in enumerate(samples):
            global_index = count + row
            gt_cmd_id = int(gt_cmd[row])
            pred_cmd_id = int(pred_cmd[row])
            cmd_name = ID2COMMAND.get(gt_cmd_id, "UNKNOWN")
            if 0 <= gt_cmd_id < len(COMMANDS) and 0 <= pred_cmd_id < len(COMMANDS):
                confusion[gt_cmd_id, pred_cmd_id] += 1
                pred_command_counts[ID2COMMAND.get(pred_cmd_id, "UNKNOWN")] += 1
            update_by_command(by_command, cmd_name, dist[row], pred_cmd_id, gt_cmd_id)
            if torch.is_tensor(guard_cpu):
                group = "guarded" if bool(guard_cpu[row]) else "unguarded"
                guard_rec = guard_groups[group]
                guard_rec["count"] += 1
                guard_rec["cmd_correct"] += int(pred_cmd_id == gt_cmd_id)
                guard_rec["ade_sum"] += float(dist[row].mean())
                guard_rec["fde_sum"] += float(dist[row, -1])
            final_error = float(dist[row, -1])
            structured_command = ID2COMMAND.get(pred_cmd_id, "UNKNOWN")
            structured_waypoints = pred_wp[row].tolist()
            if structured_f:
                payload = {
                    "global_index": global_index,
                    "sample_id": sample.get("sample_id"),
                    "token": sample.get("token"),
                    "split": sample.get("split", split),
                    "pred_waypoints": structured_waypoints,
                    "gt_waypoints": gt_wp[row].tolist(),
                    "pred_command": structured_command,
                    "gt_command": ID2COMMAND.get(gt_cmd_id, "UNKNOWN"),
                    "command_logits": command_logits[row].tolist(),
                    "pred_command_logits": command_logits[row].tolist(),
                    "ade": float(dist[row].mean()),
                    "fde": final_error,
                }
                if split == "trainval":
                    payload["global_index"] = global_index
                if gate_cpu is not None:
                    payload.update(
                        {
                            "gate": float(gate_cpu[row]),
                            "gate_value": float(gate_cpu[row]),
                        }
                    )
                if delta_cpu is not None:
                    delta_norm = float(delta_cpu[row].norm(dim=-1).mean())
                    payload.update(
                        {
                            "delta_waypoints": delta_cpu[row].tolist(),
                            "delta_waypoint_norm": delta_norm,
                        }
                    )
                if base_wp_cpu is not None:
                    payload["base_waypoints"] = base_wp_cpu[row].tolist()
                elif sample.get("base_waypoints") is not None:
                    payload["base_waypoints"] = sample.get("base_waypoints")
                if base_logits_cpu is not None:
                    payload["base_command_logits"] = base_logits_cpu[row].tolist()
                elif sample.get("base_command_logits") is not None:
                    payload["base_command_logits"] = sample.get("base_command_logits")
                if risk_zone_wp_cpu is not None:
                    payload["risk_zone_waypoints"] = risk_zone_wp_cpu[row].tolist()
                    payload["final_mixed_waypoints"] = structured_waypoints
                if torch.is_tensor(predicted_evidence_cpu):
                    payload["predicted_evidence"] = predicted_evidence_cpu[row].tolist()
                    payload["predicted_route_risk_vector"] = predicted_evidence_cpu[row].tolist()
                    payload["val_gt_evidence_fed_to_planner"] = False
                if torch.is_tensor(alpha_cpu):
                    payload["alpha_shape"] = float(alpha_cpu[row])
                if torch.is_tensor(guard_cpu):
                    payload["base_shape_guard_applied"] = bool(guard_cpu[row])
                    payload["safety_mixer"] = cfg.get("safety_mixer")
                    payload["base_shape_guard_source"] = cfg.get(
                        "base_shape_guard_source",
                        "frozen_ego_base_waypoints_only",
                    )
                if torch.is_tensor(turn_guard_cpu):
                    payload["turn_right_guard_applied"] = bool(turn_guard_cpu[row])
                if torch.is_tensor(turn_guard_norm_cpu):
                    payload["turn_right_guard_residual_norm"] = float(turn_guard_norm_cpu[row])
                if sample.get("base_command") is not None:
                    payload["base_command"] = sample.get("base_command")
                if sample.get("v2x_relevance_prior") is not None:
                    payload["v2x_relevance_prior"] = sample.get("v2x_relevance_prior")
                if sample.get("route_relevance_prior") is not None:
                    payload["route_relevance_prior"] = sample.get("route_relevance_prior")
                if sample.get("object_evidence_summary") is not None:
                    payload["object_evidence_summary"] = sample.get("object_evidence_summary")
                    features = (
                        sample.get("object_evidence_summary", {}).get("lateral_shape_features")
                        if isinstance(sample.get("object_evidence_summary"), dict)
                        else None
                    )
                    if features is not None:
                        payload["base_shape_guard_features"] = features
                structured_f.write(
                    json.dumps(payload, ensure_ascii=False)
                    + "\n"
                )
            bad_cases.append(
                {
                    "sample_id": sample.get("sample_id"),
                    "token": sample.get("token"),
                    "gt_command": ID2COMMAND.get(gt_cmd_id, "UNKNOWN"),
                    "pred_command": structured_command,
                    "fde": final_error,
                    "ade": float(dist[row].mean()),
                }
            )
            should_record_text = bool(generate_all_text or (examples_f and global_index % examples_every_n == 0))
            text = gen_texts[row] if row < len(gen_texts) else ""
            three_part_parse = (
                parse_three_part_cot(text)
                if should_record_text and three_part_diagnostic
                else None
            )
            raw_command = (
                parse_raw_command(text)
                if should_record_text and not three_part_diagnostic
                else None
            )
            raw_waypoints = (
                parse_raw_waypoints(text)
                if should_record_text and not three_part_diagnostic
                else None
            )
            raw_waypoint_metrics = None
            if should_record_text:
                generated_texts.append(text)
                action_stats["generated_count"] += 1
                action_stats["raw_format_ok"] += int(
                    bool(three_part_parse["format_parse_ok"])
                    if three_part_parse is not None
                    else parse_rate(text)
                )
                if not three_part_diagnostic:
                    action_stats["raw_command_parse_ok"] += int(raw_command is not None)
                    action_stats["raw_command_structured_agree"] += int(raw_command == structured_command)
                    action_stats["raw_waypoint_parse_ok"] += int(raw_waypoints is not None)
                    if raw_waypoints is not None:
                        raw_waypoint_metrics = waypoint_pair_metrics(raw_waypoints, structured_waypoints)
                        action_stats["raw_waypoint_structured_ade_sum"] += raw_waypoint_metrics["ade"]
                        action_stats["raw_waypoint_structured_fde_sum"] += raw_waypoint_metrics["fde"]
                    action_stats["action_consistent_part4_valid"] += int(
                        structured_command in COMMANDS and len(structured_waypoints) == 6
                    )
            if examples_f and global_index % examples_every_n == 0:
                text = gen_texts[row] if row < len(gen_texts) else ""
                if action_first_report:
                    payload = {
                        "global_index": global_index,
                        "sample_id": sample.get("sample_id"),
                        "token": sample.get("token"),
                        "split": sample.get("split", split),
                        "raw_generated_cot": text,
                        "structured_pred_command": structured_command,
                        "structured_pred_waypoints": structured_waypoints,
                        "gt_command": ID2COMMAND.get(gt_cmd_id, "UNKNOWN"),
                        "gt_waypoints": gt_wp[row].tolist(),
                        "format_parse_ok": cot_format_parse_ok(
                            text, resolved_cot_format
                        ),
                        "cot_diagnostic_format": resolved_cot_format,
                        "ade": float(dist[row].mean()),
                        "fde": final_error,
                    }
                    if three_part_diagnostic:
                        payload["three_part_parse"] = three_part_parse or parse_three_part_cot(text)
                    else:
                        payload.update(
                            {
                                "action_consistent_part4": format_action_consistent_part4(
                                    structured_command,
                                    structured_waypoints,
                                ),
                                "raw_parsed_command": raw_command,
                                "raw_command_agrees_with_structured": raw_command
                                == structured_command,
                                "raw_waypoints_parse_ok": raw_waypoints is not None,
                            }
                        )
                        if raw_waypoint_metrics is not None:
                            payload["raw_waypoint_vs_structured_ade"] = raw_waypoint_metrics["ade"]
                            payload["raw_waypoint_vs_structured_fde"] = raw_waypoint_metrics["fde"]
                else:
                    payload = {
                            "global_index": global_index,
                            "sample_id": sample.get("sample_id"),
                            "token": sample.get("token"),
                            "split": sample.get("split", split),
                            "generated_text": text,
                            "gt_command": ID2COMMAND.get(gt_cmd_id, "UNKNOWN"),
                            "pred_command": structured_command,
                            "gt_waypoints": gt_wp[row].tolist(),
                            "pred_waypoints": structured_waypoints,
                            "format_parse_ok": cot_format_parse_ok(
                                text, resolved_cot_format
                            ),
                            "cot_diagnostic_format": resolved_cot_format,
                            "ade": float(dist[row].mean()),
                            "fde": final_error,
                    }
                    if three_part_diagnostic:
                        payload["three_part_parse"] = three_part_parse or parse_three_part_cot(text)
                examples_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                examples_written += 1
                examples_indices.append(global_index)
        postprocess_elapsed = time.perf_counter() - postprocess_start
        if measured_batch:
            add_n = batch_size
            if max_measured_samples > 0:
                add_n = max(0, min(batch_size, max_measured_samples - profile_measured))
            if add_n > 0:
                denom_batch = max(batch_size, 1)
                data_ms = load_elapsed * 1000.0 / denom_batch
                device_model_postprocess_ms = (
                    (time.perf_counter() - batch_profile_start) * 1000.0 / denom_batch
                )
                profile_model_ms.extend([model_elapsed_ms / denom_batch] * add_n)
                profile_device_model_return_ms.extend(
                    [device_model_return_elapsed_ms / denom_batch] * add_n
                )
                profile_device_model_postprocess_ms.extend([device_model_postprocess_ms] * add_n)
                profile_end_to_end_ms.extend([data_ms + device_model_postprocess_ms] * add_n)
                profile_to_device_ms.extend([to_device_elapsed * 1000.0 / denom_batch] * add_n)
                profile_postprocess_ms.extend([postprocess_elapsed * 1000.0 / denom_batch] * add_n)
                profile_data_ms.extend([data_ms] * add_n)
                profile_measured_tokens.extend(
                    str(sample.get("token") or sample.get("sample_id") or "")
                    for sample in samples[:add_n]
                )
                profile_measured += add_n
        profile_seen += batch_size
        count += gt_wp.shape[0]
        if max_batches and bi + 1 >= max_batches:
            break
        bi += 1
    if examples_f:
        examples_f.close()
    if structured_f:
        structured_f.close()

    denom = max(count, 1)
    per_command_metrics = {
        cmd: {
            "count": rec["count"],
            "command_accuracy": rec["cmd_correct"] / max(rec["count"], 1),
            "ADE": rec["ade_sum"] / max(rec["count"], 1),
            "FDE": rec["fde_sum"] / max(rec["count"], 1),
        }
        for cmd, rec in sorted(by_command.items())
    }
    present_command_acc = [
        rec["command_accuracy"]
        for rec in per_command_metrics.values()
        if rec["count"] > 0
    ]
    all_command_acc = [
        per_command_metrics.get(cmd, {"command_accuracy": 0.0})["command_accuracy"]
        for cmd in COMMANDS
    ]
    generated_for_action = max(int(action_stats["generated_count"]), 1)
    raw_waypoint_parsed = max(int(action_stats["raw_waypoint_parse_ok"]), 1)
    if three_part_diagnostic:
        language_diagnostic = aggregate_three_part_parses(
            parse_three_part_cot(text) for text in generated_texts
        )
        language_diagnostic.update(
            {
                "scope": "all_samples" if generate_all_text else "sparse_examples",
                "reporting_note": (
                    "Language contains only scene understanding, critical evidence, "
                    "and decision reasoning. Text-command and text-waypoint agreement "
                    "are outside the contract and were not computed."
                ),
            }
        )
        raw_vs_structured = None
    else:
        language_diagnostic = None
        raw_vs_structured = {
            "scope": "all_samples" if generate_all_text else "sparse_examples",
            "generated_count": int(action_stats["generated_count"]),
            "raw_generated_cot_format_parse_rate": action_stats["raw_format_ok"] / generated_for_action,
            "raw_part4_command_parse_rate": action_stats["raw_command_parse_ok"] / generated_for_action,
            "raw_part4_command_vs_structured_agreement_rate": (
                action_stats["raw_command_structured_agree"] / generated_for_action
            ),
            "raw_part4_waypoint_parse_rate": action_stats["raw_waypoint_parse_ok"] / generated_for_action,
            "raw_waypoint_vs_structured_ADE": action_stats["raw_waypoint_structured_ade_sum"] / raw_waypoint_parsed,
            "raw_waypoint_vs_structured_FDE": action_stats["raw_waypoint_structured_fde_sum"] / raw_waypoint_parsed,
            "action_consistent_part4_valid_rate": action_stats["action_consistent_part4_valid"] / generated_for_action,
            "reporting_note": (
                "raw_generated_cot is preserved separately from structured_pred_command and "
                "structured_pred_waypoints; action_consistent_part4 is derived from structured heads. "
                "Raw text is not expected to emit numeric waypoints in this run."
            ),
        }
    per_step_error_distribution = [
        {
            "step": idx,
            "median": percentile(values, 50),
            "p90": percentile(values, 90),
            "p95": percentile(values, 95),
            "count": len(values),
        }
        for idx, values in enumerate(step_error_values)
    ]
    base_improvement_stats = {
        "status": "ok" if base_fde_delta_values else "unavailable_no_base_waypoints",
        "improved_count": base_improved_count,
        "degraded_count": base_degraded_count,
        "tied_count": base_tied_count,
        "FDE_improvement_distribution": value_distribution(base_fde_delta_values),
        "ADE_improvement_distribution": value_distribution(base_ade_delta_values),
    }
    calibration_stats = (
        command_calibration_stats(torch.cat(all_command_logits, dim=0), torch.cat(all_gt_command_ids, dim=0))
        if all_command_logits
        else {"status": "empty"}
    )
    residual_stats = None
    if gate_values:
        sorted_gate = sorted(gate_values)
        sorted_delta = sorted(delta_norms)
        p90_idx = min(len(sorted_gate) - 1, int(0.9 * (len(sorted_gate) - 1)))
        d90_idx = min(len(sorted_delta) - 1, int(0.9 * (len(sorted_delta) - 1)))
        residual_stats = {
            "gate_mean": sum(gate_values) / len(gate_values),
            "gate_p90": sorted_gate[p90_idx],
            "gate_max": max(gate_values),
            "delta_waypoint_norm_mean": sum(delta_norms) / max(len(delta_norms), 1),
            "delta_waypoint_norm_p90": sorted_delta[d90_idx] if sorted_delta else 0.0,
        }
    base_shape_guard_diagnostics = None
    if alpha_values:
        base_shape_guard_diagnostics = {
            "alpha_shape_mean": sum(alpha_values) / len(alpha_values),
            "alpha_shape_min": min(alpha_values),
            "alpha_shape_max": max(alpha_values),
            "guarded_count": int(sum(1 for x in alpha_values if x < 0.999)),
            "groups": {
                name: {
                    "count": rec["count"],
                    "command_accuracy": rec["cmd_correct"] / max(rec["count"], 1),
                    "ADE": rec["ade_sum"] / max(rec["count"], 1),
                    "FDE": rec["fde_sum"] / max(rec["count"], 1),
                }
                for name, rec in sorted(guard_groups.items())
            },
        }
    profile_end_rss, _ = cpu_rss_bytes()
    profile_smi_after = query_nvidia_smi() if profile_enabled else None
    measured_model_sec = sum(profile_model_ms) / 1000.0
    measured_device_model_return_sec = sum(profile_device_model_return_ms) / 1000.0
    measured_device_model_postprocess_sec = sum(profile_device_model_postprocess_ms) / 1000.0
    measured_e2e_sec = sum(profile_end_to_end_ms) / 1000.0
    cuda_peak_alloc = torch.cuda.max_memory_allocated() if profile_enabled and torch.cuda.is_available() else None
    cuda_peak_reserved = torch.cuda.max_memory_reserved() if profile_enabled and torch.cuda.is_available() else None
    cuda_alloc_after = torch.cuda.memory_allocated() if profile_enabled and torch.cuda.is_available() else None
    cuda_reserved_after = torch.cuda.memory_reserved() if profile_enabled and torch.cuda.is_available() else None
    compute_profile = {
        "enabled": bool(profile_enabled),
        "profile_eval_version": 3,
        "profile_scope": (
            "common_accepted_token_device_model_return"
            if not generate_all_text
            else "structured_planning_with_explicit_text_generation"
        ),
        "timing_contract": (
            "batch 1; data loading and metric/serialization postprocessing excluded; "
            "CPU-to-device transfer, model inference, and returned tensor construction included"
        ),
        "status": "ok" if profile_enabled else "disabled",
        "measured_count": int(profile_measured if profile_enabled else 0),
        "warmup_count": int(warmup_done if profile_enabled else 0),
        "warmup_tokens": warmup_tokens if profile_enabled else [],
        "measured_tokens": profile_measured_tokens if profile_enabled else [],
        "profile_max_measured_samples": int(max_measured_samples),
        "measured_sample_policy": (
            "Warmup samples run in a separate pass before timing. Planning metrics still cover "
            "the evaluated split/max_batches rows. If profile_max_measured_samples is 0, timing "
            "is collected for all evaluated rows; otherwise timing is truncated to the first "
            "profile_max_measured_samples evaluated rows."
        ),
        "profile_warning": (
            "Timing distributions are truncated by profile_max_measured_samples and may not "
            "represent the full eval set."
            if max_measured_samples > 0
            else None
        ),
        "split": split,
        "eval_batch_size": int(cfg.get("eval_batch_size", 1)),
        "num_workers": 0,
        "image_max_pixels": cfg.get("image_max_pixels"),
        "ego_image_max_pixels": cfg.get("ego_image_max_pixels"),
        "infra_image_max_pixels": cfg.get("infra_image_max_pixels"),
        "mode": cfg.get("mode", "v2x_image"),
        "causal_ego_anchor": bool(cfg.get("causal_ego_anchor", False)),
        "vision_encode_per_view": bool(
            getattr(model, "vision_encode_per_view", False)
        ),
        "fusion_mode": cfg.get("fusion_mode"),
        "head_pooling": getattr(model, "head_pooling", cfg.get("head_pooling")),
        "safety_mixer": cfg.get("safety_mixer"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_version": torch.__version__,
        "cuda": cuda_device_info(),
        "model_dtype": "bf16" if bool(cfg.get("bf16", True)) and torch.cuda.is_available() else "fp32_or_config_default",
        "model_forward_ms_per_sample": value_distribution(profile_model_ms),
        "device_model_return_ms_per_sample": value_distribution(
            profile_device_model_return_ms
        ),
        "device_model_postprocess_ms_per_sample": value_distribution(profile_device_model_postprocess_ms),
        "end_to_end_ms_per_sample": value_distribution(profile_end_to_end_ms),
        "to_device_ms_per_sample": value_distribution(profile_to_device_ms),
        "postprocess_ms_per_sample": value_distribution(profile_postprocess_ms),
        "data_loading_and_collate_ms_per_sample": value_distribution(profile_data_ms),
        "throughput_samples_per_sec_model_only": (
            profile_measured / measured_model_sec if measured_model_sec > 0 else None
        ),
        "throughput_samples_per_sec_common": (
            profile_measured / measured_device_model_return_sec
            if measured_device_model_return_sec > 0
            else None
        ),
        "throughput_samples_per_sec_device_model_postprocess": (
            profile_measured / measured_device_model_postprocess_sec
            if measured_device_model_postprocess_sec > 0
            else None
        ),
        "throughput_samples_per_sec_end_to_end": (
            profile_measured / measured_e2e_sec if measured_e2e_sec > 0 else None
        ),
        "total_profile_wall_time_sec": (time.perf_counter() - profile_wall_start) if profile_enabled else None,
        "cuda_peak_memory_allocated_bytes": cuda_peak_alloc,
        "cuda_peak_memory_allocated_gib": bytes_to_gib(cuda_peak_alloc),
        "cuda_peak_memory_reserved_bytes": cuda_peak_reserved,
        "cuda_peak_memory_reserved_gib": bytes_to_gib(cuda_peak_reserved),
        "cuda_memory_allocated_after_bytes": cuda_alloc_after,
        "cuda_memory_allocated_after_gib": bytes_to_gib(cuda_alloc_after),
        "cuda_memory_reserved_after_bytes": cuda_reserved_after,
        "cuda_memory_reserved_after_gib": bytes_to_gib(cuda_reserved_after),
        "nvidia_smi_before": profile_smi_before,
        "nvidia_smi_after": profile_smi_after,
        "cpu_rss_before_bytes": profile_start_rss,
        "cpu_rss_before_gib": bytes_to_gib(profile_start_rss),
        "cpu_rss_after_bytes": profile_end_rss,
        "cpu_rss_after_gib": bytes_to_gib(profile_end_rss),
        "cpu_rss_source": profile_rss_source,
        "parameter_counts": parameter_profile,
        "checkpoint_size": checkpoint_profile,
        "input_provenance": profile_input_provenance,
        "input_stats": {name: value_distribution(values) for name, values in sorted(profile_input_stats.items())},
        "input_tensor_shapes": profile_batch_shapes,
        "flops_status": "not_measured_dynamic_multimodal_model",
    }
    metrics = {
        "count": count,
        "split": split,
        "checkpoint_dir": checkpoint_dir,
        "model_loaded_from": model_path,
        "loaded_checkpoint_backbone": loaded_checkpoint_backbone,
        "config_path": config_path,
        "examples_output": examples_output,
        "structured_output": structured_output,
        "examples_every_n": examples_every_n,
        "examples_written": examples_written,
        "examples_global_indices": examples_indices,
        "run_metadata": {
            "stage": "eval",
            "created_at_utc": utc_now(),
            "cot_diagnostic_format": resolved_cot_format,
            "language_contract": (
                THREE_PART_CONTRACT if three_part_diagnostic else "legacy_four_part"
            ),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "max_batches": max_batches,
            "eval_batch_size": int(cfg.get("eval_batch_size", 1)),
            "eval_max_new_tokens": int(cfg.get("eval_max_new_tokens", 256)),
            "eval_generate_all_text": bool(generate_all_text),
            "text_stats_scope": "all_samples" if generate_all_text else "sparse_examples",
            "action_first_report": bool(action_first_report),
            "image_max_pixels": cfg.get("image_max_pixels"),
            "ego_image_max_pixels": cfg.get("ego_image_max_pixels"),
            "infra_image_max_pixels": cfg.get("infra_image_max_pixels"),
        },
        "waypoint_ADE_each_step": [float(x / denom) for x in ade_sum],
        "final_displacement_error": fde_sum / denom,
        "FDE_median": percentile(fde_values, 50),
        "FDE_p90": percentile(fde_values, 90),
        "FDE_p95": percentile(fde_values, 95),
        "FDE_p99": percentile(fde_values, 99),
        "FDE_std": statistics.pstdev(fde_values) if len(fde_values) > 1 else 0.0,
        "ADE_median": percentile(ade_values, 50),
        "ADE_p90": percentile(ade_values, 90),
        "ADE_p95": percentile(ade_values, 95),
        "ADE_p99": percentile(ade_values, 99),
        "ADE_std": statistics.pstdev(ade_values) if len(ade_values) > 1 else 0.0,
        "per_step_error_distribution": per_step_error_distribution,
        "base_vs_final_improvement": base_improvement_stats,
        "command_accuracy": cmd_correct / denom,
        "macro_command_accuracy_present_classes": (
            sum(present_command_acc) / max(len(present_command_acc), 1)
        ),
        "macro_command_accuracy_all_classes": sum(all_command_acc) / max(len(all_command_acc), 1),
        "command_labels": COMMANDS,
        "command_confusion_matrix": confusion.tolist(),
        "command_prediction_distribution": pred_command_counts,
        "command_calibration": calibration_stats,
        "per_command": per_command_metrics,
        "bad_cases_topk": sorted(bad_cases, key=lambda x: x["fde"], reverse=True)[:topk_bad_cases],
        "generated_text_parse_failure_examples": [
            {"index": i, "text_head": text[:500]}
            for i, text in enumerate(generated_texts)
            if not cot_format_parse_ok(text, resolved_cot_format)
        ][:10],
        "action_first_reporting_enabled": bool(action_first_report),
        "cot_diagnostic_format": resolved_cot_format,
        "residual_stats": residual_stats,
        "base_shape_guard_diagnostics": base_shape_guard_diagnostics,
        "collision_eval": "reserved_interface_not_enabled",
        "text_stats_scope": "all_samples" if generate_all_text else "sparse_examples",
        "compute_profile": compute_profile,
        **text_stats(generated_texts, resolved_cot_format),
    }
    if three_part_diagnostic:
        metrics["three_part_language_diagnostic"] = language_diagnostic
    else:
        assert raw_vs_structured is not None
        metrics.update(
            {
                "raw_vs_structured_agreement": raw_vs_structured,
                "raw_generated_cot_format_parse_rate": raw_vs_structured[
                    "raw_generated_cot_format_parse_rate"
                ],
                "raw_part4_command_parse_rate": raw_vs_structured[
                    "raw_part4_command_parse_rate"
                ],
                "raw_part4_command_vs_structured_agreement_rate": raw_vs_structured[
                    "raw_part4_command_vs_structured_agreement_rate"
                ],
                "raw_part4_waypoint_parse_rate": raw_vs_structured[
                    "raw_part4_waypoint_parse_rate"
                ],
                "action_consistent_part4_valid_rate": raw_vs_structured[
                    "action_consistent_part4_valid_rate"
                ],
            }
        )
    if profile_output:
        write_json(Path(profile_output), compute_profile)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CoVLM baseline command/waypoint metrics.")
    parser.add_argument("--config", required=True, help="YAML config used for training.")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory containing saved action_heads.pt and backbone adapters.")
    parser.add_argument("--split", default="val", help="Index split to evaluate.")
    parser.add_argument("--max-batches", type=int, default=0, help="Optional debug limit.")
    parser.add_argument("--output", default=None, help="Optional metrics JSON path.")
    parser.add_argument("--examples-output", default=None, help="Optional generated examples JSONL path.")
    parser.add_argument("--examples-every-n", type=int, default=50, help="Save one generated CoT example every N global samples.")
    parser.add_argument("--generate-all-text", action="store_true", help="Generate CoT for every evaluated sample instead of sparse examples only.")
    parser.add_argument("--action-first-report", action="store_true", help="Write sparse samples with raw CoT and structured action fields separated.")
    parser.add_argument("--structured-output", default=None, help="Optional JSONL export of structured command/waypoint predictions for every evaluated sample. Does not generate CoT text.")
    parser.add_argument("--profile-eval", action="store_true", help="Record rich latency, memory, input, parameter, and calibration profiling.")
    parser.add_argument("--profile-warmup-samples", type=int, default=20, help="Warmup samples before timing; metrics still use all evaluated rows.")
    parser.add_argument("--profile-max-measured-samples", type=int, default=0, help="Measured profiling samples; 0 means all rows after warmup.")
    parser.add_argument("--profile-output", default=None, help="Optional separate JSON path for compute_profile only.")
    parser.add_argument("--profile-no-cot", action=argparse.BooleanOptionalAction, default=True, help="Disable CoT text generation while profiling unless --generate-all-text is explicit.")
    parser.add_argument(
        "--cot-diagnostic-format",
        choices=("legacy_four_part", "three_part"),
        default=None,
        help="Language-output contract used for parsing diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output) if args.output else Path(args.checkpoint_dir) / f"eval_{args.split}_metrics.json"
    if args.examples_output is not None:
        examples_output = args.examples_output
    elif args.structured_output and not args.generate_all_text:
        examples_output = None
    else:
        examples_output = str(output.with_name(f"{output.stem}_cot_every{max(1, args.examples_every_n)}.jsonl"))
    metrics = evaluate(
        args.config,
        args.checkpoint_dir,
        split=args.split,
        max_batches=args.max_batches,
        examples_output=examples_output,
        examples_every_n=args.examples_every_n,
        generate_all_text=args.generate_all_text,
        action_first_report=args.action_first_report,
        structured_output=args.structured_output,
        profile_eval=args.profile_eval,
        profile_warmup_samples=args.profile_warmup_samples,
        profile_max_measured_samples=args.profile_max_measured_samples,
        profile_output=args.profile_output,
        profile_no_cot=args.profile_no_cot,
        cot_diagnostic_format=args.cot_diagnostic_format,
    )
    write_json(output, metrics)
    cot_msg = f"; CoT samples saved to {examples_output}" if examples_output else "; CoT samples disabled"
    print(f"[DONE] Eval metrics saved to {output}{cot_msg}; examples_written={metrics['examples_written']}")


if __name__ == "__main__":
    main()
