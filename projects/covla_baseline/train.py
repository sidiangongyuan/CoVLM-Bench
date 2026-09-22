#!/usr/bin/env python3
"""Train the single-stage CoVLM Qwen-VL baseline.

Smoke example:
    python -m projects.covla_baseline.smoke_train \
      --config projects/covla_baseline/configs/qwen3_vl_8b_dual_image_lora.yaml
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from projects.covla_baseline.data.build_index import DEFAULT_V2X_ROOT, build_index
from projects.covla_baseline.data.collator import CoVLACollator
from projects.covla_baseline.data.dataset import (
    COMMAND2ID,
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

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is expected in training envs.
    np = None


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cfg_int(cfg: Dict[str, Any], key: str, default: int) -> int:
    """Read integer config values while preserving explicit 0 as a real setting."""
    value = cfg.get(key, None)
    return int(default if value is None else value)


def validate_causal_anchor_config(
    cfg: Dict[str, Any],
    *,
    resolved_head_pooling: Optional[str] = None,
) -> None:
    causal_anchor = bool(cfg.get("causal_ego_anchor", False))
    head_pooling = str(
        resolved_head_pooling
        if resolved_head_pooling is not None
        else cfg.get("head_pooling", "last")
    ).lower()
    uses_anchor_pooling = head_pooling == "causal_ego_anchor_mean"
    uses_matched_full_sequence_pooling = head_pooling == "mean"
    if uses_anchor_pooling and not causal_anchor:
        raise ValueError(
            "head_pooling=causal_ego_anchor_mean requires causal_ego_anchor"
        )
    if causal_anchor and not (
        uses_anchor_pooling or uses_matched_full_sequence_pooling
    ):
        raise ValueError(
            "causal_ego_anchor requires head_pooling=causal_ego_anchor_mean or "
            "the matched full-sequence ablation head_pooling=mean"
        )
    if causal_anchor and (
        str(cfg.get("mode", "v2x_image")) != "v2x_image"
        or str(cfg.get("view_order", "ego_infra")).lower() != "ego_infra"
    ):
        raise ValueError(
            "causal_ego_anchor requires mode=v2x_image and view_order=ego_infra"
        )
    if bool(cfg.get("vision_encode_per_view", False)) and not causal_anchor:
        raise ValueError(
            "vision_encode_per_view is currently restricted to causal_ego_anchor"
        )


def import_processor(model_name_or_path: str) -> Any:
    try:
        from transformers import AutoProcessor  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "transformers is required to load the Qwen processor. Install a Qwen-VL compatible transformers build. "
            f"Original error: {exc}"
        )
    return AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)


def configure_seed(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Enable deterministic seed control only when the config explicitly asks for it."""
    if cfg.get("seed", None) is None:
        return None
    seed = int(cfg["seed"])
    deterministic = bool(cfg.get("deterministic", False))
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "seed": seed,
        "deterministic": deterministic,
        "numpy_seeded": np is not None,
        "torch_cuda_seeded": torch.cuda.is_available(),
    }


def ensure_index(cfg: Dict[str, Any]) -> str:
    index_path = Path(cfg.get("index_path", "output/covla_baseline/index_l4_full_fixed.jsonl"))
    if index_path.exists():
        return str(index_path)
    if bool(cfg.get("index_must_exist", False)):
        raise FileNotFoundError(
            f"Required prebuilt index is missing: {index_path}. "
            "Refusing to rebuild it with an implicit target contract."
        )
    l4_dir = Path(cfg.get("l4_dir", "data/covlm_bench/l4_cot_v3_full_latest"))
    print(f"[INFO] Index not found; building {index_path} from {l4_dir}")
    build_index(
        l4_dir=l4_dir,
        output=index_path,
        v2x_root=Path(cfg.get("v2x_root", DEFAULT_V2X_ROOT)),
        limit=cfg.get("index_limit"),
        target_parts=int(cfg.get("index_target_parts", 4)),
    )
    return str(index_path)


def memory_report() -> str:
    if not torch.cuda.is_available():
        return "cuda=not_available"
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    return f"gpu_alloc={alloc:.2f}GB gpu_reserved={reserved:.2f}GB"


def command_distribution(dataset: CoVLABaselineDataset) -> Dict[str, Any]:
    counts = {name: 0 for name in COMMANDS}
    for item in dataset.items:
        cmd = canonical_command(item.get("command", "UNKNOWN"))
        counts[cmd] = counts.get(cmd, 0) + 1
    total = sum(counts.values())
    return {
        "total": total,
        "counts": counts,
        "ratios": {cmd: (count / max(total, 1)) for cmd, count in counts.items()},
    }


def resolve_command_class_weights(
    cfg: Dict[str, Any],
    dataset: CoVLABaselineDataset,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    dist_payload = command_distribution(dataset)
    method = str(cfg.get("command_class_weighting", "none") or "none").lower()
    reduction = str(cfg.get("command_class_weight_reduction", "mean") or "mean").lower()
    if reduction not in {"mean", "fixed_normalizer"}:
        raise ValueError(
            "command_class_weight_reduction must be 'mean' or 'fixed_normalizer', "
            f"got {reduction!r}"
        )
    cap = float(cfg.get("command_class_weight_cap", 5.0))
    manual = cfg.get("command_class_weights")

    def add_normalizer(payload: Dict[str, Any], weights: Optional[torch.Tensor]) -> Dict[str, Any]:
        normalizer = 1.0
        if weights is not None and reduction == "fixed_normalizer":
            counts = torch.tensor(
                [dist_payload["counts"].get(cmd, 0) for cmd in COMMANDS],
                dtype=torch.float32,
            )
            total = counts.sum().clamp_min(1.0)
            normalizer = float((weights.detach().float() * counts).sum().item() / float(total))
        payload.update(
            {
                "weight_reduction": reduction,
                "weight_normalizer": float(normalizer),
                "normalizer_source": (
                    "train_distribution_expected_weight"
                    if weights is not None and reduction == "fixed_normalizer"
                    else "none"
                ),
            }
        )
        return payload

    if manual is not None:
        weights = torch.tensor([float(x) for x in manual], dtype=torch.float32)
        if weights.numel() != len(COMMANDS):
            raise ValueError(f"command_class_weights must have {len(COMMANDS)} values.")
        payload = {
            **dist_payload,
            "weighting": "manual",
            "weights": dict(zip(COMMANDS, weights.tolist())),
        }
        payload = add_normalizer(payload, weights)
        return weights, payload

    if method in {"none", "false", "0", ""}:
        payload = add_normalizer({**dist_payload, "weighting": "none", "weights": None}, None)
        return None, payload
    if method not in {
        "auto_sqrt_inverse_freq",
        "sqrt_inverse_freq",
        "auto_mild_inverse_freq",
        "mild_inverse_freq",
    }:
        raise ValueError(f"Unsupported command_class_weighting={method!r}")

    counts = torch.tensor([dist_payload["counts"].get(cmd, 0) for cmd in COMMANDS], dtype=torch.float32)
    present = counts > 0
    weights = torch.zeros_like(counts)
    alpha = 0.5
    blend = 1.0
    if method in {"auto_mild_inverse_freq", "mild_inverse_freq"}:
        alpha = float(cfg.get("command_class_weight_alpha", 0.25))
        blend = float(cfg.get("command_class_weight_blend", 1.0))
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("command_class_weight_alpha must be in [0, 1].")
        if not (0.0 <= blend <= 1.0):
            raise ValueError("command_class_weight_blend must be in [0, 1].")
    if bool(present.any()):
        present_counts = counts[present]
        raw = torch.pow(present_counts.mean() / present_counts, alpha)
        raw = raw / raw.mean().clamp_min(1e-12)
        raw = 1.0 + blend * (raw - 1.0)
        weights[present] = torch.clamp(raw, max=cap)
    payload = {
        **dist_payload,
        "weighting": method,
        "weight_cap": cap,
        "weight_alpha": alpha,
        "weight_blend": blend,
        "weights": dict(zip(COMMANDS, weights.tolist())),
    }
    payload = add_normalizer(payload, weights)
    return weights, payload


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
    mode = str(cfg.get("fusion_mode") or "").lower()
    if mode == "evidence_directed_residual":
        return "route_conditioned"
    return "l1"


def resolve_evidence_feature_dim(cfg: Dict[str, Any]) -> int:
    if str(cfg.get("fusion_mode") or "").lower() == "evidence_directed_residual":
        return int(cfg.get("evidence_feature_dim", ROUTE_EVIDENCE_FEATURE_DIM))
    return int(cfg.get("evidence_feature_dim", L1_EVIDENCE_FEATURE_DIM))


def build_residual_balanced_sampler(
    cfg: Dict[str, Any],
    dataset: CoVLABaselineDataset,
) -> Tuple[Optional[WeightedRandomSampler], Dict[str, Any]]:
    mode = str(cfg.get("balanced_sampler", "none") or "none").lower()
    if mode in {"none", "false", "0", ""}:
        return None, {"enabled": False, "mode": mode}
    if mode not in {"evidence_directed", "residual_v2x"}:
        raise ValueError(f"Unsupported balanced_sampler={mode!r}")
    weights: List[float] = []
    command_counts = defaultdict(int)
    for item in dataset.items:
        command_counts[canonical_command(item.get("command", "UNKNOWN"))] += 1
    for item in dataset.items:
        command = canonical_command(item.get("command", "UNKNOWN"))
        command_weight = math.sqrt(len(dataset.items) / max(command_counts[command], 1))
        base = dataset._lookup_base_prediction(item)
        base_error_weight = 1.0
        if isinstance(base, dict):
            try:
                fde = float(base.get("base_fde", 0.0))
            except (TypeError, ValueError):
                fde = 0.0
            base_error_weight += min(max(fde - 3.0, 0.0) / 5.0, 2.0)
        route_weight = 1.0
        if item.get("infra_only_critical_ids"):
            route_weight += 0.25
        if command == "GO_STRAIGHT":
            command_weight *= float(cfg.get("balanced_sampler_go_straight_scale", 0.75))
        weights.append(max(0.05, command_weight * base_error_weight * route_weight))
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )
    return sampler, {
        "enabled": True,
        "mode": mode,
        "weight_min": min(weights) if weights else None,
        "weight_max": max(weights) if weights else None,
        "weight_mean": sum(weights) / max(len(weights), 1),
        "command_counts": dict(command_counts),
    }


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def distributed_context() -> Dict[str, Any]:
    """Initialize native torchrun DDP when WORLD_SIZE > 1."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA devices.")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
    return {
        "distributed": distributed,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "is_main": rank == 0,
    }


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_optimizer_steps: int,
    warmup_ratio: float,
    lr_scheduler_type: str,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_optimizer_steps = max(1, int(total_optimizer_steps))
    warmup_steps = int(total_optimizer_steps * max(0.0, float(warmup_ratio)))
    scheduler_name = str(lr_scheduler_type or "constant").lower()

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        if scheduler_name == "cosine":
            progress = (current_step - warmup_steps) / max(1, total_optimizer_steps - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def checkpoint_backbone_path(checkpoint_dir: str) -> str:
    backbone = Path(checkpoint_dir) / "backbone"
    return str(backbone if backbone.exists() else Path(checkpoint_dir))


def load_heads_if_available(model: QwenVLABaseline, checkpoint_dir: str) -> bool:
    path = Path(checkpoint_dir) / "action_heads.pt"
    if not path.exists():
        return False
    state = torch.load(path, map_location="cpu")
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


def trainable_parameter_summary(model: torch.nn.Module) -> Dict[str, Any]:
    groups = {
        "backbone": {"total": 0, "trainable": 0},
        "action_head": {"total": 0, "trainable": 0},
        "command_head": {"total": 0, "trainable": 0},
        "shape_command_head": {"total": 0, "trainable": 0},
        "other_heads": {"total": 0, "trainable": 0},
    }
    other_head_prefixes = (
        "residual_gate_head.",
        "geov2x_residual_head.",
        "predicted_evidence_residual_head.",
        "predicted_evidence_token_residual_head.",
        "evidence_residual_head.",
        "evidence_directed_residual_head.",
        "task_query_pooler.",
    )
    total = 0
    trainable = 0
    for name, param in model.named_parameters():
        count = int(param.numel())
        total += count
        if param.requires_grad:
            trainable += count
        if name.startswith("backbone."):
            group = "backbone"
        elif name.startswith("action_head."):
            group = "action_head"
        elif name.startswith("command_head."):
            group = "command_head"
        elif name.startswith("shape_command_head."):
            group = "shape_command_head"
        elif name.startswith(other_head_prefixes):
            group = "other_heads"
        else:
            group = "other_heads"
        groups[group]["total"] += count
        if param.requires_grad:
            groups[group]["trainable"] += count
    for payload in groups.values():
        payload["frozen"] = int(payload["total"] - payload["trainable"])
        payload["trainable_ratio"] = payload["trainable"] / max(payload["total"], 1)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "trainable_parameter_ratio": trainable / max(total, 1),
        "module_groups": groups,
    }


def balanced_checkpoint_gates_from_config(cfg: Dict[str, Any]) -> Dict[str, float]:
    return {
        "fde_max": float(cfg.get("balanced_checkpoint_fde_max", 5.14295)),
        "cmd_pass": float(cfg.get("balanced_checkpoint_cmd_pass", 0.76064)),
        "macro_pass": float(cfg.get("balanced_checkpoint_macro_pass", 0.56452)),
        "cmd_strong": float(cfg.get("balanced_checkpoint_cmd_strong", 0.76564)),
        "macro_strong": float(cfg.get("balanced_checkpoint_macro_strong", 0.56952)),
    }


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def nonzero_prediction_classes(metrics: Dict[str, Any]) -> List[str]:
    distribution = metrics.get("command_prediction_distribution", {}) or {}
    nonzero: List[str] = []
    for cmd in COMMANDS:
        try:
            count = int(distribution.get(cmd, 0))
        except (TypeError, ValueError):
            count = 0
        if count > 0:
            nonzero.append(cmd)
    return nonzero


def metric_float(metrics: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(metrics.get(key, default))
    except (TypeError, ValueError):
        value = default
    return value


def best_checkpoint_record(
    *,
    epoch: int,
    checkpoint_dir: Path,
    metrics_path: Path,
    metrics: Dict[str, Any],
    selected_from_non_single_class: bool,
    selection_reason: str,
) -> Dict[str, Any]:
    fde = metric_float(metrics, "final_displacement_error", float("inf"))
    macro = metric_float(metrics, "macro_command_accuracy_present_classes", 0.0)
    cmd = metric_float(metrics, "command_accuracy", 0.0)
    return {
        "epoch": int(epoch),
        "checkpoint_dir": str(checkpoint_dir),
        "metrics_path": str(metrics_path),
        "selected_from_non_single_class": bool(selected_from_non_single_class),
        "selection_reason": selection_reason,
        "final_displacement_error": fde,
        "macro_command_accuracy_present_classes": macro,
        "command_accuracy": cmd,
        "nonzero_predicted_classes": list(metrics.get("nonzero_predicted_classes", [])),
        "single_class_prediction": bool(metrics.get("single_class_prediction", False)),
        "updated_at_utc": utc_now(),
    }


def annotate_balanced_checkpoint_candidate(candidate: Dict[str, Any], gates: Dict[str, float]) -> None:
    fde = float(candidate.get("final_displacement_error", float("inf")))
    macro = float(candidate.get("macro_command_accuracy_present_classes", 0.0))
    cmd = float(candidate.get("command_accuracy", 0.0))
    non_single = bool(candidate.get("selected_from_non_single_class", False))
    gate_status = {
        "non_single_class": non_single,
        "fde_within_max": fde <= float(gates["fde_max"]),
        "cmd_pass": cmd >= float(gates["cmd_pass"]),
        "macro_pass": macro >= float(gates["macro_pass"]),
        "cmd_strong": cmd >= float(gates["cmd_strong"]),
        "macro_strong": macro >= float(gates["macro_strong"]),
    }
    candidate["balanced_checkpoint_gates"] = dict(gates)
    candidate["balanced_checkpoint_gate_status"] = gate_status
    candidate["balanced_checkpoint_pass"] = bool(
        gate_status["non_single_class"]
        and gate_status["fde_within_max"]
        and gate_status["cmd_pass"]
        and gate_status["macro_pass"]
    )


def is_better_checkpoint(candidate: Dict[str, Any], current: Optional[Dict[str, Any]]) -> bool:
    if current is None:
        return True
    candidate_is_eligible = bool(candidate.get("selected_from_non_single_class", False))
    current_is_eligible = bool(current.get("selected_from_non_single_class", False))
    if candidate_is_eligible != current_is_eligible:
        return candidate_is_eligible
    cand_fde = float(candidate.get("final_displacement_error", float("inf")))
    cur_fde = float(current.get("final_displacement_error", float("inf")))
    if cand_fde < cur_fde - 1e-12:
        return True
    if abs(cand_fde - cur_fde) <= 1e-12:
        cand_macro = float(candidate.get("macro_command_accuracy_present_classes", 0.0))
        cur_macro = float(current.get("macro_command_accuracy_present_classes", 0.0))
        return cand_macro > cur_macro
    return False


def is_better_balanced_checkpoint(candidate: Dict[str, Any], current: Optional[Dict[str, Any]]) -> bool:
    if current is None:
        return True
    candidate_pass = bool(candidate.get("balanced_checkpoint_pass", False))
    current_pass = bool(current.get("balanced_checkpoint_pass", False))
    if candidate_pass != current_pass:
        return candidate_pass
    if candidate_pass and current_pass:
        cand_fde = float(candidate.get("final_displacement_error", float("inf")))
        cur_fde = float(current.get("final_displacement_error", float("inf")))
        if cand_fde < cur_fde - 1e-12:
            return True
        if abs(cand_fde - cur_fde) <= 1e-12:
            cand_macro = float(candidate.get("macro_command_accuracy_present_classes", 0.0))
            cur_macro = float(current.get("macro_command_accuracy_present_classes", 0.0))
            if cand_macro > cur_macro + 1e-12:
                return True
            if abs(cand_macro - cur_macro) <= 1e-12:
                cand_cmd = float(candidate.get("command_accuracy", 0.0))
                cur_cmd = float(current.get("command_accuracy", 0.0))
                return cand_cmd > cur_cmd
        return False
    return is_better_checkpoint(candidate, current)


def publish_best_checkpoint(source_dir: Path, stable_dir: Path) -> None:
    if stable_dir.exists():
        shutil.rmtree(stable_dir)
    shutil.copytree(source_dir, stable_dir)


@torch.no_grad()
def evaluate_current_model(
    model: torch.nn.Module,
    processor: Any,
    cfg: Dict[str, Any],
    index_path: str,
    split: str,
    device: torch.device,
    checkpoint_dir: str,
    max_batches: int = 0,
) -> Dict[str, Any]:
    """Evaluate the in-memory model at epoch end without loading a second Qwen copy."""
    model_to_eval = unwrap_model(model)
    was_training = model_to_eval.training
    model_to_eval.eval()
    dataset = CoVLABaselineDataset(
        index_path,
        mode=cfg.get("mode", "v2x_image"),
        split=split,
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
        v2x_root=str(cfg.get("v2x_root", DEFAULT_V2X_ROOT)),
        infra_condition=str(cfg.get("infra_condition", "normal")),
        infra_condition_mix=None,
        shuffled_infra_map_path=cfg.get("shuffled_infra_map_path"),
        use_l1_evidence_tokens=str(cfg.get("fusion_mode") or "").lower() in evidence_fusion_modes(),
        evidence_top_k=int(cfg.get("route_evidence_top_k", cfg.get("evidence_top_k", 12))),
        evidence_mode=dataset_evidence_mode(cfg),
        route_corridor_width_m=float(cfg.get("route_corridor_width_m", 6.0)),
        route_near_horizon_steps=int(cfg.get("route_near_horizon_steps", 6)),
        route_relevance_variant=str(cfg.get("route_relevance_variant", "strict")),
        # Teacher targets are a training-only regularizer. In particular, a
        # train-split prediction file should never become a validation input.
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
        processor=processor,
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
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("eval_batch_size", 1)),
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    ade_sum = torch.zeros(6)
    fde_sum = 0.0
    cmd_correct = 0
    count = 0
    confusion = torch.zeros((len(COMMANDS), len(COMMANDS)), dtype=torch.long)
    pred_command_counts = {cmd: 0 for cmd in COMMANDS}
    by_command: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "cmd_correct": 0, "ade_sum": 0.0, "fde_sum": 0.0})
    gate_values: List[float] = []
    delta_norms: List[float] = []
    alpha_values: List[float] = []
    guard_groups: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "cmd_correct": 0, "ade_sum": 0.0, "fde_sum": 0.0}
    )

    for bi, batch in enumerate(loader):
        batch = to_device(batch, device)
        out = model_to_eval(**batch)
        pred_wp = out["pred_waypoints"].detach().cpu()
        gt_wp = batch["waypoints"].detach().cpu()
        dist = torch.norm(pred_wp - gt_wp, dim=-1)
        ade_sum += dist.sum(dim=0)
        fde_sum += float(dist[:, -1].sum())
        pred_cmd = out["command_logits"].argmax(dim=-1).detach().cpu()
        gt_cmd = batch["command_id"].detach().cpu()
        alpha_cpu = None
        guard_cpu = None
        if "alpha_shape" in out:
            alpha_cpu = out["alpha_shape"].detach().float().cpu().view(-1)
            alpha_values.extend([float(x) for x in alpha_cpu])
        if "base_shape_guard_applied" in out:
            guard_cpu = out["base_shape_guard_applied"].detach().bool().cpu().view(-1)
        if "gate" in out:
            gate_values.extend([float(x) for x in out["gate"].detach().float().cpu().view(-1)])
        if "delta_waypoints" in out:
            norms = out["delta_waypoints"].detach().float().cpu().norm(dim=-1).mean(dim=1)
            delta_norms.extend([float(x) for x in norms])
        cmd_correct += int((pred_cmd == gt_cmd).sum())
        for row in range(int(gt_cmd.shape[0])):
            gt_cmd_id = int(gt_cmd[row])
            pred_cmd_id = int(pred_cmd[row])
            if 0 <= gt_cmd_id < len(COMMANDS) and 0 <= pred_cmd_id < len(COMMANDS):
                confusion[gt_cmd_id, pred_cmd_id] += 1
                pred_command_counts[ID2COMMAND.get(pred_cmd_id, "UNKNOWN")] += 1
            cmd_name = ID2COMMAND.get(gt_cmd_id, "UNKNOWN")
            rec = by_command[cmd_name]
            rec["count"] += 1
            rec["cmd_correct"] += int(pred_cmd_id == gt_cmd_id)
            rec["ade_sum"] += float(dist[row].mean())
            rec["fde_sum"] += float(dist[row, -1])
            if guard_cpu is not None:
                group = "guarded" if bool(guard_cpu[row]) else "unguarded"
                guard_rec = guard_groups[group]
                guard_rec["count"] += 1
                guard_rec["cmd_correct"] += int(pred_cmd_id == gt_cmd_id)
                guard_rec["ade_sum"] += float(dist[row].mean())
                guard_rec["fde_sum"] += float(dist[row, -1])
        count += int(gt_wp.shape[0])
        if max_batches and bi + 1 >= max_batches:
            break

    if was_training:
        model_to_eval.train()
    denom = max(count, 1)
    per_command = {
        cmd: {
            "count": rec["count"],
            "command_accuracy": rec["cmd_correct"] / max(rec["count"], 1),
            "ADE": rec["ade_sum"] / max(rec["count"], 1),
            "FDE": rec["fde_sum"] / max(rec["count"], 1),
        }
        for cmd, rec in sorted(by_command.items())
    }
    present_acc = [rec["command_accuracy"] for rec in per_command.values() if rec["count"] > 0]
    all_acc = [per_command.get(cmd, {"command_accuracy": 0.0})["command_accuracy"] for cmd in COMMANDS]
    metrics = {
        "count": count,
        "split": split,
        "checkpoint_dir": checkpoint_dir,
        "loaded_checkpoint_backbone": True,
        "run_metadata": {
            "stage": "epoch_eval",
            "created_at_utc": utc_now(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "max_batches": max_batches,
            "eval_batch_size": int(cfg.get("eval_batch_size", 1)),
            "image_max_pixels": cfg.get("image_max_pixels"),
        },
        "waypoint_ADE_each_step": [float(x / denom) for x in ade_sum],
        "final_displacement_error": fde_sum / denom,
        "command_accuracy": cmd_correct / denom,
        "macro_command_accuracy_present_classes": sum(present_acc) / max(len(present_acc), 1),
        "macro_command_accuracy_all_classes": sum(all_acc) / max(len(all_acc), 1),
        "command_labels": COMMANDS,
        "command_confusion_matrix": confusion.tolist(),
        "command_prediction_distribution": pred_command_counts,
        "per_command": per_command,
    }
    if gate_values:
        sorted_gate = sorted(gate_values)
        sorted_delta = sorted(delta_norms)
        p90_idx = min(len(sorted_gate) - 1, int(0.9 * (len(sorted_gate) - 1)))
        d90_idx = min(len(sorted_delta) - 1, int(0.9 * (len(sorted_delta) - 1)))
        metrics["residual_stats"] = {
            "gate_mean": sum(gate_values) / len(gate_values),
            "gate_p90": sorted_gate[p90_idx],
            "gate_max": max(gate_values),
            "delta_waypoint_norm_mean": sum(delta_norms) / max(len(delta_norms), 1),
            "delta_waypoint_norm_p90": sorted_delta[d90_idx] if sorted_delta else 0.0,
        }
    if alpha_values:
        metrics["base_shape_guard_diagnostics"] = {
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
    return metrics


def _run_training_impl(
    config_path: str,
    smoke: bool,
    overrides: Optional[Dict[str, Any]],
    ddp: Dict[str, Any],
) -> None:
    cfg = load_config(config_path)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    validate_causal_anchor_config(cfg)
    seed_payload = configure_seed(cfg)
    if smoke:
        cfg["batch_size"] = 1
        cfg["max_steps"] = max(1, min(int(cfg.get("max_steps", 2) or 2), 2))
        cfg["gradient_accumulation_steps"] = 1

    if ddp["distributed"] and cfg.get("device_map"):
        cleanup_distributed()
        raise ValueError("DDP training is incompatible with device_map; set device_map: null.")

    model_name = cfg.get("model_name_or_path")
    if not model_name:
        raise ValueError("model_name_or_path is missing in config")
    if not Path(str(model_name)).exists() and str(model_name).startswith("/"):
        raise FileNotFoundError(
            f"Configured model path does not exist: {model_name}. Update model_name_or_path in {config_path}."
        )

    index_path = ensure_index(cfg)
    processor = import_processor(str(model_name))
    train_set = CoVLABaselineDataset(
        index_path=index_path,
        mode=cfg.get("mode", "v2x_image"),
        split=cfg.get("train_split", "train"),
        load_images=False,
        no_cot=bool(cfg.get("no_cot", False)),
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
        v2x_root=str(cfg.get("v2x_root", DEFAULT_V2X_ROOT)),
        infra_condition=str(cfg.get("infra_condition", "normal")),
        infra_condition_mix=cfg.get("infra_condition_mix"),
        shuffled_infra_map_path=cfg.get("shuffled_infra_map_path"),
        use_l1_evidence_tokens=str(cfg.get("fusion_mode") or "").lower() in evidence_fusion_modes(),
        evidence_top_k=int(cfg.get("route_evidence_top_k", cfg.get("evidence_top_k", 12))),
        evidence_mode=dataset_evidence_mode(cfg),
        route_corridor_width_m=float(cfg.get("route_corridor_width_m", 6.0)),
        route_near_horizon_steps=int(cfg.get("route_near_horizon_steps", 6)),
        route_relevance_variant=str(cfg.get("route_relevance_variant", "strict")),
        teacher_prediction_path=cfg.get("teacher_prediction_path"),
        teacher_exclude_decision_relevant=bool(
            cfg.get("teacher_exclude_decision_relevant", False)
        ),
        route_risk_supervision=bool(cfg.get("route_risk_supervision", False)),
        use_trajectory_grounded_role=bool(
            cfg.get("use_trajectory_grounded_role", False)
        ),
        trajectory_relevant_repeat=int(cfg.get("trajectory_relevant_repeat", 1)),
        view_order=str(cfg.get("view_order", "ego_infra")),
        causal_ego_anchor=bool(cfg.get("causal_ego_anchor", False)),
    )
    train_lm_targets = bool(cfg.get("train_lm_targets", True))
    command_weights, command_payload = resolve_command_class_weights(cfg, train_set)
    collator = CoVLACollator(
        processor=processor,
        mode=cfg.get("mode", "v2x_image"),
        image_max_pixels=cfg.get("image_max_pixels"),
        max_length=cfg.get("max_length"),
        include_targets=train_lm_targets,
        sample_passthrough_keys=cfg.get("sample_passthrough_keys"),
        task_query_text=cfg.get("task_query_text"),
        task_query_tail_tokens=int(cfg.get("task_query_tail_tokens", 0)),
        view_order=str(cfg.get("view_order", "ego_infra")),
        ego_image_max_pixels=cfg.get("ego_image_max_pixels"),
        infra_image_max_pixels=cfg.get("infra_image_max_pixels"),
        causal_ego_anchor=bool(cfg.get("causal_ego_anchor", False)),
    )
    sampler = DistributedSampler(train_set, shuffle=True) if ddp["distributed"] else None
    sampler_payload: Dict[str, Any] = {"enabled": False, "mode": "distributed" if sampler is not None else "none"}
    if sampler is None:
        sampler, sampler_payload = build_residual_balanced_sampler(cfg, train_set)
    loader_generator = None
    if sampler is None and seed_payload is not None:
        loader_generator = torch.Generator()
        loader_generator.manual_seed(int(seed_payload["seed"]))
    loader = DataLoader(
        train_set,
        batch_size=int(cfg.get("batch_size", 1)),
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collator,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=bool(cfg.get("pin_memory", False)),
        generator=loader_generator,
    )

    init_checkpoint_dir = cfg.get("init_checkpoint_dir") or cfg.get("resume_from_checkpoint")
    model_load_path = checkpoint_backbone_path(str(init_checkpoint_dir)) if init_checkpoint_dir else str(model_name)
    model = QwenVLABaseline(
        model_name_or_path=model_load_path,
        lambda_lm=float(cfg.get("lambda_lm", 1.0)),
        lambda_wp=float(cfg.get("lambda_wp", 1.0)),
        lambda_cmd=float(cfg.get("lambda_cmd", 0.2)),
        lambda_fde=float(cfg.get("lambda_fde", 0.0)),
        lambda_residual_l2=float(cfg.get("lambda_residual_l2", 0.0)),
        lambda_gate_l1=float(cfg.get("lambda_gate_l1", 0.0)),
        lambda_gate_prior=float(cfg.get("lambda_gate_prior", 0.0)),
        lambda_direct_residual=float(cfg.get("lambda_direct_residual", 0.0)),
        lambda_gate_benefit=float(cfg.get("lambda_gate_benefit", 0.0)),
        command_class_weights=command_weights,
        command_class_weight_reduction=str(command_payload.get("weight_reduction", "mean")),
        command_class_weight_normalizer=float(command_payload.get("weight_normalizer", 1.0)),
        command_head_mode=str(cfg.get("command_head_mode", "standard")),
        command_shape_hidden_size=int(cfg.get("command_shape_hidden_size", 64)),
        use_lora=bool(cfg.get("use_lora", False)),
        use_qlora=bool(cfg.get("use_qlora", False)),
        lora_r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        bf16=bool(cfg.get("bf16", True)),
        device_map=cfg.get("device_map"),
        head_pooling=str(cfg.get("head_pooling", "last")),
        fusion_mode=cfg.get("fusion_mode"),
        residual_gate_init_bias=float(cfg.get("residual_gate_init_bias", -3.0)),
        evidence_feature_dim=resolve_evidence_feature_dim(cfg),
        evidence_hidden_size=int(cfg.get("evidence_hidden_size", 256)),
        evidence_layers=int(cfg.get("evidence_layers", 2)),
        evidence_heads=int(cfg.get("evidence_heads", 4)),
        evidence_gate_mode=str(cfg.get("evidence_gate_mode", "per_step")),
        safety_mixer=cfg.get("safety_mixer"),
        base_shape_guard_alpha=float(cfg.get("base_shape_guard_alpha", 0.2)),
        base_shape_guard_waypoints_only=bool(cfg.get("base_shape_guard_waypoints_only", True)),
        base_shape_guard_source=str(
            cfg.get("base_shape_guard_source", "frozen_ego_base_waypoints_only")
        ),
        turn_right_guard_alpha=float(cfg.get("turn_right_guard_alpha", 0.05)),
        turn_right_guard_prior_threshold=float(cfg.get("turn_right_guard_prior_threshold", 0.9)),
        turn_right_guard_residual_threshold=float(cfg.get("turn_right_guard_residual_threshold", 0.1)),
        right_turn_shape_guard_alpha=float(cfg.get("right_turn_shape_guard_alpha", 0.4)),
        right_turn_shape_guard_prior_threshold=float(cfg.get("right_turn_shape_guard_prior_threshold", 0.65)),
        lambda_teacher_wp=float(cfg.get("lambda_teacher_wp", 0.0)),
        lambda_teacher_cmd=float(cfg.get("lambda_teacher_cmd", 0.0)),
        geometry_dim=int(cfg.get("geometry_dim", 11)),
        geov2x_hidden_size=int(cfg.get("geov2x_hidden_size", 256)),
        geov2x_heads=int(cfg.get("geov2x_heads", 4)),
        lambda_counterfactual_base=float(cfg.get("lambda_counterfactual_base", 1.0)),
        lambda_counterfactual_gate=float(cfg.get("lambda_counterfactual_gate", 0.1)),
        lambda_counterfactual_cmd=float(cfg.get("lambda_counterfactual_cmd", 0.2)),
        predicted_evidence_dim=int(cfg.get("predicted_evidence_dim", 10)),
        predicted_evidence_hidden_size=int(cfg.get("predicted_evidence_hidden_size", 256)),
        lambda_predicted_evidence=float(cfg.get("lambda_predicted_evidence", 0.0)),
        task_query_dim=int(cfg.get("task_query_dim", 256)),
        task_query_heads=int(cfg.get("task_query_heads", 4)),
        task_query_dropout=float(cfg.get("task_query_dropout", 0.0)),
        lambda_task_query_ego_wp=float(cfg.get("lambda_task_query_ego_wp", 0.0)),
        lambda_task_query_ego_cmd=float(cfg.get("lambda_task_query_ego_cmd", 0.0)),
        lambda_task_query_consistency_wp=float(
            cfg.get("lambda_task_query_consistency_wp", 0.0)
        ),
        lambda_task_query_consistency_cmd=float(
            cfg.get("lambda_task_query_consistency_cmd", 0.0)
        ),
        lambda_task_query_infra_attention=float(
            cfg.get("lambda_task_query_infra_attention", 0.0)
        ),
        vision_encode_per_view=bool(cfg.get("vision_encode_per_view", False)),
    )
    loaded_init_heads = load_heads_if_available(model, str(init_checkpoint_dir)) if init_checkpoint_dir else False
    freeze_backbone = bool(cfg.get("freeze_backbone", False))
    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
    freeze_action_head = bool(cfg.get("freeze_action_head", False))
    if freeze_action_head:
        for param in model.action_head.parameters():
            param.requires_grad = False
    if str(cfg.get("command_head_mode", "standard")).lower() == "waypoint_shape_fusion":
        for param in model.command_head.parameters():
            param.requires_grad = False
    parameter_summary = trainable_parameter_summary(model)
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable parameters remain after applying freeze settings.")
    device = torch.device(f"cuda:{ddp['local_rank']}" if ddp["distributed"] else ("cuda" if torch.cuda.is_available() else "cpu"))
    if not cfg.get("device_map") and not bool(cfg.get("use_qlora", False)):
        model.to(device)
    if ddp["distributed"]:
        model = DistributedDataParallel(
            model,
            device_ids=[ddp["local_rank"]],
            output_device=ddp["local_rank"],
            find_unused_parameters=bool(cfg.get("ddp_find_unused_parameters", False)),
        )
    model.train()
    frozen_backbone_eval = bool(cfg.get("frozen_backbone_eval", freeze_backbone))
    if freeze_backbone and frozen_backbone_eval:
        unwrap_model(model).backbone.eval()
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(cfg.get("learning_rate", 2e-5)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    configured_max_steps = int(cfg.get("max_steps", 0) or 0)
    num_train_epochs = max(1, int(cfg.get("num_train_epochs", 1)))
    accum = max(1, int(cfg.get("gradient_accumulation_steps", 1)))
    log_every = max(1, int(cfg.get("log_every", 1)))
    total_micro_steps = configured_max_steps if configured_max_steps > 0 else len(loader) * num_train_epochs
    total_optimizer_steps = int(math.ceil(total_micro_steps / float(accum)))
    scheduler = build_scheduler(
        optimizer,
        total_optimizer_steps=total_optimizer_steps,
        warmup_ratio=float(cfg.get("warmup_ratio", 0.0)),
        lr_scheduler_type=str(cfg.get("lr_scheduler_type", "constant")),
    )
    max_grad_norm = float(cfg.get("max_grad_norm", 0.0) or 0.0)
    output_dir = str(cfg.get("output_dir", "output/covla_baseline/debug"))
    output_path = Path(output_dir)
    run_root = output_path.parent
    save_epoch_checkpoints = bool(cfg.get("save_epoch_checkpoints", False))
    eval_at_epoch_end = bool(cfg.get("eval_at_epoch_end", False))
    eval_every_n_epochs = max(1, int(cfg.get("eval_every_n_epochs", 1) or 1))
    save_best_checkpoint = bool(cfg.get("save_best_checkpoint", False))
    balanced_checkpoint_selection = bool(cfg.get("balanced_checkpoint_selection", False))
    balanced_checkpoint_gates = balanced_checkpoint_gates_from_config(cfg)
    epoch_eval_split = str(cfg.get("epoch_eval_split", cfg.get("eval_split", "val")))
    epoch_eval_max_batches = int(cfg.get("epoch_eval_max_batches", 0) or 0)
    early_stop_if_single_class_prediction = bool(cfg.get("early_stop_if_single_class_prediction", False))
    epoch_checkpoint_root = Path(cfg.get("epoch_checkpoint_root", str(run_root / "epoch_checkpoints")))
    best_checkpoint_dir = Path(cfg.get("best_checkpoint_dir", str(run_root / "best_checkpoint")))
    best_checkpoint_json = Path(cfg.get("best_checkpoint_json", str(run_root / "best_checkpoint.json")))
    log_path = output_path / "train_log.jsonl"
    if ddp["is_main"]:
        os.makedirs(output_dir, exist_ok=True)
        write_json(output_path / "command_distribution.json", command_payload)
    run_metadata = {
        "stage": "train",
        "status": "running",
        "started_at_utc": utc_now(),
        "config_path": config_path,
        "config_sha256": file_sha256(Path(config_path).resolve()),
        "output_dir": output_dir,
        "index_path": index_path,
        "index_sha256": file_sha256(Path(index_path).resolve()),
        "model_name_or_path": str(model_name),
        "model_loaded_from": model_load_path,
        "init_checkpoint_dir": str(init_checkpoint_dir) if init_checkpoint_dir else None,
        "loaded_init_heads": bool(loaded_init_heads),
        "mode": cfg.get("mode", "v2x_image"),
        "train_split": cfg.get("train_split", "train"),
        "train_lm_targets": train_lm_targets,
        "freeze_backbone": freeze_backbone,
        "frozen_backbone_eval": frozen_backbone_eval,
        "freeze_action_head": freeze_action_head,
        "command_head_mode": str(cfg.get("command_head_mode", "standard")),
        "command_shape_hidden_size": int(cfg.get("command_shape_hidden_size", 64)),
        "parameter_summary": parameter_summary,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "max_steps": configured_max_steps,
        "resolved_total_micro_steps": total_micro_steps,
        "num_train_epochs": num_train_epochs,
        "batch_size": int(cfg.get("batch_size", 1)),
        "gradient_accumulation_steps": accum,
        "global_batch_size": int(cfg.get("batch_size", 1)) * accum * int(ddp["world_size"]),
        "optimizer_update_steps": total_optimizer_steps,
        "learning_rate": float(cfg.get("learning_rate", 2e-5)),
        "lr_scheduler_type": str(cfg.get("lr_scheduler_type", "constant")),
        "warmup_ratio": float(cfg.get("warmup_ratio", 0.0)),
        "weight_decay": float(cfg.get("weight_decay", 0.0)),
        "max_grad_norm": max_grad_norm,
        "lambda_lm": float(cfg.get("lambda_lm", 1.0)),
        "lambda_wp": float(cfg.get("lambda_wp", 1.0)),
        "lambda_cmd": float(cfg.get("lambda_cmd", 0.2)),
        "lambda_fde": float(cfg.get("lambda_fde", 0.0)),
        "lambda_residual_l2": float(cfg.get("lambda_residual_l2", 0.0)),
        "lambda_gate_l1": float(cfg.get("lambda_gate_l1", 0.0)),
        "lambda_gate_prior": float(cfg.get("lambda_gate_prior", 0.0)),
        "lambda_direct_residual": float(cfg.get("lambda_direct_residual", 0.0)),
        "lambda_gate_benefit": float(cfg.get("lambda_gate_benefit", 0.0)),
        "head_pooling": str(cfg.get("head_pooling", "last")),
        "causal_ego_anchor": bool(cfg.get("causal_ego_anchor", False)),
        "vision_encode_per_view": bool(cfg.get("vision_encode_per_view", False)),
        "view_order": str(cfg.get("view_order", "ego_infra")),
        "no_cot": bool(cfg.get("no_cot", False)),
        "image_max_pixels": cfg.get("image_max_pixels"),
        "ego_image_max_pixels": cfg.get("ego_image_max_pixels"),
        "infra_image_max_pixels": cfg.get("infra_image_max_pixels"),
        "task_query_text": cfg.get("task_query_text"),
        "task_query_tail_tokens": int(cfg.get("task_query_tail_tokens", 0)),
        "task_query_dim": int(cfg.get("task_query_dim", 256)),
        "task_query_heads": int(cfg.get("task_query_heads", 4)),
        "task_query_dropout": float(cfg.get("task_query_dropout", 0.0)),
        "lambda_task_query_ego_wp": float(cfg.get("lambda_task_query_ego_wp", 0.0)),
        "lambda_task_query_ego_cmd": float(cfg.get("lambda_task_query_ego_cmd", 0.0)),
        "lambda_task_query_consistency_wp": float(
            cfg.get("lambda_task_query_consistency_wp", 0.0)
        ),
        "lambda_task_query_consistency_cmd": float(
            cfg.get("lambda_task_query_consistency_cmd", 0.0)
        ),
        "lambda_task_query_infra_attention": float(
            cfg.get("lambda_task_query_infra_attention", 0.0)
        ),
        "fusion_mode": cfg.get("fusion_mode"),
        "base_prediction_path": cfg.get("base_prediction_path"),
        "residual_gate_init_bias": float(cfg.get("residual_gate_init_bias", -3.0)),
        "evidence_top_k": int(cfg.get("route_evidence_top_k", cfg.get("evidence_top_k", 12))),
        "route_evidence_top_k": int(cfg.get("route_evidence_top_k", cfg.get("evidence_top_k", 12))),
        "route_corridor_width_m": float(cfg.get("route_corridor_width_m", 6.0)),
        "route_near_horizon_steps": int(cfg.get("route_near_horizon_steps", 6)),
        "route_relevance_variant": str(cfg.get("route_relevance_variant", "strict")),
        "safety_mixer": cfg.get("safety_mixer"),
        "base_shape_guard_alpha": float(cfg.get("base_shape_guard_alpha", 0.2)),
        "base_shape_guard_waypoints_only": bool(cfg.get("base_shape_guard_waypoints_only", True)),
        "base_shape_guard_source": str(
            cfg.get("base_shape_guard_source", "frozen_ego_base_waypoints_only")
        ),
        "turn_right_guard_alpha": float(cfg.get("turn_right_guard_alpha", 0.05)),
        "turn_right_guard_prior_threshold": float(cfg.get("turn_right_guard_prior_threshold", 0.9)),
        "turn_right_guard_residual_threshold": float(cfg.get("turn_right_guard_residual_threshold", 0.1)),
        "right_turn_shape_guard_alpha": float(cfg.get("right_turn_shape_guard_alpha", 0.4)),
        "right_turn_shape_guard_prior_threshold": float(cfg.get("right_turn_shape_guard_prior_threshold", 0.65)),
        "lambda_teacher_wp": float(cfg.get("lambda_teacher_wp", 0.0)),
        "lambda_teacher_cmd": float(cfg.get("lambda_teacher_cmd", 0.0)),
        "teacher_prediction_path": cfg.get("teacher_prediction_path"),
        "geometry_dim": int(cfg.get("geometry_dim", 11)),
        "geov2x_hidden_size": int(cfg.get("geov2x_hidden_size", 256)),
        "geov2x_heads": int(cfg.get("geov2x_heads", 4)),
        "lambda_counterfactual_base": float(cfg.get("lambda_counterfactual_base", 1.0)),
        "lambda_counterfactual_gate": float(cfg.get("lambda_counterfactual_gate", 0.1)),
        "lambda_counterfactual_cmd": float(cfg.get("lambda_counterfactual_cmd", 0.2)),
        "predicted_evidence_dim": int(cfg.get("predicted_evidence_dim", 10)),
        "predicted_evidence_hidden_size": int(cfg.get("predicted_evidence_hidden_size", 256)),
        "lambda_predicted_evidence": float(cfg.get("lambda_predicted_evidence", 0.0)),
        "infra_condition": str(cfg.get("infra_condition", "normal")),
        "infra_condition_mix": cfg.get("infra_condition_mix"),
        "shuffled_infra_map_path": cfg.get("shuffled_infra_map_path"),
        "use_trajectory_grounded_role": bool(
            cfg.get("use_trajectory_grounded_role", False)
        ),
        "trajectory_relevant_repeat": int(cfg.get("trajectory_relevant_repeat", 1)),
        "lateral_shape_protection": bool(cfg.get("lateral_shape_protection", False)),
        "lateral_shape_gate_damping": bool(cfg.get("lateral_shape_gate_damping", False)),
        "lateral_shape_residual_weight": float(cfg.get("lateral_shape_residual_weight", 0.5)),
        "lateral_shape_detection_source": str(
            cfg.get("lateral_shape_detection_source", "ego_base_waypoints_only")
        ),
        "evidence_feature_dim": resolve_evidence_feature_dim(cfg),
        "evidence_hidden_size": int(cfg.get("evidence_hidden_size", 256)),
        "evidence_layers": int(cfg.get("evidence_layers", 2)),
        "evidence_heads": int(cfg.get("evidence_heads", 4)),
        "evidence_gate_mode": str(cfg.get("evidence_gate_mode", "per_step")),
        "command_class_weighting": command_payload.get("weighting"),
        "command_class_weight_reduction": command_payload.get("weight_reduction"),
        "command_class_weight_normalizer": command_payload.get("weight_normalizer"),
        "command_distribution": str(output_path / "command_distribution.json"),
        "balanced_sampler": sampler_payload,
        "log_every": log_every,
        "ddp_enabled": bool(ddp["distributed"]),
        "rank": int(ddp["rank"]),
        "world_size": int(ddp["world_size"]),
        "device_map": cfg.get("device_map"),
        "save_epoch_checkpoints": save_epoch_checkpoints,
        "eval_at_epoch_end": eval_at_epoch_end,
        "eval_every_n_epochs": eval_every_n_epochs,
        "save_best_checkpoint": save_best_checkpoint,
        "balanced_checkpoint_selection": balanced_checkpoint_selection,
        "balanced_checkpoint_gates": balanced_checkpoint_gates,
        "epoch_eval_split": epoch_eval_split,
        "epoch_eval_max_batches": epoch_eval_max_batches,
        "early_stop_if_single_class_prediction": early_stop_if_single_class_prediction,
        "epoch_checkpoint_root": str(epoch_checkpoint_root),
        "best_checkpoint_dir": str(best_checkpoint_dir),
        "best_checkpoint_json": str(best_checkpoint_json),
        "seed": seed_payload,
    }
    if ddp["is_main"]:
        write_json(output_path / "run_metadata_train.json", run_metadata)

    micro_step = 0
    optimizer_step = 0
    pending_accum = 0
    epoch = 0
    stopped_early = False
    early_stop_info: Dict[str, Any] = {}
    best_info: Optional[Dict[str, Any]] = None
    optimizer.zero_grad(set_to_none=True)
    while micro_step < total_micro_steps:
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        for batch in loader:
            if micro_step >= total_micro_steps:
                break
            batch = to_device(batch, device)
            sync_grad = ((pending_accum + 1) % accum == 0) or (micro_step + 1 >= total_micro_steps)
            sync_context = model.no_sync() if ddp["distributed"] and not sync_grad else contextlib.nullcontext()
            with sync_context:
                out = model(**batch)
                loss = out["total_loss"] / accum
                if not torch.isfinite(loss.detach()):
                    raise FloatingPointError(
                        "Non-finite loss at step={}: total={} lm={} wp={} cmd={} fde={}".format(
                            micro_step,
                            float(out["total_loss"].detach().cpu()),
                            float(out["lm_loss"].detach().cpu()),
                            float(out["waypoint_loss"].detach().cpu()),
                            float(out["command_loss"].detach().cpu()),
                            float(out["fde_loss"].detach().cpu()),
                        )
                    )
                loss.backward()
            pending_accum += 1
            if sync_grad:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                pending_accum = 0
            if ddp["is_main"] and micro_step % log_every == 0:
                log_rec = {
                    "step": micro_step,
                    "epoch": epoch,
                    "optimizer_step": optimizer_step,
                    "lr": float(scheduler.get_last_lr()[0]),
                    "total_loss": float(out["total_loss"].detach().cpu()),
                    "lm_loss": float(out["lm_loss"].detach().cpu()),
                    "waypoint_loss": float(out["waypoint_loss"].detach().cpu()),
                    "command_loss": float(out["command_loss"].detach().cpu()),
                    "fde_loss": float(out["fde_loss"].detach().cpu()),
                    "lambda_lm": float(cfg.get("lambda_lm", 1.0)),
                    "lambda_wp": float(cfg.get("lambda_wp", 1.0)),
                    "lambda_cmd": float(cfg.get("lambda_cmd", 0.2)),
                    "lambda_fde": float(cfg.get("lambda_fde", 0.0)),
                    "memory": memory_report(),
                    "time_utc": utc_now(),
                }
                for key in [
                    "gate_mean",
                    "gate_p90",
                    "gate_max",
                    "delta_waypoint_norm",
                    "residual_l2_loss",
                    "gate_l1_loss",
                    "gate_prior_loss",
                    "direct_residual_loss",
                    "gate_benefit_loss",
                    "teacher_wp_loss",
                    "teacher_cmd_loss",
                    "predicted_evidence_loss",
                    "alpha_shape",
                    "task_query_ego_attention_mass",
                    "task_query_infra_attention_mass",
                    "task_query_ego_wp_loss",
                    "task_query_ego_cmd_loss",
                    "task_query_consistency_wp_loss",
                    "task_query_consistency_cmd_loss",
                    "task_query_infra_attention_loss",
                ]:
                    if key in out:
                        value = out[key].detach().float().cpu()
                        log_rec[key] = float(value.mean())
                if "base_shape_guard_applied" in out:
                    log_rec["base_shape_guard_applied_count"] = int(
                        out["base_shape_guard_applied"].detach().bool().cpu().sum()
                    )
                if "turn_right_guard_applied" in out:
                    log_rec["turn_right_guard_applied_count"] = int(
                        out["turn_right_guard_applied"].detach().bool().cpu().sum()
                    )
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(log_rec, ensure_ascii=False) + "\n")
                print(
                    "[TRAIN] step={step} epoch={epoch} opt={opt} lr={lr:.3e} total={total:.4f} "
                    "lm={lm:.4f} wp={wp:.4f} cmd={cmd:.4f} fde={fde:.4f} {mem}".format(
                        step=log_rec["step"],
                        epoch=log_rec["epoch"],
                        opt=log_rec["optimizer_step"],
                        lr=log_rec["lr"],
                        total=log_rec["total_loss"],
                        lm=log_rec["lm_loss"],
                        wp=log_rec["waypoint_loss"],
                        cmd=log_rec["command_loss"],
                        fde=log_rec["fde_loss"],
                        mem=log_rec["memory"],
                    )
                )
            micro_step += 1
        completed_epoch = epoch + 1
        if ddp["distributed"]:
            dist.barrier()
        is_final_epoch = completed_epoch >= num_train_epochs or micro_step >= total_micro_steps
        should_eval_epoch = bool(eval_at_epoch_end and (completed_epoch % eval_every_n_epochs == 0 or is_final_epoch))
        should_save_epoch = bool(
            save_epoch_checkpoints
            and ((not eval_at_epoch_end) or should_eval_epoch)
        )
        if save_best_checkpoint and should_eval_epoch:
            should_save_epoch = True
        if ddp["is_main"] and (should_save_epoch or should_eval_epoch):
            model_to_save = unwrap_model(model)
            epoch_ckpt_dir = epoch_checkpoint_root / f"epoch_{completed_epoch:03d}"
            model_to_save.save_pretrained(str(epoch_ckpt_dir))
            if should_eval_epoch:
                metrics = evaluate_current_model(
                    model=model,
                    processor=processor,
                    cfg=cfg,
                    index_path=index_path,
                    split=epoch_eval_split,
                    device=device,
                    checkpoint_dir=str(epoch_ckpt_dir),
                    max_batches=epoch_eval_max_batches,
                )
                if freeze_backbone and frozen_backbone_eval:
                    unwrap_model(model).backbone.eval()
                metrics_path = run_root / "metrics" / f"epoch_{completed_epoch:03d}_eval_{epoch_eval_split}.json"
                nonzero_preds = nonzero_prediction_classes(metrics)
                metrics["nonzero_predicted_classes"] = nonzero_preds
                metrics["single_class_prediction"] = bool(metrics.get("count", 0) > 0 and len(nonzero_preds) <= 1)
                metrics["early_stop_triggered"] = bool(
                    early_stop_if_single_class_prediction and metrics["single_class_prediction"]
                )
                metrics["eval_every_n_epochs"] = eval_every_n_epochs
                metrics["is_final_epoch_eval"] = bool(is_final_epoch)
                metrics["best_checkpoint_candidate"] = bool(save_best_checkpoint)
                if metrics["early_stop_triggered"]:
                    stopped_early = True
                    early_stop_info = {
                        "epoch": completed_epoch,
                        "reason": "single_class_prediction",
                        "nonzero_predicted_classes": nonzero_preds,
                        "metrics_path": str(metrics_path),
                    }
                    metrics["early_stop_reason"] = early_stop_info["reason"]
                    print(
                        "[EARLY_STOP] epoch={} reason=single_class_prediction nonzero_preds={}".format(
                            completed_epoch,
                            nonzero_preds,
                        ),
                        flush=True,
                    )
                write_json(metrics_path, metrics)
                if save_best_checkpoint:
                    selected_from_non_single = not bool(metrics["single_class_prediction"])
                    candidate = best_checkpoint_record(
                        epoch=completed_epoch,
                        checkpoint_dir=epoch_ckpt_dir,
                        metrics_path=metrics_path,
                        metrics=metrics,
                        selected_from_non_single_class=selected_from_non_single,
                        selection_reason=(
                            "non_single_lowest_fde_tie_macro"
                            if selected_from_non_single
                            else "provisional_single_class_until_non_single_epoch"
                        ),
                    )
                    if balanced_checkpoint_selection:
                        annotate_balanced_checkpoint_candidate(candidate, balanced_checkpoint_gates)
                        if candidate["balanced_checkpoint_pass"]:
                            candidate["selection_reason"] = "balanced_pass_lowest_fde_tie_macro_cmd"
                        else:
                            candidate["selection_reason"] = (
                                "balanced_gate_failed_fallback_lowest_fde"
                                if selected_from_non_single
                                else "balanced_gate_failed_single_class_fallback"
                            )
                        metrics["balanced_checkpoint_selection"] = True
                        metrics["balanced_checkpoint_gates"] = candidate["balanced_checkpoint_gates"]
                        metrics["balanced_checkpoint_gate_status"] = candidate[
                            "balanced_checkpoint_gate_status"
                        ]
                        metrics["balanced_checkpoint_pass"] = candidate["balanced_checkpoint_pass"]
                        metrics["balanced_checkpoint_selection_reason"] = candidate["selection_reason"]
                        write_json(metrics_path, metrics)
                    is_better = (
                        is_better_balanced_checkpoint(candidate, best_info)
                        if balanced_checkpoint_selection
                        else is_better_checkpoint(candidate, best_info)
                    )
                    if is_better:
                        publish_best_checkpoint(epoch_ckpt_dir, best_checkpoint_dir)
                        candidate["stable_checkpoint_dir"] = str(best_checkpoint_dir)
                        candidate["source_checkpoint_dir"] = str(epoch_ckpt_dir)
                        best_info = candidate
                        write_json(best_checkpoint_json, best_info)
                        metrics["best_checkpoint_updated"] = True
                        metrics["stable_best_checkpoint_dir"] = str(best_checkpoint_dir)
                        write_json(metrics_path, metrics)
                        print(
                            "[BEST] epoch={} fde={:.4f} macro={:.4f} eligible={} checkpoint={}".format(
                                completed_epoch,
                                candidate["final_displacement_error"],
                                candidate["macro_command_accuracy_present_classes"],
                                candidate["selected_from_non_single_class"],
                                best_checkpoint_dir,
                            ),
                            flush=True,
                        )
                print(f"[EPOCH_EVAL] epoch={completed_epoch} metrics={metrics_path}", flush=True)
        if ddp["distributed"]:
            stop_tensor = torch.tensor([1 if stopped_early else 0], device=device, dtype=torch.int)
            dist.broadcast(stop_tensor, src=0)
            stopped_early = bool(stop_tensor.item())
            dist.barrier()
        epoch = completed_epoch
        if stopped_early:
            break
    if ddp["distributed"]:
        dist.barrier()
    if ddp["is_main"]:
        model_to_save = model.module if isinstance(model, DistributedDataParallel) else model
        model_to_save.save_pretrained(output_dir)
        cfg.update(
            {
                "resolved_total_micro_steps": total_micro_steps,
                "resolved_optimizer_update_steps": total_optimizer_steps,
                "ddp_enabled": bool(ddp["distributed"]),
                "world_size": int(ddp["world_size"]),
                "global_batch_size": int(cfg.get("batch_size", 1)) * accum * int(ddp["world_size"]),
                "model_loaded_from": model_load_path,
                "init_checkpoint_dir": str(init_checkpoint_dir) if init_checkpoint_dir else None,
                "loaded_init_heads": bool(loaded_init_heads),
                "freeze_backbone": freeze_backbone,
                "frozen_backbone_eval": frozen_backbone_eval,
                "freeze_action_head": freeze_action_head,
                "command_head_mode": str(cfg.get("command_head_mode", "standard")),
                "command_shape_hidden_size": int(cfg.get("command_shape_hidden_size", 64)),
                "parameter_summary": parameter_summary,
                "fusion_mode": cfg.get("fusion_mode"),
                "base_prediction_path": cfg.get("base_prediction_path"),
                "lambda_lm": float(cfg.get("lambda_lm", 1.0)),
                "lambda_wp": float(cfg.get("lambda_wp", 1.0)),
                "lambda_cmd": float(cfg.get("lambda_cmd", 0.2)),
                "lambda_fde": float(cfg.get("lambda_fde", 0.0)),
                "lambda_residual_l2": float(cfg.get("lambda_residual_l2", 0.0)),
                "lambda_gate_l1": float(cfg.get("lambda_gate_l1", 0.0)),
                "lambda_gate_prior": float(cfg.get("lambda_gate_prior", 0.0)),
                "lambda_direct_residual": float(cfg.get("lambda_direct_residual", 0.0)),
                "lambda_gate_benefit": float(cfg.get("lambda_gate_benefit", 0.0)),
                "residual_gate_init_bias": float(cfg.get("residual_gate_init_bias", -3.0)),
                "evidence_top_k": int(cfg.get("route_evidence_top_k", cfg.get("evidence_top_k", 12))),
                "route_evidence_top_k": int(cfg.get("route_evidence_top_k", cfg.get("evidence_top_k", 12))),
                "route_corridor_width_m": float(cfg.get("route_corridor_width_m", 6.0)),
                "route_near_horizon_steps": int(cfg.get("route_near_horizon_steps", 6)),
                "route_relevance_variant": str(cfg.get("route_relevance_variant", "strict")),
                "safety_mixer": cfg.get("safety_mixer"),
                "base_shape_guard_alpha": float(cfg.get("base_shape_guard_alpha", 0.2)),
                "base_shape_guard_waypoints_only": bool(cfg.get("base_shape_guard_waypoints_only", True)),
                "base_shape_guard_source": str(
                    cfg.get("base_shape_guard_source", "frozen_ego_base_waypoints_only")
                ),
                "turn_right_guard_alpha": float(cfg.get("turn_right_guard_alpha", 0.05)),
                "turn_right_guard_prior_threshold": float(cfg.get("turn_right_guard_prior_threshold", 0.9)),
                "turn_right_guard_residual_threshold": float(cfg.get("turn_right_guard_residual_threshold", 0.1)),
                "right_turn_shape_guard_alpha": float(cfg.get("right_turn_shape_guard_alpha", 0.4)),
                "right_turn_shape_guard_prior_threshold": float(cfg.get("right_turn_shape_guard_prior_threshold", 0.65)),
                "lambda_teacher_wp": float(cfg.get("lambda_teacher_wp", 0.0)),
                "lambda_teacher_cmd": float(cfg.get("lambda_teacher_cmd", 0.0)),
                "teacher_prediction_path": cfg.get("teacher_prediction_path"),
                "geometry_dim": int(cfg.get("geometry_dim", 11)),
                "geov2x_hidden_size": int(cfg.get("geov2x_hidden_size", 256)),
                "geov2x_heads": int(cfg.get("geov2x_heads", 4)),
                "lambda_counterfactual_base": float(cfg.get("lambda_counterfactual_base", 1.0)),
                "lambda_counterfactual_gate": float(cfg.get("lambda_counterfactual_gate", 0.1)),
                "lambda_counterfactual_cmd": float(cfg.get("lambda_counterfactual_cmd", 0.2)),
                "predicted_evidence_dim": int(cfg.get("predicted_evidence_dim", 10)),
                "predicted_evidence_hidden_size": int(cfg.get("predicted_evidence_hidden_size", 256)),
                "lambda_predicted_evidence": float(cfg.get("lambda_predicted_evidence", 0.0)),
                "balanced_checkpoint_selection": balanced_checkpoint_selection,
                "balanced_checkpoint_gates": balanced_checkpoint_gates,
                "infra_condition": str(cfg.get("infra_condition", "normal")),
                "infra_condition_mix": cfg.get("infra_condition_mix"),
                "shuffled_infra_map_path": cfg.get("shuffled_infra_map_path"),
                "evidence_feature_dim": resolve_evidence_feature_dim(cfg),
                "evidence_hidden_size": int(cfg.get("evidence_hidden_size", 256)),
                "evidence_layers": int(cfg.get("evidence_layers", 2)),
                "evidence_heads": int(cfg.get("evidence_heads", 4)),
                "evidence_gate_mode": str(cfg.get("evidence_gate_mode", "per_step")),
                "command_class_weighting": command_payload.get("weighting"),
                "command_class_weights": (
                    command_weights.detach().cpu().tolist() if command_weights is not None else None
                ),
                "command_class_weight_reduction": command_payload.get("weight_reduction"),
                "command_class_weight_normalizer": command_payload.get("weight_normalizer"),
                "seed": seed_payload,
            }
        )
        with open(output_path / "train_config_resolved.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        run_metadata.update(
            {
                "status": "stopped_early" if stopped_early else "completed",
                "finished_at_utc": utc_now(),
                "checkpoint_dir": output_dir,
                "train_log": str(log_path),
                "final_micro_step": micro_step,
                "final_optimizer_step": optimizer_step,
                "early_stop": early_stop_info or None,
                "best_checkpoint": best_info,
            }
        )
        if save_best_checkpoint and best_info is None:
            write_json(
                best_checkpoint_json,
                {
                    "status": "not_available",
                    "reason": "no_epoch_evaluation_completed",
                    "updated_at_utc": utc_now(),
                },
            )
        write_json(output_path / "run_metadata_train.json", run_metadata)
        print(f"[DONE] Saved checkpoint to {output_dir}", flush=True)
    if ddp["distributed"]:
        dist.barrier()
    cleanup_distributed()


def run_training(config_path: str, smoke: bool = False, overrides: Optional[Dict[str, Any]] = None) -> None:
    ddp = distributed_context()
    try:
        _run_training_impl(config_path, smoke, overrides, ddp)
    finally:
        cleanup_distributed()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CoVLM-Bench Qwen-VL baseline. Use --smoke for 1-2 step validation.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--smoke", action="store_true", help="Force batch_size=1 and max_steps<=2.")
    parser.add_argument("--model-name-or-path", default=None, help="Override model_name_or_path from config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(args.config, smoke=args.smoke, overrides={"model_name_or_path": args.model_name_or_path})


if __name__ == "__main__":
    main()
