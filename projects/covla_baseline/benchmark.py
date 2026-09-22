#!/usr/bin/env python3
"""Benchmark CoVLM Qwen-VL dual-image planning baseline latency and memory."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from projects.covla_baseline.data.collator import CoVLACollator
from projects.covla_baseline.data.dataset import (
    DEFAULT_PART1_OBJECT_LIMIT,
    DEFAULT_PART1_SENTENCE_LIMIT,
    CoVLABaselineDataset,
)
from projects.covla_baseline.evaluate import checkpoint_backbone_path, load_heads_if_available, model_loss_kwargs
from projects.covla_baseline.models.qwen_vla import QwenVLABaseline
from projects.covla_baseline.train import (
    cfg_int,
    import_processor,
    load_config,
    to_device,
    validate_causal_anchor_config,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return float(ordered[idx])


def maybe_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def benchmark(config_path: str, checkpoint_dir: str, split: str, warmup_samples: int, measured_samples: int) -> Dict[str, Any]:
    cfg = load_config(config_path)
    model_path, loaded_checkpoint_backbone = checkpoint_backbone_path(checkpoint_dir, cfg)
    processor = import_processor(str(cfg["model_name_or_path"]))
    dataset = CoVLABaselineDataset(
        cfg["index_path"],
        mode=cfg.get("mode", "v2x_image"),
        split=split,
        load_images=False,
        use_ego_status_prompt=bool(cfg.get("use_ego_status_prompt", False)),
        part1_object_limit=cfg_int(cfg, "part1_object_limit", DEFAULT_PART1_OBJECT_LIMIT),
        part1_sentence_limit=cfg_int(cfg, "part1_sentence_limit", DEFAULT_PART1_SENTENCE_LIMIT),
        compact_target_part1=bool(cfg.get("compact_target_part1", False)),
        view_order=str(cfg.get("view_order", "ego_infra")),
        causal_ego_anchor=bool(cfg.get("causal_ego_anchor", False)),
    )
    collator = CoVLACollator(
        processor,
        mode=cfg.get("mode", "v2x_image"),
        image_max_pixels=cfg.get("image_max_pixels"),
        include_targets=False,
        task_query_text=cfg.get("task_query_text"),
        task_query_tail_tokens=int(cfg.get("task_query_tail_tokens", 0)),
        view_order=str(cfg.get("view_order", "ego_infra")),
        ego_image_max_pixels=cfg.get("ego_image_max_pixels"),
        infra_image_max_pixels=cfg.get("infra_image_max_pixels"),
        causal_ego_anchor=bool(cfg.get("causal_ego_anchor", False)),
    )
    loader = DataLoader(dataset, batch_size=int(cfg.get("eval_batch_size", 1)), shuffle=False, collate_fn=collator, num_workers=0)
    loss_kwargs = model_loss_kwargs(cfg, checkpoint_dir)
    validate_causal_anchor_config(
        cfg,
        resolved_head_pooling=str(loss_kwargs["head_pooling"]),
    )
    model = QwenVLABaseline(
        model_name_or_path=model_path,
        **loss_kwargs,
        use_lora=False,
        gradient_checkpointing=False,
        bf16=bool(cfg.get("bf16", True)),
        device_map=cfg.get("device_map"),
    )
    load_heads_if_available(model, checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not cfg.get("device_map"):
        model.to(device)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    latencies: List[float] = []
    seen = 0
    measured = 0
    start_total = time.perf_counter()
    for batch in loader:
        samples = batch["samples"]
        batch = to_device(batch, device)
        maybe_sync()
        t0 = time.perf_counter()
        _ = model(**batch)
        maybe_sync()
        elapsed = time.perf_counter() - t0
        batch_n = len(samples)
        if seen >= warmup_samples and measured < measured_samples:
            latencies.extend([elapsed / max(batch_n, 1)] * batch_n)
            measured += batch_n
            if measured % 50 == 0:
                print(f"[BENCH] measured={measured} avg_ms={statistics.mean(latencies) * 1000:.2f}", flush=True)
        seen += batch_n
        if measured >= measured_samples:
            break
    total_time = time.perf_counter() - start_total
    avg = statistics.mean(latencies) if latencies else 0.0
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    metrics = {
        "split": split,
        "checkpoint_dir": checkpoint_dir,
        "model_loaded_from": model_path,
        "loaded_checkpoint_backbone": loaded_checkpoint_backbone,
        "run_metadata": {
            "stage": "benchmark",
            "created_at_utc": utc_now(),
            "config_path": config_path,
            "mode": cfg.get("mode", "v2x_image"),
        },
        "measured_samples": int(measured),
        "warmup_samples": int(min(seen, warmup_samples)),
        "avg_latency_ms": avg * 1000.0,
        "p50_latency_ms": percentile(latencies, 50) * 1000.0,
        "p95_latency_ms": percentile(latencies, 95) * 1000.0,
        "throughput_samples_per_sec": (measured / sum(latencies)) if latencies else 0.0,
        "total_time_sec": total_time,
        "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0,
        "max_memory_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else 0.0,
        "gpu_name": props.name if props else "cpu",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "batch_size": int(cfg.get("eval_batch_size", 1)),
        "image_max_pixels": cfg.get("image_max_pixels"),
        "eval_max_new_tokens": int(cfg.get("eval_max_new_tokens", 256)),
    }
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CoVLM Qwen-VL planning baseline latency.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--warmup-samples", type=int, default=5)
    parser.add_argument("--measured-samples", type=int, default=100)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = benchmark(args.config, args.checkpoint_dir, args.split, args.warmup_samples, args.measured_samples)
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(f"[DONE] Benchmark metrics saved to {args.output}; measured_samples={metrics['measured_samples']}")


if __name__ == "__main__":
    main()
