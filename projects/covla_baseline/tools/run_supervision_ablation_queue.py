"""GPU-serial execution of the appendix supervision ablations.

Uses the accepted training/evaluation implementations. Training stages fail closed:
an interrupted training directory is never mistaken for an optimizer resume.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from projects.covla_baseline.tools import run_raw_dual_threepart_submission as common

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output/covla_baseline"
CP_SOURCE = OUT / "qwen3_vl_8b_raw_dual_threepart_submission_seed20260524_20260830_101809"
QA_SOURCE = OUT / "qwen3_vl_8b_l3qa_pretrain_to_l4_cot_seed20260524_20260608_193311"
QA_EVAL = OUT / "l3qa_grounded_paper_contract_20260828"
MODULE = "projects.covla_baseline.tools.run_supervision_ablation_queue"
VARIANTS = {"cp_only": (False, False), "qa_then_cp": (True, False),
            "qa_then_cp_cot": (True, True)}


def paths(run: Path, variant: str, smoke: bool = False) -> tuple[Path, Path]:
    parent = run / "smoke" if smoke else run
    return parent / variant, parent / "configs" / f"{variant}.yaml"


def validate_config(cfg: dict, qa: bool, cot: bool) -> None:
    expected = {"seed": 20260524, "no_cot": False, "use_ego_status_prompt": False,
                "train_lm_targets": cot, "lambda_lm": float(cot),
                "head_pooling": "mean", "fusion_mode": "standard",
                "mode": "v2x_image", "image_max_pixels": 262144,
                "gradient_accumulation_steps": 8, "index_target_parts": 3}
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"Frozen contract mismatch: {key}={cfg.get(key)!r}, expected {value!r}")
    if bool(cfg.get("init_checkpoint_dir")) != qa:
        raise ValueError("QA adapter initialization mismatch")
    if qa:
        adapter = Path(cfg["init_checkpoint_dir"])
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            if not (adapter / name).is_file():
                raise FileNotFoundError(adapter / name)
        if (adapter / "action_heads.pt").exists():
            raise ValueError("QA initialization must not contain planning heads")


def prepare(run: Path) -> None:
    manifest = run / "contract.json"
    if manifest.exists():
        raise FileExistsError(f"Already prepared: {manifest}; use smoke/queue to continue")
    cfg = common.load_yaml(CP_SOURCE / "configs/standard/train_10ep.yaml")
    rows = common.read_jsonl(Path(cfg["index_path"]))
    if Counter(r["split"] for r in rows) != {"train": 1475, "val": 654}:
        raise ValueError("Canonical CP population mismatch")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    train_rows = [r for r in rows if r["split"] == "train"]
    lengths = [len(tokenizer.encode(r["target_text"])) for r in train_rows]
    worst = sorted(range(len(train_rows)), key=lambda i: lengths[i], reverse=True)[:8]
    val_rows = [r for r in rows if r["split"] == "val"][:4]
    smoke_index = run / "smoke/index.jsonl"
    common.write_jsonl(smoke_index, [train_rows[i] for i in worst] + val_rows)
    for row in rows:
        for key in ("ego_image", "infra_image"):
            if not Path(row[key]).is_file():
                raise FileNotFoundError(row[key])
        if len(row["waypoints"]) != 6:
            raise ValueError(f"Invalid trajectory: {row['sample_id']}")
    for variant, (qa, cot) in VARIANTS.items():
        for smoke in (False, True):
            dest, config_path = paths(run, variant, smoke)
            updated = dict(cfg)
            updated.update(train_lm_targets=cot, lambda_lm=float(cot),
                           init_checkpoint_dir=str(QA_SOURCE / "stage1_l3qa/model") if qa else None,
                           output_dir=str(dest / "model"),
                           epoch_checkpoint_root=str(dest / "epoch_checkpoints"),
                           best_checkpoint_dir=str(dest / "best_checkpoint"),
                           best_checkpoint_json=str(dest / "best_checkpoint.json"))
            if smoke:
                updated.update(index_path=str(smoke_index), num_train_epochs=1,
                               max_steps=8, log_every=1, eval_every_n_epochs=1,
                               epoch_eval_max_batches=4,
                               early_stop_if_single_class_prediction=False)
            validate_config(updated, qa, cot)
            common.write_yaml(config_path, updated)
    common.write_json(manifest, {
        "created_at": common.utc_now(), "variants": VARIANTS,
        "cp_source": str(CP_SOURCE), "qa_source": str(QA_SOURCE),
        "qa_eval_source": str(QA_EVAL), "cp_splits": {"train": 1475, "val": 654},
        "smoke_train_ids": [train_rows[i]["sample_id"] for i in worst],
        "smoke_target_tokens": [lengths[i] for i in worst],
        "qa_pretraining_adds_compute": True,
        "training_resume": "stage boundaries only; incomplete training fails closed",
    })


def memory_receipt(receipt_path: Path, **extra: object) -> None:
    import torch
    common.write_json(receipt_path, {"status": "passed", "finished_at": common.utc_now(),
                            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                            **extra})


def audit_inputs(run: Path) -> None:
    import torch
    from projects.covla_baseline.data.dataset import CoVLABaselineDataset
    from projects.covla_baseline.data.collator import CoVLACollator
    from projects.covla_baseline.train import import_processor
    cfg = common.load_yaml(paths(run, "cp_only")[1])
    dataset = CoVLABaselineDataset(
        index_path=str(run / "smoke/index.jsonl"), mode="v2x_image", split="train",
        load_images=False, no_cot=False, part1_object_limit=0, part1_sentence_limit=0,
        compact_target_part1=False)
    processor = import_processor(cfg["model_name_or_path"])
    collators = [CoVLACollator(processor=processor, mode="v2x_image",
                             image_max_pixels=262144, include_targets=flag)
                 for flag in (False, True)]
    checked = []
    for sample in dataset:
        without, with_cot = [collate([sample]) for collate in collators]
        n = without["input_ids"].shape[1]
        for key in ("input_ids", "head_attention_mask", "ego_view_mask", "infra_view_mask"):
            if not torch.equal(without[key], with_cot[key][:, :n]):
                raise ValueError(f"CoT removal changed prompt-side {key}")
        if with_cot["head_attention_mask"][:, n:].any():
            raise ValueError("Planning heads can pool rationale targets")
        for key in ("pixel_values", "image_grid_thw", "waypoints", "command_id"):
            if not torch.equal(without[key], with_cot[key]):
                raise ValueError(f"CoT removal changed {key}")
        if "labels" in without or not (with_cot["labels"][:, :n] == -100).all():
            raise ValueError("Unexpected supervision mask")
        checked.append(sample["sample_id"])
    common.write_json(run / "input_audit.json", {"status": "passed", "sample_ids": checked,
                      "identical_prompt_ids_pixels_heads_targets": True})


def train_cp(run: Path, variant: str, smoke: bool) -> None:
    import torch
    from projects.covla_baseline.train import run_training
    dest, config = paths(run, variant, smoke)
    cfg = common.load_yaml(config)
    validate_config(cfg, *VARIANTS[variant])
    metadata = dest / "model/run_metadata_train.json"
    receipt = dest / "train_receipt.json"
    if receipt.exists():
        return
    if metadata.exists() or (dest / "model/train_log.jsonl").exists():
        raise RuntimeError(f"Incomplete training cannot be silently restarted: {dest}")
    torch.cuda.reset_peak_memory_stats()
    run_training(str(config))
    info = common.read_json(metadata)
    expected = 8 if smoke else 14750
    protocol_stop = (info["status"] == "stopped_early" and
                     (info.get("early_stop") or {}).get("reason") == "single_class_prediction")
    if not protocol_stop and (info["status"] != "completed" or info["final_micro_step"] != expected):
        raise RuntimeError(f"Training not complete under frozen contract: {info}")
    for row in common.read_jsonl(dest / "model/train_log.jsonl"):
        for key in ("total_loss", "lm_loss", "waypoint_loss", "command_loss", "fde_loss"):
            if not math.isfinite(row[key]):
                raise FloatingPointError(f"Non-finite training log: {key}")
        if not VARIANTS[variant][1] and row["lm_loss"] != 0:
            raise ValueError("LM supervision unexpectedly active")
    from safetensors import safe_open
    changed = 0
    saved = dest / "model/backbone/adapter_model.safetensors"
    with safe_open(saved, framework="pt", device="cpu") as tensors:
        if VARIANTS[variant][0]:
            original_path = Path(cfg["init_checkpoint_dir"]) / "adapter_model.safetensors"
            with safe_open(original_path, framework="pt", device="cpu") as original:
                for key in tensors.keys():
                    changed += int(not torch.equal(tensors.get_tensor(key), original.get_tensor(key)))
        else:
            changed = sum(int(bool(tensors.get_tensor(k).count_nonzero()))
                          for k in tensors.keys() if "lora_B" in k)
    if not changed:
        raise RuntimeError("No LoRA tensors were updated by training")
    memory_receipt(receipt, micro_steps=info["final_micro_step"],
                   optimizer_steps=info["final_optimizer_step"], changed_lora_tensors=changed,
                   training_outcome=info["status"],
                   paper_eligible=bool(info["best_checkpoint"]["selected_from_non_single_class"]),
                   protocol_early_stop=protocol_stop)


def evaluate_cp(run: Path, variant: str, smoke: bool) -> None:
    import torch
    from projects.covla_baseline import evaluate
    dest, config = paths(run, variant, smoke)
    if (dest / "eval_receipt.json").is_file():
        return
    output = dest / "metrics/val_structured.json"
    predictions = dest / "predictions/val_structured.jsonl"
    torch.cuda.reset_peak_memory_stats()
    sys.argv = ["evaluate", "--config", str(config), "--checkpoint-dir",
                str(dest / "best_checkpoint"), "--split", "val", "--output", str(output),
                "--structured-output", str(predictions), "--profile-eval",
                "--profile-warmup-samples", "0", "--cot-diagnostic-format", "three_part"]
    if smoke:
        sys.argv += ["--max-batches", "4"]
    evaluate.main()
    cfg = common.load_yaml(config)
    rows = [r for r in common.read_jsonl(Path(cfg["index_path"])) if r["split"] == "val"]
    audit = common.validate_structured_predictions(path=predictions, expected_rows=rows)
    metrics = common.read_json(output)
    for key in ("final_displacement_error", "command_accuracy"):
        if not math.isfinite(metrics[key]):
            raise ValueError(f"Non-finite evaluation: {key}")
    for row in common.read_jsonl(predictions):
        if not all(math.isfinite(v) for point in row["pred_waypoints"] for v in point):
            raise ValueError("Non-finite waypoint")
    memory_receipt(dest / "eval_receipt.json", **audit)


def evaluate_qa(run: Path, variant: str, smoke: bool, adapter: Path | None) -> None:
    import torch
    from projects.covla_baseline.tools import run_l3qa_grounded_full_serial as qa
    dest, config = paths(run, variant, smoke)
    if (dest / "qa_eval_receipt.json").is_file():
        return
    eval_dir = dest / "grounded_qa"
    qa.ensure_run_dirs(eval_dir)
    for path_fn in (qa.prompt_manifest_path, qa.overlay_audit_path,
                    qa.source_membership_audit_path, qa.token_source_leakage_audit_path,
                    qa.task_scorability_audit_path, qa.run_contract_path):
        target = path_fn(eval_dir)
        if not target.exists():
            shutil.copy2(path_fn(QA_EVAL), target)
    assets = qa.load_prepared_prompt_assets(eval_dir, 2616)
    if assets is None:
        raise RuntimeError("Missing canonical grounded QA assets")
    rows = assets[0]
    if smoke:
        longest = max(rows, key=lambda r: len(r["prompt"]))["sample_id"]
        rows = [r for r in rows if r["sample_id"] == longest]
    adapter = adapter or dest / "best_checkpoint/backbone"
    adapter_config = common.read_json(adapter / "adapter_config.json")
    base = Path(adapter_config["base_model_name_or_path"])
    model_id = "covlm_drive"
    qa.serial.MODEL_REGISTRY[model_id] = replace(
        qa.serial.MODEL_REGISTRY[model_id], model_path=adapter, processor_path=base)
    args = argparse.Namespace(dry_run=False, max_new_tokens=256, image_max_pixels=1048576,
                              num_shards=1, shard_index=0, device_map="auto", status_every=10)
    torch.cuda.reset_peak_memory_stats()
    qa.generate_model_predictions(args=args, run_dir=eval_dir, model_id=model_id,
                                  prompt_rows=rows)
    qa.assert_prediction_completeness(run_dir=eval_dir, model_ids=[model_id], prompt_rows=rows)
    summaries = qa.summarize_models(run_dir=eval_dir, model_ids=[model_id], prompt_rows=rows)
    if not smoke:
        overall = summaries[0]["grouped"]["Overall"]["by_condition"]
        if any(value["N"] != 2861 for value in overall.values()):
            raise ValueError("QA eligible population changed")
    memory_receipt(dest / "qa_eval_receipt.json", prompt_count=len(rows))


def command_env(gpu: int) -> dict:
    env = common.clean_env(gpu)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def execute(run: Path, name: str, command: list[str], gpu: int, min_free: int) -> None:
    receipt = run / "status" / f"{name}.json"
    if receipt.exists() and common.read_json(receipt).get("status") == "completed":
        return
    log = run / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    while True:
        state = common.gpu_state(gpu)
        free = state.get("free_mib")
        if free is None:
            raise ValueError(f"Unknown GPU state schema: {state}")
        if free >= min_free:
            break
        common.write_json(run / "queue_status.json", {"status": "waiting_for_memory",
                          "stage": name, "gpu": gpu, "min_free_mib": min_free,
                          "gpu_state": state, "updated_at": common.utc_now()})
        time.sleep(30)
    with log.open("a", buffering=1) as handle:
        child = subprocess.Popen(command, cwd=ROOT, env=command_env(gpu), stdout=handle,
                                 stderr=subprocess.STDOUT)
        while child.poll() is None:
            common.write_json(run / "queue_status.json", {"status": "running", "stage": name,
                              "pid": child.pid, "log": str(log), "updated_at": common.utc_now()})
            time.sleep(15)
    status = {"status": "completed" if child.returncode == 0 else "failed",
              "returncode": child.returncode, "log": str(log), "finished_at": common.utc_now()}
    common.write_json(receipt, status)
    if child.returncode:
        common.write_json(run / "queue_status.json", {**status, "stage": name})
        raise RuntimeError(f"Stage failed; queue stopped: {name}; see {log}")


def orchestrate(args: argparse.Namespace) -> None:
    run = args.run_dir
    run.mkdir(parents=True, exist_ok=True)
    with (run / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = run / "queue_status.json"
        if status_path.is_file():
            previous = common.read_json(status_path)
            cmdline = Path(f"/proc/{previous.get('pid', -1)}/cmdline")
            if (previous.get("status") == "running" and cmdline.exists()
                    and str(run) in cmdline.read_bytes().decode(errors="replace")):
                raise RuntimeError("A child of the previous queue is still active; refusing overlap")
        if args.mode == "queue":
            if not (run / "cp_smoke_pass.json").is_file():
                raise RuntimeError("All smoke stages for CP must pass before CP training")
            if not (run / "input_audit.json").is_file():
                raise RuntimeError("Prompt-side parity audit is required")
        smoke = args.mode == "smoke"
        for variant in VARIANTS:
            for action in ("train-cp", "eval-cp", "eval-qa"):
                command = [sys.executable, "-m", MODULE, action, "--run-dir", str(run),
                           "--variant", variant]
                if smoke:
                    command.append("--smoke-stage")
                execute(run, f"{'smoke' if smoke else 'full'}_{variant}_{action}",
                        command, args.gpu, args.min_free_mib)
        common.write_json(run / ("cp_smoke_pass.json" if smoke else "cp_complete.json"),
                          {"status": "passed", "finished_at": common.utc_now()})
        qa_run = run / "qa_drafts"
        qa_module = "projects.covla_baseline.tools.run_qa_target_ablation"
        # QA uses its historical SFT encoder; its memory requirement is separate
        # from the CP collator. A failed QA preflight must not block validated CP.
        if not smoke:
            execute(run, "smoke_qa_drafts_smoke", [sys.executable, "-m", qa_module,
                    "smoke", "--run-dir", str(qa_run), "--gpu-id", str(args.gpu)],
                    args.gpu, args.qa_min_free_mib)
            adapter = qa_run / "stage1_l3qa/smoke/model"
            execute(run, "smoke_qa_drafts_eval", [sys.executable, "-m", MODULE,
                    "eval-qa", "--run-dir", str(run), "--variant", "qa_drafts",
                    "--adapter", str(adapter), "--smoke-stage"], args.gpu,
                    args.qa_min_free_mib)
        action = "smoke" if smoke else "train"
        execute(run, f"{'smoke' if smoke else 'full'}_qa_drafts_{action}",
                [sys.executable, "-m", qa_module, action, "--run-dir", str(qa_run),
                 "--gpu-id", str(args.gpu)], args.gpu, args.qa_min_free_mib)
        adapter = qa_run / "stage1_l3qa" / ("smoke/model" if smoke else "model")
        qa_command = [sys.executable, "-m", MODULE, "eval-qa", "--run-dir", str(run),
                      "--variant", "qa_drafts", "--adapter", str(adapter)]
        if smoke:
            qa_command.append("--smoke-stage")
        execute(run, f"{'smoke' if smoke else 'full'}_qa_drafts_eval", qa_command,
                args.gpu, args.qa_min_free_mib)
        common.write_json(run / ("smoke_pass.json" if smoke else "complete.json"),
                          {"status": "passed", "finished_at": common.utc_now()})
        common.write_json(run / "queue_status.json",
                          {"status": "smoke_complete" if smoke else "completed",
                           "finished_at": common.utc_now()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "audit", "smoke", "queue", "train-cp", "eval-cp", "eval-qa"])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=[*VARIANTS, "qa_drafts"], default="cp_only")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--min-free-mib", type=int, default=26500)
    parser.add_argument("--qa-min-free-mib", type=int, default=45000)
    parser.add_argument("--smoke-stage", action="store_true")
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    if args.mode == "prepare":
        prepare(args.run_dir)
    elif args.mode == "audit":
        audit_inputs(args.run_dir)
    elif args.mode in {"smoke", "queue"}:
        orchestrate(args)
    elif args.mode == "train-cp":
        train_cp(args.run_dir, args.variant, args.smoke_stage)
    elif args.mode == "eval-cp":
        evaluate_cp(args.run_dir, args.variant, args.smoke_stage)
    else:
        evaluate_qa(args.run_dir, args.variant, args.smoke_stage, args.adapter)


if __name__ == "__main__":
    main()
