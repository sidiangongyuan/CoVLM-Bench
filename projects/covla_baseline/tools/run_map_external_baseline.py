#!/usr/bin/env python3
"""Run and score MAP as a conditional external planning baseline.

The script treats MAP as an external method. It clones a pinned source tree,
generates local wrapper configs, runs inference only when a clean GPU is
available, and scores the produced planning pickle under the accepted
CoVLM-Bench 654-frame protocol.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import textwrap
import time
import zipfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = ROOT / "output/covla_baseline"
REFERENCE_RUN = OUT_ROOT / "qwen3_vl_8b_cot_supervised_reference_seed20260524_20260603_212658"

DEFAULT_RUN_DIR = OUT_ROOT / f"map_external_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
DEFAULT_MAP_REPO = "https://gitee.com/kymkym/map.git"
DEFAULT_MAP_COMMIT = "4cbada95d07cc627364ff21dd464b03dba4191f1"
DEFAULT_MAP_PYTHON = Path(
    os.environ.get(
        "MAP_PYTHON",
        "python",
    )
)
DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "COVLM_V2X_DATA_ROOT",
        "data/V2X-Seq-SPD/cooperative",
    )
)
DEFAULT_INFO_ROOT = Path(
    os.environ.get(
        "COVLM_V2X_INFO_ROOT",
        "data/infos/V2X-Seq-SPD/cooperative",
    )
)
DEFAULT_SPLIT_FILE = Path(
    os.environ.get(
        "COVLM_V2X_SPLIT_FILE",
        str(ROOT / "data/split_datas/cooperative-split-data-spd.json"),
    )
)
DEFAULT_CKPT = Path(os.environ.get("MAP_CHECKPOINT", str(ROOT / "ckpts/univ2x_coop_e2e_stg2.pth")))
DEFAULT_COVLM_INDEX = REFERENCE_RUN / "index_v3_l4_usable_2129.jsonl"
DEFAULT_COVLM_STRUCTURED = REFERENCE_RUN / "samples/structured_predictions_normal_val654.jsonl"

HORIZONS = ["0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s"]
COMMAND_NA = "--"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "detach") and callable(value.detach):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist") and callable(value.tolist):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def line_count(path: Path) -> int:
    if not path.exists():
        return -1
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    env: Optional[Mapping[str, str]] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
            env=merged_env,
            timeout=timeout,
            check=False,
        )
    return {
        "cmd": list(cmd),
        "cwd": str(cwd),
        "returncode": proc.returncode,
        "elapsed_sec": time.perf_counter() - start,
        "log": str(log_path),
    }


def query_gpus() -> List[Dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except Exception:
        return []
    rows: List[Dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_mib": int(float(parts[2])),
                "memory_total_mib": int(float(parts[3])),
                "utilization_gpu_percent": int(float(parts[4])),
            }
        )
    return rows


def select_gpu(args: argparse.Namespace) -> Tuple[Optional[int], Dict[str, Any]]:
    rows = query_gpus()
    status = {
        "gpus": rows,
        "memory_threshold_mib": args.gpu_memory_threshold_mib,
        "util_threshold_percent": args.gpu_util_threshold_percent,
        "requested_gpu": args.gpu,
        "allow_busy_gpu": bool(args.allow_busy_gpu),
    }
    if args.gpu is not None:
        selected = next((row for row in rows if row["index"] == args.gpu), None)
        if selected is None:
            status.update({"state": "missing_requested_gpu"})
            return None, status
        ok = (
            selected["memory_used_mib"] <= args.gpu_memory_threshold_mib
            and selected["utilization_gpu_percent"] <= args.gpu_util_threshold_percent
        )
        if ok or args.allow_busy_gpu:
            status.update({"state": "selected_requested_gpu", "selected_gpu": args.gpu})
            return args.gpu, status
        status.update({"state": "requested_gpu_busy", "selected_gpu": None})
        return None, status

    for row in rows:
        if (
            row["memory_used_mib"] <= args.gpu_memory_threshold_mib
            and row["utilization_gpu_percent"] <= args.gpu_util_threshold_percent
        ):
            status.update({"state": "selected_auto_gpu", "selected_gpu": row["index"]})
            return int(row["index"]), status
    status.update({"state": "no_free_gpu", "selected_gpu": None})
    return None, status


def mkdirs(run_dir: Path) -> None:
    for sub in ["configs", "external", "logs", "metrics", "samples", "scripts", "tables"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)


def git_head(path: Path) -> Optional[str]:
    if not (path / ".git").exists():
        return None
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(path),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def prepare_map_source(args: argparse.Namespace, run_dir: Path) -> Tuple[Optional[Path], Dict[str, Any]]:
    map_dir = args.map_source_dir or (run_dir / "external/map")
    status: Dict[str, Any] = {
        "repo_url": args.map_repo_url,
        "target_commit": args.map_commit,
        "map_dir": str(map_dir),
    }
    if map_dir.exists() and not (map_dir / ".git").exists():
        status.update({"state": "failed", "reason": "map_dir_exists_without_git"})
        return None, status
    if not map_dir.exists():
        clone = run_cmd(
            ["git", "clone", "--depth", "1", args.map_repo_url, str(map_dir)],
            cwd=run_dir,
            log_path=run_dir / "logs/map_clone.log",
            timeout=args.git_timeout_sec,
        )
        status["clone"] = clone
        if clone["returncode"] != 0:
            status.update({"state": "failed_clone"})
            return None, status

    head = git_head(map_dir)
    status["head_before_checkout"] = head
    if head != args.map_commit:
        fetch = run_cmd(
            ["git", "fetch", "--depth", "1", "origin", args.map_commit],
            cwd=map_dir,
            log_path=run_dir / "logs/map_fetch_commit.log",
            timeout=args.git_timeout_sec,
        )
        checkout_ref = "FETCH_HEAD" if fetch["returncode"] == 0 else args.map_commit
        checkout = run_cmd(
            ["git", "checkout", "--detach", checkout_ref],
            cwd=map_dir,
            log_path=run_dir / "logs/map_checkout_commit.log",
            timeout=args.git_timeout_sec,
        )
        status["fetch"] = fetch
        status["checkout"] = checkout
        if checkout["returncode"] != 0:
            status.update({"state": "failed_checkout"})
            return None, status
    status["head_after_checkout"] = git_head(map_dir)
    status["state"] = "ready" if status["head_after_checkout"] == args.map_commit else "ready_unpinned"
    return map_dir, status


def python_smoke(args: argparse.Namespace, run_dir: Path) -> Dict[str, Any]:
    cmd = [
        str(args.python),
        "-c",
        (
            "import torch, mmcv, mmdet, mmdet3d; "
            "print('ok', torch.__version__, mmcv.__version__)"
        ),
    ]
    result = run_cmd(cmd, cwd=ROOT, log_path=run_dir / "logs/python_import_smoke.log")
    result["python"] = str(args.python)
    result["state"] = "ready" if result["returncode"] == 0 else "failed"
    return result


def load_val_tokens(index_path: Path) -> Dict[str, Dict[str, Any]]:
    rows = {}
    for row in read_jsonl(index_path):
        if row.get("split") == "val":
            rows[str(row["token"])] = row
    return rows


def preflight_paths(args: argparse.Namespace) -> Dict[str, Any]:
    val_index = load_val_tokens(args.covlm_index) if args.covlm_index.exists() else {}
    paths = {
        "python": str(args.python),
        "python_exists": args.python.exists(),
        "data_root": str(args.data_root),
        "data_root_exists": args.data_root.exists(),
        "info_root": str(args.info_root),
        "val_ann_file": str(args.info_root / "spd_infos_temporal_val.pkl"),
        "val_ann_file_exists": (args.info_root / "spd_infos_temporal_val.pkl").exists(),
        "split_file": str(args.split_file),
        "split_file_exists": args.split_file.exists(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_exists": args.checkpoint.exists(),
        "covlm_index": str(args.covlm_index),
        "covlm_index_exists": args.covlm_index.exists(),
        "covlm_val_count": len(val_index),
        "covlm_structured": str(args.covlm_structured),
        "covlm_structured_exists": args.covlm_structured.exists(),
        "covlm_structured_rows": line_count(args.covlm_structured),
    }
    paths["state"] = "ready" if all(
        [
            paths["python_exists"],
            paths["data_root_exists"],
            paths["val_ann_file_exists"],
            paths["split_file_exists"],
            paths["checkpoint_exists"],
            paths["covlm_index_exists"],
            paths["covlm_val_count"] == 654,
            paths["covlm_structured_exists"],
            paths["covlm_structured_rows"] == 654,
        ]
    ) else "blocked"
    return paths


def _path_text(path: Path) -> str:
    text = str(path)
    return text if text.endswith("/") else text + "/"


def write_generated_config(
    *,
    run_dir: Path,
    map_dir: Path,
    config_name: str,
    base_config: str,
    ann_file: Path,
    args: argparse.Namespace,
) -> Path:
    config_path = run_dir / "configs" / config_name
    base = map_dir / "projects/configs_e2e_univ2x" / base_config
    data_root = _path_text(args.data_root)
    body = f"""
_base_ = [r"{base}"]

from mmcv import Config as _Config

_base_cfg = _Config.fromfile(r"{base}")
data = _base_cfg.data
file_client_args = dict(backend="disk")
_covlm_data_root = r"{data_root}"
_covlm_val_ann_file = r"{ann_file}"
_covlm_train_ann_file = r"{args.info_root / 'spd_infos_temporal_train.pkl'}"
_covlm_split_datas_file = r"{args.split_file}"

data["workers_per_gpu"] = {int(args.workers_per_gpu)}
for _split_name in ["train", "val", "test"]:
    if _split_name in data:
        _split = data[_split_name]
        _split["file_client_args"] = file_client_args
        _split["data_root"] = _covlm_data_root
        _split["split_datas_file"] = _covlm_split_datas_file
        _split["ann_file"] = _covlm_train_ann_file if _split_name == "train" else _covlm_val_ann_file
        if "pipeline" in _split and len(_split["pipeline"]) > 0:
            _split["pipeline"][0]["file_client_args"] = file_client_args
            _split["pipeline"][0]["img_root"] = _covlm_data_root
data["test"]["test_mode"] = True
"""
    config_path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return config_path


class NumpyCoreCompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_pickle_compat(path: Path) -> Any:
    with path.open("rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as exc:
            if exc.name != "numpy._core":
                raise
    with path.open("rb") as f:
        return NumpyCoreCompatUnpickler(f).load()


def make_covlm_eval_ann(args: argparse.Namespace, run_dir: Path) -> Path:
    src = args.info_root / "spd_infos_temporal_val.pkl"
    dst = run_dir / "configs" / "spd_infos_temporal_val_covlm654_compat.pkl"
    accepted_tokens = set(load_val_tokens(args.covlm_index))
    payload = load_pickle_compat(src)
    if not isinstance(payload, dict) or not isinstance(payload.get("infos"), list):
        raise TypeError(f"Unsupported annotation payload type: {type(payload)}")
    new_payload = dict(payload)
    infos = [info for info in payload["infos"] if str(info.get("token")) in accepted_tokens]
    if len(infos) != 654:
        raise ValueError(f"Expected 654 accepted val infos, found {len(infos)}")
    new_payload["infos"] = infos
    with dst.open("wb") as f:
        pickle.dump(new_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return dst


def make_subset_ann(args: argparse.Namespace, run_dir: Path, count: int) -> Path:
    src = make_covlm_eval_ann(args, run_dir)
    dst = run_dir / "configs" / f"spd_infos_temporal_val_first{count}.pkl"
    payload = load_pickle_compat(src)
    if isinstance(payload, dict) and isinstance(payload.get("infos"), list):
        new_payload = dict(payload)
        new_payload["infos"] = payload["infos"][:count]
    elif isinstance(payload, list):
        new_payload = payload[:count]
    else:
        raise TypeError(f"Unsupported annotation payload type: {type(payload)}")
    with dst.open("wb") as f:
        pickle.dump(new_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return dst


def write_profile_script(run_dir: Path) -> Path:
    script = run_dir / "scripts/map_profile_eval.py"
    script.write_text(
        textwrap.dedent(
            r'''
            #!/usr/bin/env python3
            from __future__ import annotations

            import argparse
            from datetime import datetime, timezone
            import json
            import os
            from pathlib import Path
            import sys
            import time
            from typing import Any, Dict, List

            import torch
            from mmcv import Config
            from mmcv.runner import load_checkpoint, wrap_fp16_model
            from mmcv.parallel import MMDataParallel
            from mmdet.datasets import replace_ImageToTensor
            from mmdet3d.datasets import build_dataset
            from mmdet3d.models import build_model


            def write_json(path: Path, payload: Dict[str, Any]) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


            def stats(values: List[float]) -> Dict[str, Any]:
                if not values:
                    return {"count": 0, "status": "empty"}
                vals = sorted(float(v) for v in values)
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / len(vals)
                return {
                    "count": len(vals),
                    "status": "ok",
                    "mean": mean,
                    "std": var ** 0.5,
                    "min": vals[0],
                    "max": vals[-1],
                    "median": vals[len(vals) // 2],
                }


            def import_config_plugins(cfg: Config, config_path: Path) -> None:
                if cfg.get("custom_imports", None):
                    from mmcv.utils import import_modules_from_strings
                    import_modules_from_strings(**cfg["custom_imports"])
                if getattr(cfg, "plugin", False):
                    import importlib
                    plugin_dir = getattr(cfg, "plugin_dir", None)
                    if plugin_dir:
                        module_path = ".".join(Path(plugin_dir).parts)
                    else:
                        module_path = ".".join(config_path.parent.parts)
                    importlib.import_module(module_path)


            def prepare_test_data_cfg(cfg: Config) -> int:
                samples_per_gpu = 1
                if isinstance(cfg.data.test, dict):
                    cfg.data.test.test_mode = True
                    samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
                    if samples_per_gpu > 1:
                        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
                else:
                    for ds_cfg in cfg.data.test:
                        ds_cfg.test_mode = True
                    samples_per_gpu = max(ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test)
                    if samples_per_gpu > 1:
                        for ds_cfg in cfg.data.test:
                            ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
                return int(samples_per_gpu)


            def build_models(cfg: Config, checkpoint_path: Path) -> Any:
                from projects.mmdet3d_plugin.univ2x.detectors.multi_agent import MultiAgent

                other_agents = {}
                for key in cfg.keys():
                    if "model_other_agent" in key:
                        agent_cfg = cfg.get(key)
                        agent_cfg.train_cfg = None
                        model = build_model(agent_cfg, test_cfg=cfg.get("test_cfg"))
                        if agent_cfg.load_from:
                            load_checkpoint(
                                model,
                                agent_cfg.load_from,
                                map_location="cpu",
                                revise_keys=[(rf"^{key}\.", "")],
                            )
                        other_agents[key] = model

                cfg.model_ego_agent.train_cfg = None
                ego = build_model(cfg.model_ego_agent, test_cfg=cfg.get("test_cfg"))
                if cfg.model_ego_agent.load_from:
                    load_checkpoint(
                        ego,
                        cfg.model_ego_agent.load_from,
                        map_location="cpu",
                        revise_keys=[(r"^model_ego_agent\.", "")],
                    )
                model = MultiAgent(ego, other_agents)
                if cfg.get("fp16", None):
                    wrap_fp16_model(model)
                load_checkpoint(model, str(checkpoint_path), map_location="cpu")
                return model


            def main() -> None:
                parser = argparse.ArgumentParser()
                parser.add_argument("--map-source", type=Path, required=True)
                parser.add_argument("--config", type=Path, required=True)
                parser.add_argument("--checkpoint", type=Path, required=True)
                parser.add_argument("--output", type=Path, required=True)
                parser.add_argument("--warmup", type=int, default=20)
                parser.add_argument("--measure", type=int, default=100)
                parser.add_argument("--workers-per-gpu", type=int, default=0)
                args = parser.parse_args()

                args.map_source = args.map_source.resolve()
                args.config = args.config.resolve()
                args.checkpoint = args.checkpoint.resolve()
                args.output = args.output.resolve()

                os.chdir(args.map_source)
                sys.path.insert(0, str(args.map_source))
                from projects.mmdet3d_plugin.datasets.builder import build_dataloader

                cfg = Config.fromfile(str(args.config))
                import_config_plugins(cfg, args.config)
                if cfg.get("cudnn_benchmark", False):
                    torch.backends.cudnn.benchmark = True
                samples_per_gpu = prepare_test_data_cfg(cfg)
                dataset = build_dataset(cfg.data.test)
                data_loader = build_dataloader(
                    dataset,
                    samples_per_gpu=samples_per_gpu,
                    workers_per_gpu=args.workers_per_gpu,
                    dist=False,
                    shuffle=False,
                    nonshuffler_sampler=cfg.data.nonshuffler_sampler,
                )
                model = build_models(cfg, args.checkpoint)
                param_count = int(sum(p.numel() for p in model.parameters()))
                model = MMDataParallel(model.cuda(), device_ids=[0])
                model.eval()
                if samples_per_gpu != 1:
                    raise ValueError(f"common profiler requires batch size 1, got {samples_per_gpu}")
                if len(dataset) < max(args.warmup, args.measure):
                    raise ValueError("dataset is shorter than the common profiling sample set")
                data_infos = getattr(dataset, "data_infos", [])
                dataset_tokens = [str(info.get("token") or "") for info in data_infos]
                torch.cuda.empty_cache()

                timings: List[float] = []
                with torch.no_grad():
                    for index, data in enumerate(data_loader):
                        if index >= args.warmup:
                            break
                        _ = model(return_loss=False, rescale=True, **data)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()

                wall_start = time.perf_counter()
                with torch.no_grad():
                    for index, data in enumerate(data_loader):
                        if index >= args.measure:
                            break
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        _ = model(return_loss=False, rescale=True, **data)
                        torch.cuda.synchronize()
                        elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(samples_per_gpu, 1)
                        timings.append(elapsed_ms)

                latency = stats(timings)
                mean_ms = latency.get("mean")
                write_json(
                    args.output,
                    {
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "ok",
                        "profile_scope": "common_accepted_token_device_model_return",
                        "timing_contract": (
                            "batch 1; data loading and metric/serialization postprocessing excluded; "
                            "CPU-to-device transfer, model inference, and returned tensor construction included"
                        ),
                        "config": str(args.config),
                        "checkpoint": str(args.checkpoint),
                        "dataset_count": len(dataset),
                        "samples_per_gpu": samples_per_gpu,
                        "workers_per_gpu": args.workers_per_gpu,
                        "warmup_count": args.warmup,
                        "warmup_tokens": dataset_tokens[: args.warmup],
                        "measured_count": len(timings),
                        "measured_tokens": dataset_tokens[: len(timings)],
                        "latency_ms_per_sample": mean_ms,
                        "throughput_samples_per_sec": (1000.0 / mean_ms) if mean_ms else None,
                        "throughput_samples_per_sec_common": (1000.0 / mean_ms) if mean_ms else None,
                        "device_model_return_ms_per_sample": latency,
                        "model_forward_ms_per_sample": latency,
                        "total_profile_wall_time_sec": time.perf_counter() - wall_start,
                        "cuda_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "cuda_peak_memory_allocated_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
                        "cuda_peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
                        "model_parameter_count": param_count,
                        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "torch_version": torch.__version__,
                        "cuda_runtime": torch.version.cuda,
                        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    },
                )


            if __name__ == "__main__":
                main()
            '''
        ).lstrip(),
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def write_numpy_pickle_sitecustomize(run_dir: Path) -> Path:
    path = run_dir / "scripts" / "sitecustomize.py"
    path.write_text(
        textwrap.dedent(
            """
            import importlib
            import sys

            try:
                import numpy.core as _np_core

                sys.modules.setdefault("numpy._core", _np_core)
                for _name in ["multiarray", "numeric", "fromnumeric", "umath"]:
                    try:
                        _module = importlib.import_module(f"numpy.core.{_name}")
                    except Exception:
                        continue
                    sys.modules.setdefault(f"numpy._core.{_name}", _module)
            except Exception:
                pass
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return path


def inference_env(
    gpu: int,
    *,
    map_dir: Optional[Path] = None,
    run_dir: Optional[Path] = None,
) -> Dict[str, str]:
    env = {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONUNBUFFERED": "1",
        "RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(29500 + (os.getpid() % 1000)),
    }
    pythonpath = []
    if run_dir is not None:
        write_numpy_pickle_sitecustomize(run_dir)
        pythonpath.append(str(run_dir / "scripts"))
    if map_dir is not None:
        pythonpath.append(str(map_dir))
    if pythonpath:
        existing = os.environ.get("PYTHONPATH")
        if existing:
            pythonpath.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return env


def run_map_inference(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    map_dir: Path,
    config: Path,
    out_pkl: Path,
    gpu: int,
    label: str,
) -> Dict[str, Any]:
    cmd = [
        str(args.python),
        "tools/inference.py",
        str(config),
        str(args.checkpoint),
        "--out",
        str(out_pkl),
        "--launcher",
        "pytorch",
        "--seed",
        "0",
        "--deterministic",
    ]
    result = run_cmd(
        cmd,
        cwd=map_dir,
        log_path=run_dir / "logs" / f"{label}_inference.log",
        env=inference_env(gpu, map_dir=map_dir),
        timeout=args.inference_timeout_sec,
    )
    result["output_pkl"] = str(out_pkl)
    result["output_exists"] = out_pkl.exists()
    result["gpu"] = gpu
    return result


def run_map_profile(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    map_dir: Path,
    config: Path,
    gpu: int,
) -> Dict[str, Any]:
    script = write_profile_script(run_dir)
    out = run_dir / "metrics/map_external_baseline_profile.json"
    cmd = [
        str(args.python),
        str(script),
        "--map-source",
        str(map_dir),
        "--config",
        str(config),
        "--checkpoint",
        str(args.checkpoint),
        "--output",
        str(out),
        "--warmup",
        str(args.profile_warmup),
        "--measure",
        str(args.profile_measure),
        "--workers-per-gpu",
        str(args.workers_per_gpu),
    ]
    result = run_cmd(
        cmd,
        cwd=map_dir,
        log_path=run_dir / "logs/map_profile.log",
        env=inference_env(gpu, map_dir=map_dir),
        timeout=args.profile_timeout_sec,
    )
    result["profile_output"] = str(out)
    result["profile_output_exists"] = out.exists()
    return result


def tensorish_to_list(value: Any) -> Any:
    if hasattr(value, "detach") and callable(value.detach):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist") and callable(value.tolist):
        return value.tolist()
    return value


def normalize_xy_sequence(value: Any) -> List[List[float]]:
    raw = tensorish_to_list(value)
    while isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list):
        raw = raw[0]
    if not isinstance(raw, list):
        raise TypeError(f"trajectory is not list-like: {type(raw)}")
    points: List[List[float]] = []
    for point in raw:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        points.append([float(point[0]), float(point[1])])
    return points


def get_nested(mapping: Mapping[str, Any], keys: Sequence[str]) -> Optional[Any]:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def extract_external_items(payload: Any) -> List[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        for key in ["bbox_results", "results", "outputs"]:
            value = payload.get(key)
            if isinstance(value, list):
                return value
        if all(isinstance(v, Mapping) for v in payload.values()):
            return list(payload.values())
    if isinstance(payload, list):
        return payload
    raise TypeError(f"Unsupported external pickle payload: {type(payload)}")


def extract_token(item: Mapping[str, Any]) -> Optional[str]:
    for key in ["token", "sample_idx", "sample_id"]:
        if key in item:
            value = item[key]
            if isinstance(value, str):
                return value.replace("val_", "")
            return str(value)
    meta = item.get("img_metas")
    if isinstance(meta, Mapping):
        value = meta.get("sample_idx") or meta.get("token")
        if value is not None:
            return str(value).replace("val_", "")
    return None


def extract_planning_traj(item: Mapping[str, Any]) -> Optional[List[List[float]]]:
    candidates = [
        item.get("planning_traj"),
        get_nested(item, ["planning", "result_planning", "sdc_traj"]),
        item.get("sdc_traj"),
        item.get("pred_planning_traj"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            points = normalize_xy_sequence(candidate)
        except Exception:
            continue
        if len(points) >= 6:
            return points
    return None


def univ2x_to_covlm(points: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[-float(point[1]), float(point[0])] for point in points]


def l2_steps(pred: Sequence[Sequence[float]], gt: Sequence[Sequence[float]]) -> List[float]:
    return [
        math.hypot(float(p[0]) - float(g[0]), float(p[1]) - float(g[1]))
        for p, g in zip(pred[:6], gt[:6])
    ]


def mean(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    if not vals:
        return None
    return sum(vals) / len(vals)


def load_covlm_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        rows[str(row["token"])] = row
    return rows


def score_external_pkl(
    *,
    run_dir: Path,
    pkl_path: Path,
    covlm_index_path: Path,
    covlm_structured_path: Path,
    profile_path: Optional[Path],
    allow_partial: bool = False,
) -> Dict[str, Any]:
    with pkl_path.open("rb") as f:
        payload = pickle.load(f)
    items = extract_external_items(payload)
    external: Dict[str, Dict[str, Any]] = {}
    parse_failures: List[Dict[str, Any]] = []
    for idx, item in enumerate(items):
        token = extract_token(item)
        traj = extract_planning_traj(item)
        if token is None or traj is None:
            parse_failures.append(
                {
                    "idx": idx,
                    "has_token": token is not None,
                    "has_traj": traj is not None,
                    "keys": sorted(str(k) for k in item.keys())[:30] if isinstance(item, Mapping) else [],
                }
            )
            continue
        external[token] = {
            "token": token,
            "pred_univ2x": traj,
            "command_raw": tensorish_to_list(item.get("command")),
        }

    index_rows = load_val_tokens(covlm_index_path)
    covlm_rows = load_covlm_rows(covlm_structured_path)
    accepted_tokens = set(index_rows)
    overlap_tokens = sorted(accepted_tokens & set(external) & set(covlm_rows))
    missing = sorted(accepted_tokens - set(external))
    extra = sorted(set(external) - accepted_tokens)

    per_sample: List[Dict[str, Any]] = []
    step_values: List[List[float]] = [[] for _ in HORIZONS]
    ade_values: List[float] = []
    fde_values: List[float] = []
    for token in overlap_tokens:
        pred_covlm = univ2x_to_covlm(external[token]["pred_univ2x"][:6])
        gt_covlm = covlm_rows[token]["gt_waypoints"][:6]
        distances = l2_steps(pred_covlm, gt_covlm)
        for idx, value in enumerate(distances):
            step_values[idx].append(value)
        ade = mean(distances)
        fde = distances[-1]
        ade_values.append(float(ade))
        fde_values.append(float(fde))
        per_sample.append(
            {
                "token": token,
                "sample_id": covlm_rows[token].get("sample_id"),
                "map_pred_univ2x_6": external[token]["pred_univ2x"][:6],
                "map_pred_covlm_6": pred_covlm,
                "covlm_gt_waypoints_6": gt_covlm,
                "ade": float(ade),
                "fde": float(fde),
                "command_raw": external[token].get("command_raw"),
                "gt_command": covlm_rows[token].get("gt_command"),
            }
        )

    profile: Optional[Dict[str, Any]] = None
    if profile_path and profile_path.exists():
        profile = read_json(profile_path)

    full_ok = len(overlap_tokens) == 654 and not missing
    fde = mean(fde_values)
    ade = mean(ade_values)
    metrics = {
        "created_at_utc": utc_now(),
        "method": "MAP",
        "source": "https://gitee.com/kymkym/map.git",
        "external_pkl": str(pkl_path),
        "accepted_val_count": len(accepted_tokens),
        "external_parsed_count": len(external),
        "overlap_count": len(overlap_tokens),
        "missing_accepted_tokens": missing[:50],
        "missing_accepted_count": len(missing),
        "extra_external_tokens": extra[:50],
        "extra_external_count": len(extra),
        "parse_failure_count": len(parse_failures),
        "parse_failure_examples": parse_failures[:20],
        "coordinate_conversion": {
            "native": "MAP/UniV2X [forward,lateral]",
            "covlm": "CoVLM [x_lateral,y_forward]=[-lateral,forward]",
        },
        "horizon_l2": {h: mean(vals) for h, vals in zip(HORIZONS, step_values)},
        "ADE": ade,
        "FDE": fde,
        "command_accuracy": None,
        "macro_command_accuracy_present_classes": None,
        "command_policy": "not_reported_MAP_uses_planning_command_context_not_independent_command_output",
        "profile": profile,
        "acceptance_gates": {
            "n_654": len(overlap_tokens) == 654,
            "tokens_match": full_ok,
            "trajectory_horizon_at_least_6": all(len(row["map_pred_univ2x_6"]) == 6 for row in per_sample),
            "finite_fde": fde is not None and math.isfinite(float(fde)),
            "profile_available": profile is not None and profile.get("status") == "ok",
            "cmd_macro_na": True,
        },
    }
    gates = metrics["acceptance_gates"]
    metrics["state"] = (
        "accepted_for_table"
        if all(gates.values())
        else ("partial_scored" if allow_partial and overlap_tokens else "rejected_protocol")
    )
    metrics["required_table_metrics"] = ["FDE", "Cmd", "Macro", "FPS", "Mem"]
    metrics["table_row_metrics"] = build_table_row_metrics(metrics)

    sample_path = run_dir / "samples/map_bridge_per_sample.jsonl"
    write_jsonl(sample_path, per_sample)
    metrics["artifacts"] = {
        "per_sample": str(sample_path),
        "metrics": str(run_dir / "metrics/map_external_baseline_metrics.json"),
        "tex_row": str(run_dir / "tables/map_external_baseline_row.tex"),
    }
    write_json(run_dir / "metrics/map_external_baseline_metrics.json", metrics)
    write_tex_row(run_dir, metrics)
    return metrics


def fmt_float(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return COMMAND_NA
    return f"{float(value):.{digits}f}"


def build_table_row_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    profile = metrics.get("profile") if isinstance(metrics.get("profile"), Mapping) else {}
    fps = profile.get("throughput_samples_per_sec") if isinstance(profile, Mapping) else None
    mem = profile.get("cuda_peak_memory_allocated_gib") if isinstance(profile, Mapping) else None
    return {
        "method": "MAP",
        "FDE": metrics.get("FDE"),
        "Cmd": None,
        "Macro": None,
        "FPS": fps,
        "Mem": mem,
        "cmd_macro_policy": "not_reported_for_MAP_without_independent_command_prediction",
        "latex_cells": {
            "FDE": fmt_float(metrics.get("FDE"), 3),
            "Cmd": COMMAND_NA,
            "Macro": COMMAND_NA,
            "FPS": fmt_float(fps, 2),
            "Mem": fmt_float(mem, 2),
        },
    }


def write_tex_row(run_dir: Path, metrics: Mapping[str, Any]) -> None:
    cells = build_table_row_metrics(metrics)["latex_cells"]
    row = (
        "MAP~\\benchref{yin2025map} & "
        f"{cells['FDE']} & {cells['Cmd']} & {cells['Macro']} & "
        f"{cells['FPS']} & {cells['Mem']} \\\\\n"
    )
    (run_dir / "tables/map_external_baseline_row.tex").write_text(row, encoding="utf-8")


def write_status(run_dir: Path, payload: Mapping[str, Any]) -> None:
    write_json(run_dir / "metrics/serial_status.json", payload)


def ensure_map_datasets_package(map_dir: Path) -> Dict[str, Any]:
    plugin_dir = map_dir / "projects" / "mmdet3d_plugin"
    datasets_dir = plugin_dir / "datasets"
    builder = datasets_dir / "builder.py"
    zip_path = plugin_dir / "datasets.zip"
    status: Dict[str, Any] = {
        "datasets_dir": str(datasets_dir),
        "builder": str(builder),
        "zip_path": str(zip_path),
    }
    if builder.exists():
        status["state"] = "ready_existing"
        return status
    if not zip_path.exists():
        status["state"] = "missing"
        return status
    datasets_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(datasets_dir)
    status["state"] = "ready_unpacked" if builder.exists() else "failed_unpack"
    return status


def ensure_map_base_configs(map_dir: Path) -> Dict[str, Any]:
    src = ROOT / "projects" / "configs_e2e_univ2x" / "_base_" / "datasets" / "nus-3d.py"
    dst = (
        map_dir
        / "projects"
        / "configs_e2e_univ2x"
        / "_base_"
        / "datasets"
        / "nus-3d.py"
    )
    status = {"source": str(src), "target": str(dst)}
    if dst.exists():
        status["state"] = "ready_existing"
        return status
    if not src.exists():
        status["state"] = "missing_source"
        return status
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    status["state"] = "ready_copied" if dst.exists() else "failed_copy"
    return status


def patch_map_spd_locations(map_dir: Path) -> Dict[str, Any]:
    locations = "['yizhuang06', 'yizhuang08', 'yizhuang09', 'yizhuang10', 'yizhuang13', 'yizhuang16']"
    vector_path = map_dir / "projects/mmdet3d_plugin/datasets/data_utils/vector_map.py"
    dataset_path = map_dir / "projects/mmdet3d_plugin/datasets/spd_vehicle_e2e_dataset.py"
    status: Dict[str, Any] = {
        "locations": locations,
        "vector_map": str(vector_path),
        "dataset": str(dataset_path),
        "patched": [],
    }
    if not vector_path.exists() or not dataset_path.exists():
        status["state"] = "missing_source"
        return status

    vector_text = vector_path.read_text(encoding="utf-8")
    if "self.MAPS = ['yizhuang09']" in vector_text:
        vector_text = vector_text.replace("self.MAPS = ['yizhuang09']", f"self.MAPS = {locations}")
        vector_path.write_text(vector_text, encoding="utf-8")
        status["patched"].append("vector_map")

    dataset_text = dataset_path.read_text(encoding="utf-8")
    old_block = """                    # 'yizhuang06': NuScenesMap(dataroot=self.data_root, map_name='yizhuang06'),
                    # 'yizhuang08': NuScenesMap(dataroot=self.data_root, map_name='yizhuang08'),
                    'yizhuang09': NuScenesMap(dataroot=self.data_root, map_name='yizhuang09'),
                    # 'yizhuang10': NuScenesMap(dataroot=self.data_root, map_name='yizhuang10'),
                    # 'yizhuang13': NuScenesMap(dataroot=self.data_root, map_name='yizhuang13'),
                    # 'yizhuang16': NuScenesMap(dataroot=self.data_root, map_name='yizhuang16')"""
    new_block = """                    'yizhuang06': NuScenesMap(dataroot=self.data_root, map_name='yizhuang06'),
                    'yizhuang08': NuScenesMap(dataroot=self.data_root, map_name='yizhuang08'),
                    'yizhuang09': NuScenesMap(dataroot=self.data_root, map_name='yizhuang09'),
                    'yizhuang10': NuScenesMap(dataroot=self.data_root, map_name='yizhuang10'),
                    'yizhuang13': NuScenesMap(dataroot=self.data_root, map_name='yizhuang13'),
                    'yizhuang16': NuScenesMap(dataroot=self.data_root, map_name='yizhuang16')"""
    if old_block in dataset_text:
        dataset_text = dataset_text.replace(old_block, new_block)
        dataset_path.write_text(dataset_text, encoding="utf-8")
        status["patched"].append("dataset")

    status["state"] = "ready"
    return status


def command_preflight(args: argparse.Namespace) -> int:
    run_dir = args.run_dir
    mkdirs(run_dir)
    map_dir, map_status = prepare_map_source(args, run_dir)
    dataset_package = ensure_map_datasets_package(map_dir) if map_dir is not None else None
    base_configs = ensure_map_base_configs(map_dir) if map_dir is not None else None
    spd_locations = patch_map_spd_locations(map_dir) if map_dir is not None else None
    status = {
        "created_at_utc": utc_now(),
        "state": "preflight",
        "run_dir": str(run_dir),
        "map_source": map_status,
        "map_dataset_package": dataset_package,
        "map_base_configs": base_configs,
        "map_spd_locations": spd_locations,
        "paths": preflight_paths(args),
        "python_smoke": python_smoke(args, run_dir),
        "gpu": select_gpu(args)[1],
    }
    write_status(run_dir, status)
    if map_dir and status["paths"]["state"] == "ready":
        full_ann = make_covlm_eval_ann(args, run_dir)
        write_generated_config(
            run_dir=run_dir,
            map_dir=map_dir,
            config_name="map_primary_covlm654.py",
            base_config="univ2x_coop_e2e.py",
            ann_file=full_ann,
            args=args,
        )
        smoke_ann = make_subset_ann(args, run_dir, args.smoke_samples)
        write_generated_config(
            run_dir=run_dir,
            map_dir=map_dir,
            config_name=f"map_primary_smoke{args.smoke_samples}.py",
            base_config="univ2x_coop_e2e.py",
            ann_file=smoke_ann,
            args=args,
        )
    ready = (
        status["paths"]["state"] == "ready"
        and map_status["state"].startswith("ready")
        and dataset_package is not None
        and str(dataset_package.get("state", "")).startswith("ready")
        and base_configs is not None
        and str(base_configs.get("state", "")).startswith("ready")
        and spd_locations is not None
        and str(spd_locations.get("state", "")).startswith("ready")
    )
    return 0 if ready else 1


def command_run(args: argparse.Namespace) -> int:
    run_dir = args.run_dir
    mkdirs(run_dir)
    map_dir, map_status = prepare_map_source(args, run_dir)
    dataset_package = ensure_map_datasets_package(map_dir) if map_dir is not None else None
    base_configs = ensure_map_base_configs(map_dir) if map_dir is not None else None
    spd_locations = patch_map_spd_locations(map_dir) if map_dir is not None else None
    path_status = preflight_paths(args)
    py_status = python_smoke(args, run_dir)
    gpu, gpu_status = select_gpu(args)
    status: Dict[str, Any] = {
        "created_at_utc": utc_now(),
        "state": "starting",
        "run_dir": str(run_dir),
        "map_source": map_status,
        "map_dataset_package": dataset_package,
        "map_base_configs": base_configs,
        "map_spd_locations": spd_locations,
        "paths": path_status,
        "python_smoke": py_status,
        "gpu": gpu_status,
    }
    write_status(run_dir, status)

    if map_dir is None or not map_status["state"].startswith("ready"):
        status["state"] = "blocked_map_source"
        write_status(run_dir, status)
        return 1
    if not dataset_package or not str(dataset_package.get("state", "")).startswith("ready"):
        status["state"] = "blocked_map_dataset_package"
        write_status(run_dir, status)
        return 1
    if not base_configs or not str(base_configs.get("state", "")).startswith("ready"):
        status["state"] = "blocked_map_base_configs"
        write_status(run_dir, status)
        return 1
    if not spd_locations or not str(spd_locations.get("state", "")).startswith("ready"):
        status["state"] = "blocked_map_spd_locations"
        write_status(run_dir, status)
        return 1
    if path_status["state"] != "ready":
        status["state"] = "blocked_preflight_paths"
        write_status(run_dir, status)
        return 1
    if py_status["state"] != "ready":
        status["state"] = "blocked_python_env"
        write_status(run_dir, status)
        return 1
    if gpu is None:
        status["state"] = "pending_resources"
        write_status(run_dir, status)
        return 0

    smoke_ann = make_subset_ann(args, run_dir, args.smoke_samples)
    smoke_config = write_generated_config(
        run_dir=run_dir,
        map_dir=map_dir,
        config_name=f"map_primary_smoke{args.smoke_samples}.py",
        base_config="univ2x_coop_e2e.py",
        ann_file=smoke_ann,
        args=args,
    )
    full_config = write_generated_config(
        run_dir=run_dir,
        map_dir=map_dir,
        config_name="map_primary_covlm654.py",
        base_config="univ2x_coop_e2e.py",
        ann_file=make_covlm_eval_ann(args, run_dir),
        args=args,
    )

    smoke_pkl = run_dir / f"samples/map_primary_smoke{args.smoke_samples}.pkl"
    smoke = run_map_inference(
        args,
        run_dir=run_dir,
        map_dir=map_dir,
        config=smoke_config,
        out_pkl=smoke_pkl,
        gpu=gpu,
        label=f"smoke{args.smoke_samples}",
    )
    status["smoke"] = smoke
    if smoke["returncode"] != 0 or not smoke["output_exists"]:
        status["state"] = "failed_smoke"
        write_status(run_dir, status)
        return 1

    if args.smoke_only:
        status["state"] = "smoke_completed"
        write_status(run_dir, status)
        return 0

    full_pkl = run_dir / "samples/map_external_baseline_full.pkl"
    full = run_map_inference(
        args,
        run_dir=run_dir,
        map_dir=map_dir,
        config=full_config,
        out_pkl=full_pkl,
        gpu=gpu,
        label="full654",
    )
    status["full_inference"] = full
    if full["returncode"] != 0 or not full["output_exists"]:
        status["state"] = "failed_full_inference"
        write_status(run_dir, status)
        return 1

    profile = run_map_profile(args, run_dir=run_dir, map_dir=map_dir, config=full_config, gpu=gpu)
    status["profile"] = profile
    metrics = score_external_pkl(
        run_dir=run_dir,
        pkl_path=full_pkl,
        covlm_index_path=args.covlm_index,
        covlm_structured_path=args.covlm_structured,
        profile_path=Path(profile["profile_output"]),
        allow_partial=False,
    )
    status["score"] = {
        "state": metrics["state"],
        "overlap_count": metrics["overlap_count"],
        "FDE": metrics["FDE"],
        "table_row_metrics": metrics["table_row_metrics"],
        "acceptance_gates": metrics["acceptance_gates"],
        "metrics": metrics["artifacts"]["metrics"],
        "tex_row": metrics["artifacts"]["tex_row"],
    }
    status["state"] = metrics["state"]
    write_status(run_dir, status)
    return 0 if metrics["state"] == "accepted_for_table" else 1


def command_score(args: argparse.Namespace) -> int:
    run_dir = args.run_dir
    mkdirs(run_dir)
    profile_path = args.profile_json if args.profile_json else None
    metrics = score_external_pkl(
        run_dir=run_dir,
        pkl_path=args.external_pkl,
        covlm_index_path=args.covlm_index,
        covlm_structured_path=args.covlm_structured,
        profile_path=profile_path,
        allow_partial=args.allow_partial,
    )
    write_status(
        run_dir,
        {
            "created_at_utc": utc_now(),
            "state": metrics["state"],
            "score": {
                "overlap_count": metrics["overlap_count"],
                "FDE": metrics["FDE"],
                "table_row_metrics": metrics["table_row_metrics"],
                "acceptance_gates": metrics["acceptance_gates"],
                "metrics": metrics["artifacts"]["metrics"],
                "tex_row": metrics["artifacts"]["tex_row"],
            },
        },
    )
    return 0 if metrics["state"] in {"accepted_for_table", "partial_scored"} else 1


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--map-repo-url", default=DEFAULT_MAP_REPO)
    parser.add_argument("--map-commit", default=DEFAULT_MAP_COMMIT)
    parser.add_argument("--map-source-dir", type=Path, default=None)
    parser.add_argument("--python", type=Path, default=DEFAULT_MAP_PYTHON)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--info-root", type=Path, default=DEFAULT_INFO_ROOT)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--covlm-index", type=Path, default=DEFAULT_COVLM_INDEX)
    parser.add_argument("--covlm-structured", type=Path, default=DEFAULT_COVLM_STRUCTURED)
    parser.add_argument("--workers-per-gpu", type=int, default=0)
    parser.add_argument("--smoke-samples", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--gpu-memory-threshold-mib", type=int, default=2048)
    parser.add_argument("--gpu-util-threshold-percent", type=int, default=10)
    parser.add_argument("--git-timeout-sec", type=int, default=120)
    parser.add_argument("--inference-timeout-sec", type=int, default=7200)
    parser.add_argument("--profile-timeout-sec", type=int, default=1800)
    parser.add_argument("--profile-warmup", type=int, default=20)
    parser.add_argument("--profile-measure", type=int, default=100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser("preflight")
    add_common_args(p_pre)
    p_pre.set_defaults(func=command_preflight)

    p_run = sub.add_parser("run")
    add_common_args(p_run)
    p_run.add_argument("--smoke-only", action="store_true")
    p_run.set_defaults(func=command_run)

    p_score = sub.add_parser("score")
    add_common_args(p_score)
    p_score.add_argument("--external-pkl", type=Path, required=True)
    p_score.add_argument("--profile-json", type=Path, default=None)
    p_score.add_argument("--allow-partial", action="store_true")
    p_score.set_defaults(func=command_score)

    return parser.parse_args()


def maybe_reexec_with_map_python(args: argparse.Namespace) -> None:
    """Run/score phases need the MAP/MMDet environment for pickle imports."""
    if args.command not in {"run", "score"}:
        return
    if os.environ.get("MAP_EXTERNAL_BASELINE_REEXEC") == "1":
        return
    current = Path(sys.executable).resolve()
    target = args.python.resolve()
    if current == target:
        return
    env = os.environ.copy()
    env["MAP_EXTERNAL_BASELINE_REEXEC"] = "1"
    cmd = [str(target), str(Path(__file__).resolve()), *sys.argv[1:]]
    os.execvpe(str(target), cmd, env)


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.resolve()
    args.data_root = args.data_root.resolve()
    args.info_root = args.info_root.resolve()
    args.split_file = args.split_file.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.covlm_index = args.covlm_index.resolve()
    args.covlm_structured = args.covlm_structured.resolve()
    args.python = args.python.resolve()
    if args.map_source_dir is not None:
        args.map_source_dir = args.map_source_dir.resolve()
    maybe_reexec_with_map_python(args)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
