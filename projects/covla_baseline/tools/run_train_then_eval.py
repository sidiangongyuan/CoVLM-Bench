#!/usr/bin/env python3
"""Run one CoVLM training config followed by aligned validation evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_logged(command: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(command) + "\n")
        log_file.flush()
        subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = args.config.resolve()
    run_dir = args.run_dir.resolve()
    metrics_dir = run_dir / "metrics"
    predictions_dir = run_dir / "predictions"
    checkpoint_dir = run_dir / "best_checkpoint"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    write_json(
        run_dir / "pipeline_status.json",
        {"status": "training", "config": str(config)},
    )
    try:
        run_logged(
            [sys.executable, "-m", "projects.covla_baseline.train", "--config", str(config)],
            run_dir / "logs" / "train.log",
        )
        write_json(
            run_dir / "pipeline_status.json",
            {"status": "evaluating", "config": str(config)},
        )
        run_logged(
            [
                sys.executable,
                "-m",
                "projects.covla_baseline.evaluate",
                "--config",
                str(config),
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--split",
                "val",
                "--action-first-report",
                "--output",
                str(metrics_dir / "final_eval_val.json"),
                "--structured-output",
                str(predictions_dir / "structured_val654.jsonl"),
            ],
            run_dir / "logs" / "eval.log",
        )
    except Exception as exc:
        write_json(
            run_dir / "pipeline_status.json",
            {"status": "failed", "config": str(config), "error": str(exc)},
        )
        raise

    write_json(
        run_dir / "pipeline_status.json",
        {
            "status": "complete",
            "config": str(config),
            "elapsed_seconds": time.time() - started,
            "metrics": str(metrics_dir / "final_eval_val.json"),
            "structured_predictions": str(predictions_dir / "structured_val654.jsonl"),
        },
    )


if __name__ == "__main__":
    main()
