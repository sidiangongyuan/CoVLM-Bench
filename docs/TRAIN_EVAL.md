# Training and evaluating CoVLM-Drive

All commands run from the repository root with `PYTHONPATH="$PWD:$PYTHONPATH"`.

## Configurations

`projects/covla_baseline/configs/` contains the **resolved** configuration of
every CoVLM-Drive run reported for cooperative planning -- the settings as they
were actually applied, not a template. They are JSON and are accepted directly
by `--config`.

| File | Reported row |
| --- | --- |
| `standard_cp.json` | Qwen3-VL-8B, standard CP configuration |
| `ego_status_cp.json` | Qwen3-VL-8B + ego status |
| `backbone_qwen25_vl_7b.json` | Qwen2.5-VL-7B |
| `backbone_internvl3_8b.json` | InternVL3-8B |
| `backbone_llava_onevision_7b.json` | LLaVA-OneVision-7B |
| `backbone_idefics3_8b.json` | Idefics3-8B |
| `supervision_cp_only.json` | CP supervision only |
| `supervision_qa_then_cp.json` | QA initialization, then CP |
| `supervision_qa_then_cp_cot.json` | QA initialization, then CP with rationale supervision |

Paths in them are placeholders -- `${V2X_SEQ_SPD_ROOT}`,
`${COVLM_ANNOTATION_ROOT}`, `${HF_MODEL_ROOT}`, `${OUTPUT_ROOT}`, `${REPO_ROOT}`.
Replace them with your own before use.

The standard configuration trains for 10 epochs at learning rate 5e-5, seed
20260524, LoRA rank 16 with alpha 32, a micro-batch of one accumulated to a
global batch of eight, and loss weights
`lambda_lm=1.0`, `lambda_cmd=1.0`, `lambda_wp=2.0`, `lambda_fde=1.0`.

## Train

```bash
python -m projects.covla_baseline.train \
    --config projects/covla_baseline/configs/standard_cp.json
```

Useful flags:

- `--smoke` -- forces batch size one and at most two steps, for a fast sanity run.
- `--model-name-or-path PATH` -- overrides the backbone without editing the config.

Training jointly optimizes the backbone adapters and the structured heads with
rationale, command, and trajectory supervision. The rationale loss is token-level
cross-entropy under teacher forcing, command classification uses cross-entropy,
and waypoint regression uses Smooth L1 with an additional endpoint term.

## Evaluate

```bash
python -m projects.covla_baseline.evaluate \
    --config          projects/covla_baseline/configs/standard_cp.json \
    --checkpoint-dir  output/covla_baseline/<run>/best_checkpoint \
    --split           val \
    --output          metrics.json
```

This reports L2 at 0.5 s through 2.5 s, FDE at 3.0 s, command accuracy, and
balanced accuracy over the command classes present in the validation split.

Additional outputs:

- `--structured-output preds.jsonl` -- structured command and waypoint
  predictions for every evaluated sample, without generating rationale text.
- `--examples-output examples.jsonl --examples-every-n 50` -- sparse generated
  rationales alongside the structured outputs.
- `--generate-all-text` -- generate a rationale for every evaluated sample.
- `--profile-eval --profile-output profile.json` -- latency, memory, and
  throughput, as reported in the efficiency columns.

## Train and evaluate in one step

```bash
python -m projects.covla_baseline.tools.run_train_then_eval \
    --config  projects/covla_baseline/configs/standard_cp.json \
    --run-dir output/covla_baseline/my_run
```

## Reproducing the reported run groups

The reported CP configurations are generated and launched by dedicated drivers:

```bash
# standard CP and its ego-status variant
python -m projects.covla_baseline.tools.run_raw_dual_threepart_submission

# the supervision ablation (CP only / QA then CP / QA then CP with rationales)
python -m projects.covla_baseline.tools.run_supervision_ablation_queue
```

Each driver writes `train_config_resolved.json` next to the trained model; the
files under `configs/` are exactly those artifacts with local paths replaced.

## Inference cost

```bash
python -m projects.covla_baseline.benchmark \
    --config         projects/covla_baseline/configs/standard_cp.json \
    --checkpoint-dir output/covla_baseline/<run>/best_checkpoint
```

Measures structured planning inference with batch size one.

## Rationales

`three_part_cot.py` builds and parses the three-part rationale targets --
*scene overview*, *V2X-aware critical objects*, and *decision reasoning*. They
are auxiliary supervision, not a separate benchmark task, and a generated
rationale is not a guaranteed explanation of the predicted trajectory.

## Tables

```bash
python -m projects.covla_baseline.tools.build_unified_planning_tables
```

Rebuilds the CP comparison tables from evaluation artifacts. The reported values
are in [`../results/cp_results.json`](../results/cp_results.json).
