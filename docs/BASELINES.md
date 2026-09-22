# Compared V2X planners

Three task-trained V2X planners are compared with CoVLM-Drive on cooperative
planning. They are prior work by their respective authors; this repository
reproduces them under a shared evaluation protocol so that every method is
scored on the same 654 validation examples with the same trajectory and command
metrics.

| Method | How it is obtained | Reported FDE (m) |
| --- | --- | --- |
| UniV2X | evaluated from the released predictions | 5.861 |
| UniMM-V2X | our protocol-aligned reproduction | 5.357 |
| MAP | our protocol-aligned reproduction | 5.240 |

All three run in the **UniV2X environment**, not the CoVLM-Drive one; see
[INSTALL.md](INSTALL.md).

## Protocol alignment

Every method predicts six waypoints over a three-second horizon in the ego frame
and is scored with the same code path:

- the same fixed validation split of 654 examples,
- the same reference trajectories, derived from recorded ego motion,
- the same L2 horizons and the same FDE definition (L2 at 3.0 s),
- the same command vocabulary for the command metrics.

Efficiency is measured with batch size one on an RTX 4090 and reports model
computation time.

## UniV2X

Configurations are under `projects/configs_e2e_univ2x/`. Training and evaluation
use the distributed launch scripts:

```bash
# train:    <config> <num gpus>
bash tools/univ2x_dist_train.sh projects/configs_e2e_univ2x/univ2x_coop_e2e.py 8

# evaluate: <config> <checkpoint> <num gpus>
bash tools/univ2x_dist_eval.sh \
    projects/configs_e2e_univ2x/univ2x_coop_e2e.py \
    path/to/checkpoint.pth 8
```

The reported UniV2X row is computed from the released predictions rather than a
retrained model.

## UniMM-V2X

Uses the same launcher with the cooperative command configuration:

```bash
bash tools/univ2x_dist_train.sh \
    projects/configs_e2e_univ2x/unimmv2x_covlm_cmd_coop_e2e.py 8
```

Export its predictions into the shared evaluation format with:

```bash
python -m projects.covla_baseline.tools.export_unimmv2x_covlm_eval
```

## MAP

MAP is fetched and run by a driver script, which pins the upstream commit so the
reproduction is reproducible:

```bash
export MAP_PYTHON=/path/to/univ2x-env/bin/python
export COVLM_V2X_DATA_ROOT=data/V2X-Seq-SPD/cooperative
export COVLM_V2X_INFO_ROOT=data/infos/V2X-Seq-SPD/cooperative

python -m projects.covla_baseline.tools.run_map_external_baseline
```

Every path it needs is read from an environment variable with a repository-relative
default, so nothing has to be edited in the script.

## Data conversion

`tools/spd_data_converter/` converts V2X-Seq-SPD into the format the
UniV2X-derived planners consume, and `tools/spd_evaluator/` holds the shared
evaluation utilities. Both are derived from upstream UniV2X; see
[../NOTICE](../NOTICE).

## Zero-shot planning baselines

The zero-shot planning driver shares its backbone-loading path with the
question-answering driver, which is part of the CDQA release. It is listed in the
[roadmap](../README.md#roadmap) and is not included in this snapshot. The
reported zero-shot values are in
[`../results/cp_results.json`](../results/cp_results.json) and
[RESULTS.md](RESULTS.md).
