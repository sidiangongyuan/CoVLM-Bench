# Data preparation

## 1. V2X-Seq-SPD

CoVLM-Bench is built on paired vehicle- and infrastructure-side frames from
**V2X-Seq-SPD / DAIR-V2X-Seq**. Images and cooperative labels are **not
redistributed** in this repository. Obtain them from the upstream dataset under
its own access terms:

- <https://github.com/AIR-THU/DAIR-V2X-Seq>

Place the extracted dataset anywhere and refer to it as `$V2X_SEQ_SPD_ROOT`.
Code defaults assume `data/V2X-Seq-SPD` relative to the repository root, so a
symlink is convenient:

```bash
mkdir -p data
ln -s /path/to/V2X-Seq-SPD data/V2X-Seq-SPD
```

## 2. CoVLM-Bench annotations

The benchmark annotation layers -- the question-answer annotations, the neutral
object catalog, and the three-part rationale store -- are **not yet released**;
see the roadmap in the [README](../README.md#roadmap). Once available they are
placed under `data/covlm_bench/`, which is the default the code expects:

```
data/covlm_bench/
  l1_perception/            structured objects, visibility source, motion
  l3_vlm_qa_.../            question-answer annotations
  l4_cot_v3_full_latest/    three-part rationales
```

Every path is also overridable, so no particular layout is required. The
relevant keys are `v2x_root`, `l3_metadata_dir`, and `l4_dir` in the training
configuration, and `--v2x-root` / `--l4-dir` on the index builder.

## 3. Build the training index

The index pairs each frame with its images, planning target, command label, and
rationale target. It is the input to both training and evaluation.

```bash
python -m projects.covla_baseline.data.build_index \
    --l4-dir   data/covlm_bench/l4_cot_v3_full_latest \
    --v2x-root data/V2X-Seq-SPD \
    --output   output/covla_baseline/index.jsonl
```

Add `--fail-on-error` to stop on the first unparseable sample instead of
skipping it. Point `index_path` in the configuration at the resulting file.

## Splits

The benchmark covers 2,196 paired frames. Cooperative planning is evaluated on
654 validation examples, predicting six waypoints over a three-second horizon
from targets derived from recorded ego motion. The train/validation split is
fixed and ships with the annotation release.
