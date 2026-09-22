# Installation

CoVLM-Bench uses **two separate environments**. The CoVLM-Drive baseline needs a
recent PyTorch and `transformers`; the compared V2X planners are built on
`mmdetection3d` and need an older, pinned stack. Do not mix them.

## 1. CoVLM-Drive (cooperative planning)

This is the environment the reported CP runs used.

```bash
conda create -n covlm python=3.10 -y
conda activate covlm

# Install PyTorch matching your CUDA build first. The reported runs used
# torch 2.8.0 with CUDA 12.8:
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

Verify:

```bash
python -c "import torch, transformers, peft; \
print(torch.__version__, transformers.__version__, peft.__version__)"
# 2.8.0+cu128 4.57.1 0.19.1
```

Run every command from the repository root, so that `projects.covla_baseline`
resolves as a package:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
```

### Hardware

The standard CP configuration trains a Qwen3-VL-8B backbone with LoRA and two
structured heads. Reported inference figures use batch size one on a single
RTX 4090: 223.1 ms per sample, 4.48 FPS, 17.08 GiB peak memory. Training used a
micro-batch of one with gradient accumulation to a global batch of eight.

### Backbone weights

Backbone checkpoints are **not** included. Download them yourself and point
`model_name_or_path` in the configuration at the local directory. The reported
runs used `Qwen3-VL-8B-Instruct`, `Qwen2.5-VL-7B-Instruct`, `InternVL3-8B-hf`,
`llava-onevision-qwen2-7b-ov-hf`, and `Idefics3-8B-Llama3`.

## 2. Compared V2X planners (UniV2X, UniMM-V2X, MAP)

These follow the upstream
[UniV2X](https://github.com/AIR-THU/UniV2X) installation, which is based on
UniAD and mmdetection3d. Create a **separate** environment:

```bash
conda create -n univ2x python=3.8 -y
conda activate univ2x
# Install torch, mmcv-full, mmdet and mmdet3d versions matching your CUDA build,
# following the upstream UniV2X instructions, then:
pip install -r requirements-univ2x.txt
```

See [BASELINES.md](BASELINES.md) for running them.

## Next

- [DATA_PREP.md](DATA_PREP.md) -- obtaining the data and building the index.
- [TRAIN_EVAL.md](TRAIN_EVAL.md) -- training and evaluating CoVLM-Drive.
