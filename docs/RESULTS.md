# Cooperative planning results

Full version of the table summarized in the [README](../README.md#cooperative-planning-results). Values are reproduced from [`../results/cp_results.json`](../results/cp_results.json).

Evaluated on 654 validation examples; 6 waypoints over a 3-second horizon.

## Definitions

- `l2_m` -- mean Euclidean distance between predicted and reference waypoints at the given horizon.
- `fde_m` -- final displacement error, i.e. L2 at 3.0 s.
- `command_accuracy` -- command classification accuracy.
- `balanced_accuracy` -- recall averaged over the command classes present in the validation split.
- `parse_rate_pct` -- percentage of zero-shot responses containing a valid command and six waypoints.
- `latency_ms` -- structured planning inference, batch size one on an RTX 4090.
- `memory_gib` -- peak inference memory.

`--` denotes an unreported or inapplicable entry. &dagger; marks our protocol-aligned reproduction.

## Task-trained V2X planners

| Method | 0.5 s | 1.0 s | 1.5 s | 2.0 s | 2.5 s | **FDE** | Cmd Acc. | Bal. Acc. | Parse (%) | Lat. (ms) | FPS | Mem. (GiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UniV2X | 1.245 | 2.079 | 2.888 | 3.831 | 4.843 | 5.861 | -- | -- | -- | 812.7 | 1.23 | 2.86 |
| UniMM-V2X&dagger; | 0.893 | 1.677 | 2.528 | 3.427 | 4.336 | 5.357 | 0.8165 | 0.6604 | -- | 523.9 | 1.91 | 3.29 |
| MAP&dagger; | 0.601 | 1.408 | 2.297 | 3.233 | 4.200 | 5.240 | 0.8394 | 0.7297 | -- | 588.6 | 1.70 | 2.71 |

## CoVLM-Drive backbone variants

| Method | 0.5 s | 1.0 s | 1.5 s | 2.0 s | 2.5 s | **FDE** | Cmd Acc. | Bal. Acc. | Parse (%) | Lat. (ms) | FPS | Mem. (GiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-8B (standard CP) | 0.812 | 1.500 | 2.283 | 3.119 | 3.998 | 4.911 | 0.7706 | 0.5661 | -- | 223.1 | 4.48 | 17.08 |
| Qwen3-VL-8B + ego status | 0.798 | 1.515 | 2.330 | 3.207 | 4.136 | 5.126 | 0.7401 | 0.5497 | -- | 180.1 | 5.55 | 17.11 |
| Qwen2.5-VL-7B | 0.832 | 1.530 | 2.364 | 3.241 | 4.165 | 5.120 | 0.7569 | 0.5355 | -- | 273.8 | 3.65 | 16.23 |
| InternVL3-8B | 0.811 | 1.493 | 2.310 | 3.145 | 4.077 | 5.076 | 0.7324 | 0.5557 | -- | 305.1 | 3.28 | 16.08 |
| LLaVA-OneVision-7B | 0.825 | 1.555 | 2.388 | 3.284 | 4.238 | 5.272 | 0.7385 | 0.5309 | -- | 280.9 | 3.56 | 16.19 |
| Idefics3-8B | 0.876 | 1.691 | 2.584 | 3.525 | 4.519 | 5.538 | 0.7492 | 0.5850 | -- | 1070.6 | 0.93 | 19.24 |

## Zero-shot VLMs

| Method | 0.5 s | 1.0 s | 1.5 s | 2.0 s | 2.5 s | **FDE** | Cmd Acc. | Bal. Acc. | Parse (%) | Lat. (ms) | FPS | Mem. (GiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-4B | 3.853 | 7.251 | 10.986 | 14.827 | 18.754 | 22.751 | 0.6055 | 0.2235 | 96.3 | -- | -- | -- |
| Qwen3-VL-8B | 1.911 | 3.640 | 5.444 | 7.311 | 9.240 | 11.227 | 0.6468 | 0.2387 | 100.0 | -- | -- | -- |
| Qwen2.5-VL-3B | 3.996 | 6.784 | 10.172 | 13.582 | 17.000 | 20.426 | 0.2202 | 0.0813 | 34.4 | -- | -- | -- |
| Qwen2.5-VL-7B | 3.803 | 7.132 | 10.618 | 14.153 | 17.711 | 21.284 | 0.6774 | 0.2500 | 100.0 | -- | -- | -- |
| InternVL3-8B | 3.581 | 6.828 | 10.108 | 13.397 | 16.698 | 20.007 | 0.6774 | 0.2500 | 100.0 | -- | -- | -- |
| LLaVA-OneVision-7B | 5.324 | 77.449 | 152.437 | 227.498 | 302.597 | 377.721 | 0.6774 | 0.2500 | 100.0 | -- | -- | -- |
| Idefics3-8B | 9.099 | 10.854 | 14.049 | 17.434 | 21.606 | 25.945 | 0.6606 | 0.2438 | 99.8 | -- | -- | -- |

## Reading the table

The standard CP configuration reaches a lower FDE than the compared V2X planners and runs with lower latency, while MAP retains the stronger command metrics and the smaller memory footprint. Zero-shot VLMs prompted to emit a command and six waypoints remain far from the trained planners; the parse column reports how often their response contained a usable command and six waypoints at all.

Trajectory accuracy against recorded ego motion is an open-loop measurement and is not a closed-loop safety evaluation.

Question-answering results follow with the CDQA release; see the [roadmap](../README.md#roadmap).
