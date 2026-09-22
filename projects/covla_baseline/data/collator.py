"""HuggingFace/Qwen processor collator for multi-image CoVLM samples."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
from PIL import Image


def build_qwen_view_masks(
    input_ids: torch.Tensor,
    *,
    vision_start_token_id: int,
    vision_end_token_id: int,
    image_token_id: int,
    view_order: str = "ego_infra",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return semantic ego/infra masks for the first two Qwen visual spans.

    Only expanded image tokens are selected. Delimiters stay out of the visual
    memory so downstream readouts cannot learn a shortcut from delimiters.
    """
    order = str(view_order or "ego_infra").lower()
    if order not in {"ego_infra", "infra_ego"}:
        raise ValueError("view_order must be ego_infra or infra_ego")
    ego_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    infra_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row in range(int(input_ids.shape[0])):
        ids = input_ids[row]
        starts = torch.nonzero(ids == int(vision_start_token_id), as_tuple=False).flatten().tolist()
        segments: List[List[int]] = []
        for start in starts:
            end_offsets = torch.nonzero(
                ids[int(start) + 1 :] == int(vision_end_token_id),
                as_tuple=False,
            ).flatten()
            if end_offsets.numel() == 0:
                continue
            end = int(start) + 1 + int(end_offsets[0])
            positions = torch.nonzero(
                ids[int(start) + 1 : end] == int(image_token_id),
                as_tuple=False,
            ).flatten()
            segments.append([int(start) + 1 + int(pos) for pos in positions.tolist()])
        if segments:
            first_mask = infra_mask if order == "infra_ego" else ego_mask
            first_mask[row, segments[0]] = True
        if len(segments) > 1:
            second_mask = ego_mask if order == "infra_ego" else infra_mask
            second_mask[row, segments[1]] = True
    return ego_mask, infra_mask


def build_token_subsequence_mask(
    input_ids: torch.Tensor,
    token_ids: Sequence[int],
    *,
    tail_tokens: int = 0,
) -> torch.Tensor:
    """Locate an exact token subsequence and mask its requested suffix."""
    query = [int(token_id) for token_id in token_ids]
    if not query:
        raise ValueError("task query must tokenize to at least one token")
    keep = len(query) if int(tail_tokens) <= 0 else min(len(query), int(tail_tokens))
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row in range(int(input_ids.shape[0])):
        ids = [int(token_id) for token_id in input_ids[row].tolist()]
        starts = [
            start
            for start in range(0, len(ids) - len(query) + 1)
            if ids[start : start + len(query)] == query
        ]
        if not starts:
            raise ValueError(f"task-query token subsequence was not found in batch row {row}")
        start = starts[-1] + len(query) - keep
        mask[row, start : start + keep] = True
    return mask


class CoVLACollator:
    """Collate CoVLM samples into Qwen-VL processor batches.

    When language targets are enabled, the collator masks prompt tokens as
    ``-100`` so language-model loss is applied only to the L4 target text.
    """

    def __init__(
        self,
        processor: Any,
        mode: str = "v2x_image",
        max_length: Optional[int] = None,
        image_max_pixels: Optional[int] = None,
        train_on_prompt: bool = False,
        include_targets: bool = True,
        sample_passthrough_keys: Optional[Sequence[str]] = None,
        task_query_text: Optional[str] = None,
        task_query_tail_tokens: int = 0,
        view_order: str = "ego_infra",
        ego_image_max_pixels: Optional[int] = None,
        infra_image_max_pixels: Optional[int] = None,
        causal_ego_anchor: bool = False,
    ) -> None:
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.mode = mode
        self.max_length = max_length
        self.image_max_pixels = image_max_pixels
        self.ego_image_max_pixels = ego_image_max_pixels
        self.infra_image_max_pixels = infra_image_max_pixels
        self.train_on_prompt = train_on_prompt
        self.include_targets = include_targets
        self.sample_passthrough_keys = (
            [str(key) for key in sample_passthrough_keys]
            if sample_passthrough_keys is not None
            else None
        )
        self.task_query_text = str(task_query_text or "")
        self.task_query_tail_tokens = int(task_query_tail_tokens)
        self.view_order = str(view_order or "ego_infra").lower()
        if self.view_order not in {"ego_infra", "infra_ego"}:
            raise ValueError("view_order must be ego_infra or infra_ego")
        self.causal_ego_anchor = bool(causal_ego_anchor)
        if self.causal_ego_anchor and (
            self.mode != "v2x_image" or self.view_order != "ego_infra"
        ):
            raise ValueError(
                "causal_ego_anchor requires mode=v2x_image and view_order=ego_infra"
            )
        self.task_query_token_ids: List[int] = []
        if self.task_query_text:
            encoded_query = self.tokenizer(
                self.task_query_text.strip(),
                add_special_tokens=False,
                return_attention_mask=False,
            )
            query_ids = encoded_query["input_ids"]
            if query_ids and isinstance(query_ids[0], list):
                query_ids = query_ids[0]
            self.task_query_token_ids = [int(token_id) for token_id in query_ids]
        self.vision_start_token_id = self._token_id("<|vision_start|>", 151652)
        self.vision_end_token_id = self._token_id("<|vision_end|>", 151653)
        self.image_token_id = self._token_id("<|image_pad|>", 151655)

    def _token_id(self, token: str, fallback: int) -> int:
        convert = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        if convert is None:
            return int(fallback)
        token_id = convert(token)
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        if token_id is None or token_id == unk_id or int(token_id) < 0:
            return int(fallback)
        return int(token_id)

    def _open_images(self, sample: Dict[str, Any]) -> List[Image.Image]:
        if sample.get("images"):
            images = list(sample["images"])
        else:
            image_paths = list(sample.get("image_paths", []))
            if str(sample.get("infra_condition", "")).lower() == "blank" and len(image_paths) >= 2:
                ego = Image.open(image_paths[0]).convert("RGB")
                infra_ref = Image.open(image_paths[1]).convert("RGB")
                blank = Image.new("RGB", infra_ref.size, (128, 128, 128))
                images = [ego, blank]
            else:
                images = [Image.open(path).convert("RGB") for path in image_paths]
        if self.mode == "v2x_image" and self.view_order == "infra_ego" and len(images) >= 2:
            return [images[1], images[0], *images[2:]]
        return images

    def _max_pixels_for_position(self, position: int) -> Optional[int]:
        if self.mode != "v2x_image":
            return self.ego_image_max_pixels or self.image_max_pixels
        first_is_infra = self.view_order == "infra_ego"
        is_infra = (position == 0 and first_is_infra) or (position == 1 and not first_is_infra)
        specific = self.infra_image_max_pixels if is_infra else self.ego_image_max_pixels
        return specific or self.image_max_pixels

    @staticmethod
    def _maybe_resize(image: Image.Image, max_pixels: Optional[int]) -> Image.Image:
        if not max_pixels:
            return image
        w, h = image.size
        pixels = w * h
        if pixels <= max_pixels:
            return image
        scale = (float(max_pixels) / float(pixels)) ** 0.5
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return image.resize(new_size)

    def _chat_text(
        self,
        prompt: str,
        target: str = "",
        *,
        ego_anchor_prompt: str = "",
    ) -> str:
        if hasattr(self.processor, "apply_chat_template"):
            if self.causal_ego_anchor:
                if not ego_anchor_prompt:
                    raise ValueError(
                        "causal_ego_anchor requires a non-empty ego_anchor_prompt"
                    )
                # The first turn is byte-for-byte the ordinary ego-only planning
                # prompt through its assistant header.  The second turn appends
                # cooperative context for the LM/CoT objective without changing
                # any hidden state used by the causal planning readout.
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": ego_anchor_prompt},
                        ],
                    },
                    {"role": "assistant", "content": ""},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]
            else:
                content = [{"type": "image"}]
                if self.mode == "v2x_image":
                    content.append({"type": "image"})
                content.append({"type": "text", "text": prompt})
                messages = [{"role": "user", "content": content}]
            rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return rendered + target
        return prompt + "\n\n" + target

    def _anchor_chat_text(self, prompt: str) -> str:
        if not hasattr(self.processor, "apply_chat_template"):
            return prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _encode(self, texts: Sequence[str], nested_images: Sequence[Sequence[Image.Image]]) -> Dict[str, torch.Tensor]:
        kwargs = {"text": list(texts), "padding": True, "return_tensors": "pt"}
        if self.max_length is not None:
            kwargs.update({"max_length": self.max_length, "truncation": True})
        try:
            return self.processor(images=list(nested_images), **kwargs)
        except Exception:
            flat_images: List[Image.Image] = [im for ims in nested_images for im in ims]
            return self.processor(images=flat_images, **kwargs)

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts: List[str] = []
        prompt_texts: List[str] = []
        nested_images: List[List[Image.Image]] = []
        for sample in samples:
            prompt = str(sample["prompt"])
            ego_anchor_prompt = str(sample.get("ego_anchor_prompt", ""))
            if self.task_query_text:
                prompt = prompt.rstrip() + self.task_query_text
            target = sample.get("target_text", "") if self.include_targets else ""
            texts.append(
                self._chat_text(
                    prompt,
                    target,
                    ego_anchor_prompt=ego_anchor_prompt,
                )
            )
            prompt_texts.append(
                self._chat_text(
                    prompt,
                    "",
                    ego_anchor_prompt=ego_anchor_prompt,
                )
            )
            nested_images.append(
                [
                    self._maybe_resize(image, self._max_pixels_for_position(position))
                    for position, image in enumerate(self._open_images(sample))
                ]
            )

        batch = self._encode(texts, nested_images)
        if self.causal_ego_anchor:
            anchor_texts = [
                self._anchor_chat_text(str(sample.get("ego_anchor_prompt", "")))
                for sample in samples
            ]
            anchor_batch = self._encode(
                anchor_texts,
                [[images[0]] for images in nested_images],
            )
            main_attention = batch.get("attention_mask")
            anchor_attention = anchor_batch.get("attention_mask")
            if main_attention is None or anchor_attention is None:
                raise ValueError(
                    "causal_ego_anchor requires processor attention masks"
                )
            causal_anchor_attention_mask = torch.zeros_like(main_attention)
            for row in range(main_attention.shape[0]):
                main_positions = torch.nonzero(
                    main_attention[row].to(dtype=torch.bool),
                    as_tuple=False,
                ).flatten()
                anchor_positions = torch.nonzero(
                    anchor_attention[row].to(dtype=torch.bool),
                    as_tuple=False,
                ).flatten()
                anchor_len = int(anchor_positions.numel())
                if anchor_len == 0 or main_positions.numel() < anchor_len:
                    raise ValueError(
                        "causal ego anchor is empty or truncated from the full prompt"
                    )
                prefix_positions = main_positions[:anchor_len]
                anchor_ids = anchor_batch["input_ids"][row, anchor_positions]
                prefix_ids = batch["input_ids"][row, prefix_positions]
                if not torch.equal(prefix_ids, anchor_ids):
                    raise ValueError(
                        "causal ego anchor prefix does not match the ego-only prompt"
                    )
                causal_anchor_attention_mask[row, prefix_positions] = 1
            batch["causal_anchor_attention_mask"] = causal_anchor_attention_mask
        if self.task_query_token_ids:
            batch["task_query_attention_mask"] = build_token_subsequence_mask(
                batch["input_ids"],
                self.task_query_token_ids,
                tail_tokens=self.task_query_tail_tokens,
            )
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = 0
        prompt_lens = None
        if self.include_targets:
            labels = batch["input_ids"].clone()
            prompt_batch = self._encode(prompt_texts, nested_images)
            prompt_lens = (prompt_batch["input_ids"] != pad_id).sum(dim=1).tolist()
            if not self.train_on_prompt:
                for row, plen in enumerate(prompt_lens):
                    labels[row, : int(plen)] = -100
            batch["labels"] = labels
        if prompt_lens is None:
            if "attention_mask" in batch:
                prompt_lens = batch["attention_mask"].long().sum(dim=1).tolist()
            else:
                prompt_lens = (batch["input_ids"] != pad_id).sum(dim=1).tolist()
        head_attention_mask = torch.zeros_like(batch.get("attention_mask", batch["input_ids"]))
        for row, plen in enumerate(prompt_lens):
            head_attention_mask[row, : min(int(plen), head_attention_mask.shape[1])] = 1
        if "attention_mask" in batch:
            head_attention_mask = head_attention_mask * batch["attention_mask"].to(dtype=head_attention_mask.dtype)
        batch["head_attention_mask"] = head_attention_mask
        if self.mode in {"v2x_image", "ego_only"}:
            ego_view_mask, infra_view_mask = build_qwen_view_masks(
                batch["input_ids"],
                vision_start_token_id=self.vision_start_token_id,
                vision_end_token_id=self.vision_end_token_id,
                image_token_id=self.image_token_id,
                view_order=self.view_order if self.mode == "v2x_image" else "ego_infra",
            )
            batch["ego_view_mask"] = ego_view_mask
            batch["infra_view_mask"] = infra_view_mask
        batch["waypoints"] = torch.tensor([s["waypoints"] for s in samples], dtype=torch.float32)
        batch["command_id"] = torch.tensor([int(s["command_id"]) for s in samples], dtype=torch.long)
        if all("base_waypoints" in s for s in samples):
            batch["base_waypoints"] = torch.tensor([s["base_waypoints"] for s in samples], dtype=torch.float32)
        if all("base_command_logits" in s for s in samples):
            batch["base_command_logits"] = torch.tensor(
                [s["base_command_logits"] for s in samples],
                dtype=torch.float32,
            )
        if all("geov2x_geometry" in s for s in samples):
            batch["geov2x_geometry"] = torch.tensor(
                [s["geov2x_geometry"] for s in samples],
                dtype=torch.float32,
            )
        if all("geov2x_geometry_available" in s for s in samples):
            batch["geov2x_geometry_available"] = torch.tensor(
                [bool(s["geov2x_geometry_available"]) for s in samples],
                dtype=torch.bool,
            )
        if all("object_evidence_tokens" in s for s in samples):
            batch["object_evidence_tokens"] = torch.tensor(
                [s["object_evidence_tokens"] for s in samples],
                dtype=torch.float32,
            )
        if all("object_evidence_mask" in s for s in samples):
            batch["object_evidence_mask"] = torch.tensor(
                [s["object_evidence_mask"] for s in samples],
                dtype=torch.float32,
            )
        if all("v2x_relevance_prior" in s for s in samples):
            batch["v2x_relevance_prior"] = torch.tensor(
                [float(s["v2x_relevance_prior"]) for s in samples],
                dtype=torch.float32,
            )
        if all("route_relevance_prior" in s for s in samples):
            batch["route_relevance_prior"] = torch.tensor(
                [float(s["route_relevance_prior"]) for s in samples],
                dtype=torch.float32,
            )
        if all("lateral_shape_residual_scale" in s for s in samples):
            batch["lateral_shape_residual_scale"] = torch.tensor(
                [float(s["lateral_shape_residual_scale"]) for s in samples],
                dtype=torch.float32,
            )
        if all("route_right_turn_shape" in s for s in samples):
            batch["route_right_turn_shape"] = torch.tensor(
                [bool(s["route_right_turn_shape"]) for s in samples],
                dtype=torch.bool,
            )
        if all("route_risk_target" in s for s in samples):
            batch["route_risk_target"] = torch.tensor(
                [s["route_risk_target"] for s in samples],
                dtype=torch.float32,
            )
        if all("teacher_waypoints" in s for s in samples):
            batch["teacher_waypoints"] = torch.tensor(
                [s["teacher_waypoints"] for s in samples],
                dtype=torch.float32,
            )
        if all("teacher_command_logits" in s for s in samples):
            batch["teacher_command_logits"] = torch.tensor(
                [s["teacher_command_logits"] for s in samples],
                dtype=torch.float32,
            )
        if all("teacher_available" in s for s in samples):
            batch["teacher_available"] = torch.tensor(
                [bool(s["teacher_available"]) for s in samples],
                dtype=torch.bool,
            )
        if self.sample_passthrough_keys is None:
            batch["samples"] = samples
        else:
            batch["samples"] = [
                {key: sample.get(key) for key in self.sample_passthrough_keys if key in sample}
                for sample in samples
            ]
        return batch
