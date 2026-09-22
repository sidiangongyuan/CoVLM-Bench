"""Qwen-VL backbone with waypoint and command heads."""
from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn

from projects.covla_baseline.data.dataset import COMMAND2ID
from projects.covla_baseline.data.evidence import FEATURE_DIM as L1_EVIDENCE_FEATURE_DIM
from projects.covla_baseline.data.geometry import GEOV2X_GEOMETRY_DIM
from .action_head import (
    ActionHead,
    CommandHead,
    EvidenceDirectedResidualFusionHead,
    EvidenceResidualFusionHead,
    GeoV2XResidualHead,
    PredictedEvidenceResidualHead,
    PredictedEvidenceTokenResidualHead,
    ResidualGateHead,
    TaskQueryPooler,
    WaypointShapeCommandHead,
)


def _import_transformers():
    try:
        import transformers  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "transformers is required for QwenVLABaseline. Install a Qwen-VL compatible "
            "version of transformers, then rerun smoke_train.py. Original error: " + str(exc)
        )
    return transformers


def _select_auto_model(transformers_module: Any) -> Any:
    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"):
        cls = getattr(transformers_module, name, None)
        if cls is not None:
            return cls
    raise ImportError("No suitable HuggingFace AutoModel class found for a Qwen-VL model.")


class QwenVLABaseline(nn.Module):
    """Single-stage Qwen-VL CoVLM baseline model.

    Forward returns LM, waypoint, command, and total losses while keeping the
    original Qwen backbone output path intact.
    """

    def __init__(
        self,
        model_name_or_path: str,
        lambda_lm: float = 1.0,
        lambda_wp: float = 1.0,
        lambda_cmd: float = 0.2,
        use_lora: bool = False,
        use_qlora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        gradient_checkpointing: bool = True,
        bf16: bool = True,
        device_map: Optional[str] = None,
        trust_remote_code: bool = True,
        num_commands: int = len(COMMAND2ID),
        action_loss_type: str = "smooth_l1",
        lambda_fde: float = 0.0,
        lambda_residual_l2: float = 0.0,
        lambda_gate_l1: float = 0.0,
        lambda_gate_prior: float = 0.0,
        lambda_direct_residual: float = 0.0,
        lambda_gate_benefit: float = 0.0,
        command_class_weights: Optional[torch.Tensor] = None,
        command_class_weight_reduction: str = "mean",
        command_class_weight_normalizer: float = 1.0,
        command_head_mode: str = "standard",
        command_shape_hidden_size: int = 64,
        head_pooling: str = "last",
        fusion_mode: Optional[str] = None,
        residual_gate_init_bias: float = -3.0,
        evidence_feature_dim: int = L1_EVIDENCE_FEATURE_DIM,
        evidence_hidden_size: int = 256,
        evidence_layers: int = 2,
        evidence_heads: int = 4,
        evidence_gate_mode: str = "per_step",
        safety_mixer: Optional[str] = None,
        base_shape_guard_alpha: float = 0.2,
        base_shape_guard_waypoints_only: bool = True,
        base_shape_guard_source: str = "frozen_ego_base_waypoints_only",
        turn_right_guard_alpha: float = 0.05,
        turn_right_guard_prior_threshold: float = 0.9,
        turn_right_guard_residual_threshold: float = 0.1,
        right_turn_shape_guard_alpha: float = 0.4,
        right_turn_shape_guard_prior_threshold: float = 0.65,
        lambda_teacher_wp: float = 0.0,
        lambda_teacher_cmd: float = 0.0,
        geometry_dim: int = GEOV2X_GEOMETRY_DIM,
        geov2x_hidden_size: int = 256,
        geov2x_heads: int = 4,
        lambda_counterfactual_base: float = 1.0,
        lambda_counterfactual_gate: float = 0.1,
        lambda_counterfactual_cmd: float = 0.2,
        predicted_evidence_dim: int = 10,
        predicted_evidence_hidden_size: int = 256,
        lambda_predicted_evidence: float = 0.0,
        task_query_dim: int = 256,
        task_query_heads: int = 4,
        task_query_dropout: float = 0.0,
        lambda_task_query_ego_wp: float = 0.0,
        lambda_task_query_ego_cmd: float = 0.0,
        lambda_task_query_consistency_wp: float = 0.0,
        lambda_task_query_consistency_cmd: float = 0.0,
        lambda_task_query_infra_attention: float = 0.0,
        vision_encode_per_view: bool = False,
    ) -> None:
        super().__init__()
        model_path = Path(model_name_or_path)
        if not model_path.exists() and not ("/" in model_name_or_path and not model_name_or_path.startswith("/")):
            raise FileNotFoundError(
                f"model_name_or_path does not exist: {model_name_or_path}. "
                "Edit the YAML config to point to a local Qwen3-VL/Qwen-VL checkpoint."
            )
        transformers = _import_transformers()
        auto_model_cls = _select_auto_model(transformers)

        dtype = torch.bfloat16 if bf16 and torch.cuda.is_available() else None
        load_kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if dtype is not None:
            load_kwargs["dtype"] = dtype
        if device_map:
            load_kwargs["device_map"] = device_map
        if use_qlora:
            load_kwargs["load_in_4bit"] = True

        adapter_config = model_path / "adapter_config.json" if model_path.exists() else None
        if adapter_config is not None and adapter_config.exists():
            try:
                from peft import PeftConfig, PeftModel  # type: ignore
            except ImportError as exc:
                raise ImportError("A PEFT adapter checkpoint was provided, but peft is not installed.") from exc
            peft_cfg = PeftConfig.from_pretrained(model_name_or_path)
            base_path = getattr(peft_cfg, "base_model_name_or_path", None)
            if not base_path:
                raise RuntimeError(f"Cannot resolve base_model_name_or_path from {adapter_config}")
            base_model = auto_model_cls.from_pretrained(base_path, **load_kwargs)
            self.backbone = PeftModel.from_pretrained(base_model, model_name_or_path, is_trainable=bool(use_lora))
        else:
            self.backbone = auto_model_cls.from_pretrained(model_name_or_path, **load_kwargs)
        if gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
            enable_input_grads = getattr(self.backbone, "enable_input_require_grads", None)
            if callable(enable_input_grads):
                enable_input_grads()
        if not (adapter_config is not None and adapter_config.exists()):
            self._maybe_enable_lora(use_lora, use_qlora, lora_r, lora_alpha, lora_dropout)
        self.vision_encode_per_view = bool(vision_encode_per_view)
        if self.vision_encode_per_view:
            self._install_per_view_image_encoding()

        hidden_size = self._hidden_size()
        self.fusion_mode = str(fusion_mode or "standard").lower()
        if self.fusion_mode not in {
            "standard",
            "none",
            "ego_residual",
            "geov2x_residual",
            "predicted_evidence_residual",
            "predicted_evidence_token_residual",
            "evidence_residual",
            "evidence_directed_residual",
        }:
            raise ValueError(
                f"Unsupported fusion_mode={fusion_mode!r}; expected ego_residual, evidence_residual, "
                "evidence_directed_residual, or absent."
            )
        self.action_head = ActionHead(hidden_size=hidden_size, num_steps=6, loss_type=action_loss_type)
        self.command_head = CommandHead(hidden_size=hidden_size, num_commands=num_commands)
        self.task_query_pooler: Optional[TaskQueryPooler] = None
        self.command_head_mode = str(command_head_mode or "standard").lower()
        if self.command_head_mode not in {"standard", "waypoint_shape_fusion"}:
            raise ValueError(
                "command_head_mode must be 'standard' or 'waypoint_shape_fusion', "
                f"got {command_head_mode!r}"
            )
        self.shape_command_head: Optional[WaypointShapeCommandHead] = None
        self.command_shape_hidden_size = int(command_shape_hidden_size)
        if self.command_head_mode == "waypoint_shape_fusion":
            self.shape_command_head = WaypointShapeCommandHead(
                hidden_size=hidden_size,
                num_commands=num_commands,
                num_steps=6,
                shape_hidden_size=self.command_shape_hidden_size,
            )
        self.residual_gate_head: Optional[ResidualGateHead] = None
        self.geov2x_residual_head: Optional[GeoV2XResidualHead] = None
        self.predicted_evidence_residual_head: Optional[PredictedEvidenceResidualHead] = None
        self.predicted_evidence_token_residual_head: Optional[PredictedEvidenceTokenResidualHead] = None
        self.evidence_residual_head: Optional[EvidenceResidualFusionHead] = None
        self.evidence_directed_residual_head: Optional[EvidenceDirectedResidualFusionHead] = None
        if self.fusion_mode == "ego_residual":
            self.action_head.zero_init_output()
            self.command_head.zero_init_output()
            self.residual_gate_head = ResidualGateHead(hidden_size=hidden_size, init_bias=residual_gate_init_bias)
        if self.fusion_mode == "geov2x_residual":
            self.geov2x_residual_head = GeoV2XResidualHead(
                qwen_hidden_size=hidden_size,
                geometry_dim=int(geometry_dim),
                num_commands=num_commands,
                decoder_hidden_size=int(geov2x_hidden_size),
                num_heads=int(geov2x_heads),
                gate_init_bias=float(residual_gate_init_bias),
            )
        if self.fusion_mode == "predicted_evidence_residual":
            self.predicted_evidence_residual_head = PredictedEvidenceResidualHead(
                qwen_hidden_size=hidden_size,
                geometry_dim=int(geometry_dim),
                num_commands=num_commands,
                risk_dim=int(predicted_evidence_dim),
                hidden_size=int(predicted_evidence_hidden_size),
                gate_init_bias=float(residual_gate_init_bias),
            )
        if self.fusion_mode == "predicted_evidence_token_residual":
            self.predicted_evidence_token_residual_head = PredictedEvidenceTokenResidualHead(
                qwen_hidden_size=hidden_size,
                geometry_dim=int(geometry_dim),
                num_commands=num_commands,
                risk_dim=int(predicted_evidence_dim),
                hidden_size=int(predicted_evidence_hidden_size),
                num_heads=int(geov2x_heads),
                gate_init_bias=float(residual_gate_init_bias),
            )
        if self.fusion_mode == "evidence_residual":
            self.evidence_residual_head = EvidenceResidualFusionHead(
                qwen_hidden_size=hidden_size,
                evidence_feature_dim=int(evidence_feature_dim),
                evidence_hidden_size=int(evidence_hidden_size),
                evidence_layers=int(evidence_layers),
                evidence_heads=int(evidence_heads),
                num_commands=num_commands,
                gate_init_bias=float(residual_gate_init_bias),
                gate_mode=str(evidence_gate_mode),
            )
        if self.fusion_mode == "evidence_directed_residual":
            if str(evidence_gate_mode).lower() not in {"per_step", "per_horizon"}:
                raise ValueError("evidence_directed_residual requires evidence_gate_mode=per_horizon/per_step")
            self.evidence_directed_residual_head = EvidenceDirectedResidualFusionHead(
                qwen_hidden_size=hidden_size,
                evidence_feature_dim=int(evidence_feature_dim),
                evidence_hidden_size=int(evidence_hidden_size),
                evidence_layers=int(evidence_layers),
                evidence_heads=int(evidence_heads),
                num_commands=num_commands,
                gate_init_bias=float(residual_gate_init_bias),
            )
        self.lambda_lm = float(lambda_lm)
        self.lambda_wp = float(lambda_wp)
        self.lambda_cmd = float(lambda_cmd)
        self.lambda_fde = float(lambda_fde)
        self.lambda_residual_l2 = float(lambda_residual_l2)
        self.lambda_gate_l1 = float(lambda_gate_l1)
        self.lambda_gate_prior = float(lambda_gate_prior)
        self.lambda_direct_residual = float(lambda_direct_residual)
        self.lambda_gate_benefit = float(lambda_gate_benefit)
        self.residual_gate_init_bias = float(residual_gate_init_bias)
        self.evidence_feature_dim = int(evidence_feature_dim)
        self.evidence_hidden_size = int(evidence_hidden_size)
        self.evidence_layers = int(evidence_layers)
        self.evidence_heads = int(evidence_heads)
        self.evidence_gate_mode = str(evidence_gate_mode)
        self.safety_mixer = str(safety_mixer or "").lower()
        if self.safety_mixer not in {"", "none", "base_shape_guard", "turn_right_lateral_guard"}:
            raise ValueError(f"Unsupported safety_mixer={safety_mixer!r}")
        self.base_shape_guard_alpha = float(base_shape_guard_alpha)
        self.base_shape_guard_waypoints_only = bool(base_shape_guard_waypoints_only)
        if self.safety_mixer in {"base_shape_guard", "turn_right_lateral_guard"} and not self.base_shape_guard_waypoints_only:
            raise ValueError(f"{self.safety_mixer} currently supports waypoints-only mixing.")
        self.base_shape_guard_source = str(base_shape_guard_source or "frozen_ego_base_waypoints_only")
        self.turn_right_guard_alpha = float(turn_right_guard_alpha)
        self.turn_right_guard_prior_threshold = float(turn_right_guard_prior_threshold)
        self.turn_right_guard_residual_threshold = float(turn_right_guard_residual_threshold)
        self.right_turn_shape_guard_alpha = float(right_turn_shape_guard_alpha)
        self.right_turn_shape_guard_prior_threshold = float(right_turn_shape_guard_prior_threshold)
        self.lambda_teacher_wp = float(lambda_teacher_wp)
        self.lambda_teacher_cmd = float(lambda_teacher_cmd)
        self.geometry_dim = int(geometry_dim)
        self.geov2x_hidden_size = int(geov2x_hidden_size)
        self.geov2x_heads = int(geov2x_heads)
        self.lambda_counterfactual_base = float(lambda_counterfactual_base)
        self.lambda_counterfactual_gate = float(lambda_counterfactual_gate)
        self.lambda_counterfactual_cmd = float(lambda_counterfactual_cmd)
        self.predicted_evidence_dim = int(predicted_evidence_dim)
        self.predicted_evidence_hidden_size = int(predicted_evidence_hidden_size)
        self.lambda_predicted_evidence = float(lambda_predicted_evidence)
        self.head_pooling = str(head_pooling or "last").lower()
        if self.head_pooling not in {
            "last",
            "mean",
            "task_query",
            "query_mean",
            "ego_view_mean",
            "infra_memory_mean",
            "text_memory_mean",
            "causal_ego_anchor_mean",
        }:
            raise ValueError(
                "Unsupported head_pooling={!r}; expected 'last', 'mean', "
                "'task_query', 'query_mean', 'ego_view_mean', or "
                "'infra_memory_mean', 'text_memory_mean', or "
                "'causal_ego_anchor_mean'.".format(head_pooling)
            )
        self.task_query_dim = int(task_query_dim)
        self.task_query_heads = int(task_query_heads)
        self.task_query_dropout = float(task_query_dropout)
        self.lambda_task_query_ego_wp = float(lambda_task_query_ego_wp)
        self.lambda_task_query_ego_cmd = float(lambda_task_query_ego_cmd)
        self.lambda_task_query_consistency_wp = float(lambda_task_query_consistency_wp)
        self.lambda_task_query_consistency_cmd = float(lambda_task_query_consistency_cmd)
        self.lambda_task_query_infra_attention = float(lambda_task_query_infra_attention)
        if self.head_pooling == "task_query":
            if self.fusion_mode not in {"standard", "none"}:
                raise ValueError("head_pooling=task_query currently requires the standard fusion path")
            self.task_query_pooler = TaskQueryPooler(
                hidden_size=hidden_size,
                query_dim=self.task_query_dim,
                num_heads=self.task_query_heads,
                dropout=self.task_query_dropout,
            )
        if command_class_weights is None:
            self.register_buffer("command_class_weights", None, persistent=False)
        else:
            weights = command_class_weights.detach().float().view(-1)
            if weights.numel() != num_commands:
                raise ValueError(f"Expected {num_commands} command weights, got {weights.numel()}")
            self.register_buffer("command_class_weights", weights, persistent=True)
        self.command_class_weight_reduction = str(command_class_weight_reduction or "mean").lower()
        if self.command_class_weight_reduction not in {"mean", "fixed_normalizer"}:
            raise ValueError(
                "command_class_weight_reduction must be 'mean' or 'fixed_normalizer', "
                f"got {command_class_weight_reduction!r}"
            )
        self.command_class_weight_normalizer = float(
            command_class_weight_normalizer or 1.0
        )
        if self.command_class_weight_normalizer <= 0:
            raise ValueError("command_class_weight_normalizer must be positive.")

    @staticmethod
    def _encode_images_per_view(
        visual: Any,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], list[torch.Tensor]]:
        """Encode each camera view independently, then restore Qwen's layout.

        Qwen3-VL normally concatenates every image before the visual encoder.
        Kernel selection can then perturb the first view when a second view is
        added, even though visual attention uses per-image sequence boundaries.
        Per-view execution removes that unintended numerical coupling without
        changing weights, tokens, or the language-model architecture.
        """
        if image_grid_thw.ndim != 2 or image_grid_thw.shape[0] == 0:
            raise ValueError("image_grid_thw must contain at least one view")
        typed_pixels = pixel_values.to(dtype=visual.dtype)
        patch_counts = [int(value) for value in image_grid_thw.prod(-1).tolist()]
        if sum(patch_counts) != int(typed_pixels.shape[0]):
            raise ValueError(
                "pixel patch count does not match image_grid_thw for per-view encoding"
            )
        pixel_chunks = torch.split(typed_pixels, patch_counts, dim=0)
        image_features = []
        deep_features_by_level: Optional[list[list[torch.Tensor]]] = None
        for pixels, grid in zip(pixel_chunks, image_grid_thw):
            merged, deep_features = visual(pixels, grid.unsqueeze(0))
            image_features.append(merged)
            if deep_features_by_level is None:
                deep_features_by_level = [[] for _ in deep_features]
            if len(deep_features) != len(deep_features_by_level):
                raise ValueError("inconsistent DeepStack levels across views")
            for level, features in zip(deep_features_by_level, deep_features):
                level.append(features)
        if deep_features_by_level is None:
            raise ValueError("visual encoder returned no per-view features")
        return tuple(image_features), [
            torch.cat(level, dim=0) for level in deep_features_by_level
        ]

    def _install_per_view_image_encoding(self) -> None:
        cores = [
            module
            for module in self.backbone.modules()
            if type(module).__name__ == "Qwen3VLModel"
            and hasattr(module, "visual")
            and hasattr(module, "get_image_features")
        ]
        if len(cores) != 1:
            raise RuntimeError(
                "vision_encode_per_view requires exactly one Qwen3VLModel core; "
                f"found {len(cores)}"
            )
        core = cores[0]

        def get_image_features_per_view(
            core_model: Any,
            pixel_values: torch.Tensor,
            image_grid_thw: Optional[torch.Tensor] = None,
        ) -> tuple[tuple[torch.Tensor, ...], list[torch.Tensor]]:
            if image_grid_thw is None:
                raise ValueError(
                    "vision_encode_per_view requires image_grid_thw"
                )
            return QwenVLABaseline._encode_images_per_view(
                core_model.visual,
                pixel_values,
                image_grid_thw,
            )

        core.get_image_features = types.MethodType(  # type: ignore[method-assign]
            get_image_features_per_view,
            core,
        )

    def _maybe_enable_lora(self, use_lora: bool, use_qlora: bool, r: int, alpha: int, dropout: float) -> None:
        if not (use_lora or use_qlora):
            return
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore
        except ImportError as exc:
            raise ImportError("LoRA/QLoRA requested but peft is not installed. Install peft or set use_lora=false.") from exc
        if use_qlora:
            self.backbone = prepare_model_for_kbit_training(self.backbone)
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        self.backbone = get_peft_model(self.backbone, cfg)
        if hasattr(self.backbone, "print_trainable_parameters"):
            self.backbone.print_trainable_parameters()

    def _hidden_size(self) -> int:
        cfg = getattr(self.backbone, "config", None)
        for name in ("hidden_size", "text_config", "llm_config"):
            obj = getattr(cfg, name, None)
            if isinstance(obj, int):
                return obj
            nested = getattr(obj, "hidden_size", None)
            if isinstance(nested, int):
                return nested
        raise RuntimeError("Cannot infer backbone hidden size from model config.")

    @staticmethod
    def _pool_last_non_pad(hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is None:
            return hidden[:, -1]
        idx = attention_mask.long().sum(dim=1).clamp(min=1) - 1
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch_idx, idx]

    @staticmethod
    def _pool_mean(hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is None:
            return hidden.mean(dim=1)
        mask = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden * mask).sum(dim=1) / denom

    @staticmethod
    def _masked_sample_mean(values: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
        """Average every element belonging to an enabled batch row."""
        mask_shape = [values.shape[0]] + [1] * (values.ndim - 1)
        weights = sample_mask.to(device=values.device, dtype=values.dtype).view(*mask_shape)
        weights = weights.expand_as(values)
        return (values * weights).sum() / weights.sum().clamp_min(1.0)

    @staticmethod
    def _tokens_after_mask(
        prompt_mask: torch.Tensor,
        prefix_mask: torch.Tensor,
        skip_after_prefix: int = 0,
    ) -> torch.Tensor:
        """Select prompt tokens strictly after the final marked prefix token."""
        if int(skip_after_prefix) < 0:
            raise ValueError("skip_after_prefix must be non-negative")
        prompt = prompt_mask.to(dtype=torch.bool)
        prefix = prefix_mask.to(device=prompt.device, dtype=torch.bool)
        if not bool(prefix.any(dim=1).all()):
            raise ValueError("prefix mask must contain at least one token in every row")
        positions = torch.arange(prompt.shape[1], device=prompt.device).unsqueeze(0)
        last_prefix = positions.masked_fill(~prefix, -1).max(dim=1).values
        boundary = last_prefix + int(skip_after_prefix)
        selected = prompt & (positions > boundary.unsqueeze(1))
        if not bool(selected.any(dim=1).all()):
            raise ValueError("no prompt tokens remain after the marked prefix")
        return selected

    @staticmethod
    def _tokens_before_mask(
        prompt_mask: torch.Tensor,
        suffix_mask: torch.Tensor,
        skip_before_suffix: int = 0,
    ) -> torch.Tensor:
        """Select prompt tokens strictly before the first marked suffix token."""
        if int(skip_before_suffix) < 0:
            raise ValueError("skip_before_suffix must be non-negative")
        prompt = prompt_mask.to(dtype=torch.bool)
        suffix = suffix_mask.to(device=prompt.device, dtype=torch.bool)
        if not bool(suffix.any(dim=1).all()):
            raise ValueError("suffix mask must contain at least one token in every row")
        positions = torch.arange(prompt.shape[1], device=prompt.device).unsqueeze(0)
        first_suffix = positions.masked_fill(~suffix, prompt.shape[1]).min(dim=1).values
        boundary = first_suffix - int(skip_before_suffix)
        selected = prompt & (positions < boundary.unsqueeze(1))
        if not bool(selected.any(dim=1).all()):
            raise ValueError("no prompt tokens remain before the marked suffix")
        return selected

    def _pool_for_heads(self, hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.head_pooling == "mean":
            return self._pool_mean(hidden, attention_mask)
        return self._pool_last_non_pad(hidden, attention_mask)

    def _command_cross_entropy(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(logits.device)
        weight = None
        if self.command_class_weights is not None:
            weight = self.command_class_weights.to(device=logits.device, dtype=logits.dtype)
        if weight is None or self.command_class_weight_reduction == "mean":
            return torch.nn.functional.cross_entropy(logits, target, weight=weight)
        raw = torch.nn.functional.cross_entropy(logits, target, reduction="none")
        sample_weight = weight[target].to(device=logits.device, dtype=logits.dtype)
        normalizer = torch.as_tensor(
            self.command_class_weight_normalizer,
            device=logits.device,
            dtype=logits.dtype,
        ).clamp_min(1e-12)
        return (raw * sample_weight).mean() / normalizer

    def _base_shape_guard_alpha(
        self,
        base_waypoints: torch.Tensor,
        lateral_shape_residual_scale: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if lateral_shape_residual_scale is not None:
            scale = lateral_shape_residual_scale.to(
                device=base_waypoints.device,
                dtype=base_waypoints.dtype,
            ).view(-1).clamp(0.0, 1.0)
            guarded = torch.full_like(scale, float(self.base_shape_guard_alpha))
            unguarded = torch.ones_like(scale)
            return torch.where(scale < 0.999, guarded, unguarded)
        lateral = base_waypoints[:, :, 0]
        forward = base_waypoints[:, :, 1]
        lateral_delta = lateral[:, -1] - lateral[:, 0]
        forward_span = forward.max(dim=1).values - forward.min(dim=1).values
        d_lat = lateral[:, 1:] - lateral[:, :-1]
        d_forward = forward[:, 1:] - forward[:, :-1]
        headings = torch.atan2(d_lat, d_forward)
        heading_change_abs = torch.abs(headings[:, -1] - headings[:, 0])
        final_heading_abs = torch.abs(headings[:, -1])
        v1_lat = d_lat[:, :-1]
        v1_forward = d_forward[:, :-1]
        v2_lat = d_lat[:, 1:]
        v2_forward = d_forward[:, 1:]
        cross = v1_lat * v2_forward - v1_forward * v2_lat
        dot = v1_lat * v2_lat + v1_forward * v2_forward
        valid = (
            torch.sqrt(v1_lat.pow(2) + v1_forward.pow(2)) > 1e-6
        ) & (
            torch.sqrt(v2_lat.pow(2) + v2_forward.pow(2)) > 1e-6
        )
        curvature = torch.where(valid, torch.abs(torch.atan2(cross, dot)), torch.zeros_like(cross))
        curvature_sum = curvature.sum(dim=1)
        lateral_forward_ratio = torch.abs(lateral_delta) / forward_span.clamp_min(1e-6)
        protected = (
            (torch.abs(lateral_delta) >= 0.3)
            & (lateral_forward_ratio <= 0.08)
            & (heading_change_abs <= 0.2)
            & (curvature_sum <= 0.2)
            & (forward_span >= 12.0)
            & (final_heading_abs <= 0.08)
        )
        guarded = torch.full_like(forward_span, float(self.base_shape_guard_alpha))
        unguarded = torch.ones_like(forward_span)
        return torch.where(protected, guarded, unguarded)

    def _apply_base_shape_guard(
        self,
        risk_zone_waypoints: torch.Tensor,
        base_waypoints: torch.Tensor,
        lateral_shape_residual_scale: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        alpha = self._base_shape_guard_alpha(base_waypoints, lateral_shape_residual_scale)
        mixed_waypoints = base_waypoints + alpha.view(-1, 1, 1) * (risk_zone_waypoints - base_waypoints)
        return {
            "pred_waypoints": mixed_waypoints,
            "risk_zone_waypoints": risk_zone_waypoints,
            "alpha_shape": alpha,
            "base_shape_guard_applied": alpha < 0.999,
        }

    def _turn_right_lateral_guard_alpha(
        self,
        risk_zone_waypoints: torch.Tensor,
        base_waypoints: torch.Tensor,
        command_logits: torch.Tensor,
        base_command_logits: torch.Tensor,
        route_relevance_prior: Optional[torch.Tensor],
        lateral_shape_residual_scale: Optional[torch.Tensor],
        route_right_turn_shape: Optional[torch.Tensor],
    ) -> torch.Tensor:
        lateral_alpha = self._base_shape_guard_alpha(base_waypoints, lateral_shape_residual_scale)
        if route_right_turn_shape is not None:
            shape_mask = route_right_turn_shape.to(device=base_waypoints.device, dtype=torch.bool).view(-1)
        if route_relevance_prior is None:
            prior = torch.zeros_like(lateral_alpha)
        else:
            prior = route_relevance_prior.to(device=base_waypoints.device, dtype=base_waypoints.dtype).view(-1)
        if route_right_turn_shape is not None:
            weak_route_shape_mask = shape_mask & (prior < float(self.right_turn_shape_guard_prior_threshold))
            shape_alpha = torch.full_like(lateral_alpha, float(self.right_turn_shape_guard_alpha))
            lateral_alpha = torch.where(
                weak_route_shape_mask,
                torch.minimum(lateral_alpha, shape_alpha),
                lateral_alpha,
            )

        base_cmd = base_command_logits.argmax(dim=-1)
        pred_cmd = command_logits.argmax(dim=-1)
        turn_right_id = int(COMMAND2ID["TURN_RIGHT"])
        cmd_mask = (base_cmd == turn_right_id) | (pred_cmd == turn_right_id)
        residual_norm = torch.norm(risk_zone_waypoints[:, -1] - base_waypoints[:, -1], dim=-1)
        turn_guard = (
            cmd_mask
            & (prior >= float(self.turn_right_guard_prior_threshold))
            & (residual_norm >= float(self.turn_right_guard_residual_threshold))
        )
        turn_alpha = torch.full_like(lateral_alpha, float(self.turn_right_guard_alpha))
        return torch.where(turn_guard, torch.minimum(lateral_alpha, turn_alpha), lateral_alpha)

    def _apply_turn_right_lateral_guard(
        self,
        risk_zone_waypoints: torch.Tensor,
        base_waypoints: torch.Tensor,
        command_logits: torch.Tensor,
        base_command_logits: torch.Tensor,
        route_relevance_prior: Optional[torch.Tensor],
        lateral_shape_residual_scale: Optional[torch.Tensor],
        route_right_turn_shape: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        alpha = self._turn_right_lateral_guard_alpha(
            risk_zone_waypoints,
            base_waypoints,
            command_logits,
            base_command_logits,
            route_relevance_prior,
            lateral_shape_residual_scale,
            route_right_turn_shape,
        )
        mixed_waypoints = base_waypoints + alpha.view(-1, 1, 1) * (risk_zone_waypoints - base_waypoints)
        turn_right_id = int(COMMAND2ID["TURN_RIGHT"])
        residual_norm = torch.norm(risk_zone_waypoints[:, -1] - base_waypoints[:, -1], dim=-1)
        prior = (
            torch.zeros_like(alpha)
            if route_relevance_prior is None
            else route_relevance_prior.to(device=alpha.device, dtype=alpha.dtype).view(-1)
        )
        turn_mask = (
            ((base_command_logits.argmax(dim=-1) == turn_right_id) | (command_logits.argmax(dim=-1) == turn_right_id))
            & (prior >= float(self.turn_right_guard_prior_threshold))
            & (residual_norm >= float(self.turn_right_guard_residual_threshold))
        )
        return {
            "pred_waypoints": mixed_waypoints,
            "risk_zone_waypoints": risk_zone_waypoints,
            "alpha_shape": alpha,
            "base_shape_guard_applied": alpha < 0.999,
            "turn_right_guard_applied": turn_mask,
            "turn_right_guard_residual_norm": residual_norm,
        }

    def forward(self, **batch: Any) -> Dict[str, Any]:
        return_analysis_hidden = bool(batch.pop("return_analysis_hidden", False))
        retain_analysis_hidden_grad = bool(batch.pop("retain_analysis_hidden_grad", False))
        waypoints = batch.pop("waypoints", None)
        command_id = batch.pop("command_id", None)
        base_waypoints = batch.pop("base_waypoints", None)
        base_command_logits = batch.pop("base_command_logits", None)
        object_evidence_tokens = batch.pop("object_evidence_tokens", None)
        object_evidence_mask = batch.pop("object_evidence_mask", None)
        geov2x_geometry = batch.pop("geov2x_geometry", None)
        batch.pop("geov2x_geometry_available", None)
        v2x_relevance_prior = batch.pop("v2x_relevance_prior", None)
        route_relevance_prior = batch.pop("route_relevance_prior", None)
        lateral_shape_residual_scale = batch.pop("lateral_shape_residual_scale", None)
        route_right_turn_shape = batch.pop("route_right_turn_shape", None)
        route_risk_target = batch.pop("route_risk_target", None)
        teacher_waypoints = batch.pop("teacher_waypoints", None)
        teacher_command_logits = batch.pop("teacher_command_logits", None)
        teacher_available = batch.pop("teacher_available", None)
        head_attention_mask = batch.pop("head_attention_mask", None)
        causal_anchor_attention_mask = batch.pop(
            "causal_anchor_attention_mask",
            None,
        )
        task_query_attention_mask = batch.pop("task_query_attention_mask", None)
        ego_view_mask = batch.pop("ego_view_mask", None)
        infra_view_mask = batch.pop("infra_view_mask", None)
        samples = batch.pop("samples", None)
        backbone_inputs = {k: v for k, v in batch.items() if k not in {"target_text", "image_paths"}}
        backbone_inputs["output_hidden_states"] = True
        outputs = self.backbone(**backbone_inputs)

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            hidden = getattr(outputs, "last_hidden_state", None)
        else:
            hidden = hidden_states[-1]
        if hidden is None:
            raise RuntimeError("Backbone did not return hidden states; set output_hidden_states=True or use a compatible model.")
        if return_analysis_hidden and retain_analysis_hidden_grad and hidden.requires_grad:
            hidden.retain_grad()
        task_query_output: Optional[Dict[str, torch.Tensor]] = None
        use_task_query_aux = bool(
            self.training
            and samples is not None
            and (
                self.lambda_task_query_ego_wp > 0.0
                or self.lambda_task_query_ego_cmd > 0.0
                or self.lambda_task_query_consistency_wp > 0.0
                or self.lambda_task_query_consistency_cmd > 0.0
                or self.lambda_task_query_infra_attention > 0.0
            )
        )
        if self.head_pooling == "task_query":
            if self.task_query_pooler is None:
                raise RuntimeError("task_query_pooler is missing")
            if ego_view_mask is None or infra_view_mask is None:
                raise ValueError("head_pooling=task_query requires ego_view_mask and infra_view_mask")
            if next(self.task_query_pooler.parameters()).device != hidden.device:
                self.task_query_pooler.to(hidden.device)
            task_query_output = self.task_query_pooler(
                hidden,
                ego_view_mask,
                infra_view_mask,
                return_ego_control=use_task_query_aux,
            )
            pooled = task_query_output["action_state"]
            command_pooled = task_query_output["command_state"]
        elif self.head_pooling == "query_mean":
            if task_query_attention_mask is None:
                raise ValueError(
                    "head_pooling=query_mean requires task_query_attention_mask"
                )
            pooled = self._pool_mean(hidden, task_query_attention_mask)
            command_pooled = pooled
        elif self.head_pooling == "ego_view_mean":
            if ego_view_mask is None:
                raise ValueError("head_pooling=ego_view_mean requires ego_view_mask")
            pooled = self._pool_mean(hidden, ego_view_mask)
            command_pooled = pooled
        elif self.head_pooling == "infra_memory_mean":
            if infra_view_mask is None:
                raise ValueError("head_pooling=infra_memory_mean requires infra_view_mask")
            if not bool(infra_view_mask.to(dtype=torch.bool).any(dim=1).all()):
                raise ValueError(
                    "head_pooling=infra_memory_mean requires infrastructure tokens in every row"
                )
            prompt_mask = (
                head_attention_mask
                if head_attention_mask is not None
                else backbone_inputs.get("attention_mask")
            )
            if prompt_mask is None:
                raise ValueError("head_pooling=infra_memory_mean requires an attention mask")
            # Infrastructure patches remain in the causal Qwen context and can
            # update later prompt states, but do not directly dominate the mean
            # readout merely because they contribute many raw visual tokens.
            memory_readout_mask = prompt_mask.to(dtype=torch.bool) & ~infra_view_mask.to(
                device=prompt_mask.device,
                dtype=torch.bool,
            )
            pooled = self._pool_mean(hidden, memory_readout_mask)
            command_pooled = pooled
        elif self.head_pooling == "text_memory_mean":
            if ego_view_mask is None or infra_view_mask is None:
                raise ValueError(
                    "head_pooling=text_memory_mean requires ego_view_mask and infra_view_mask"
                )
            prompt_mask = (
                head_attention_mask
                if head_attention_mask is not None
                else backbone_inputs.get("attention_mask")
            )
            if prompt_mask is None:
                raise ValueError("head_pooling=text_memory_mean requires an attention mask")
            # Both views are retained as causal memory. Every selected token is
            # strictly after the infrastructure span and therefore can attend
            # to both views, forming a parameter-free semantic bottleneck.
            memory_readout_mask = self._tokens_after_mask(
                prompt_mask,
                infra_view_mask,
                skip_after_prefix=1,
            )
            pooled = self._pool_mean(hidden, memory_readout_mask)
            command_pooled = pooled
        elif self.head_pooling == "causal_ego_anchor_mean":
            if infra_view_mask is None:
                raise ValueError(
                    "head_pooling=causal_ego_anchor_mean requires infra_view_mask"
                )
            if causal_anchor_attention_mask is None:
                raise ValueError(
                    "head_pooling=causal_ego_anchor_mean requires "
                    "causal_anchor_attention_mask"
                )
            if not bool(
                causal_anchor_attention_mask.to(dtype=torch.bool).any(dim=1).all()
            ):
                raise ValueError(
                    "causal_anchor_attention_mask must select tokens in every row"
                )
            pooled = self._pool_mean(hidden, causal_anchor_attention_mask)
            command_pooled = pooled
        else:
            pooled = self._pool_for_heads(
                hidden,
                head_attention_mask
                if head_attention_mask is not None
                else backbone_inputs.get("attention_mask"),
            )
            command_pooled = pooled

        head_device = pooled.device
        if next(self.action_head.parameters()).device != head_device:
            self.action_head.to(head_device)
            self.command_head.to(head_device)
        head_dtype = next(self.action_head.parameters()).dtype
        pooled_for_heads = pooled.to(dtype=head_dtype)
        command_pooled_for_heads = command_pooled.to(dtype=head_dtype)
        residual_gate = None
        residual_l2_loss = zero = pooled_for_heads.sum() * 0.0
        gate_l1_loss = zero
        gate_prior_loss = zero
        direct_residual_loss = zero
        gate_benefit_loss = zero
        counterfactual_base_loss = zero
        counterfactual_gate_loss = zero
        counterfactual_cmd_loss = zero
        predicted_evidence_loss = zero
        task_query_ego_wp_loss = zero
        task_query_ego_cmd_loss = zero
        task_query_consistency_wp_loss = zero
        task_query_consistency_cmd_loss = zero
        task_query_infra_attention_loss = zero
        counterfactual_mask = None
        if samples is not None:
            conditions = [str(sample.get("infra_condition", "normal")).lower() for sample in samples]
            counterfactual_mask = torch.tensor(
                [condition in {"blank", "shuffled"} for condition in conditions],
                device=head_device,
                dtype=torch.bool,
            )
        if self.fusion_mode == "ego_residual":
            raw_waypoints, _ = self.action_head(pooled_for_heads, None)
            raw_command_logits, _ = self.command_head(pooled_for_heads, None)
            if base_waypoints is None or base_command_logits is None:
                raise ValueError("fusion_mode=ego_residual requires base_waypoints and base_command_logits in each batch.")
            if self.residual_gate_head is None:
                raise RuntimeError("residual_gate_head is missing for fusion_mode=ego_residual")
            residual_gate = self.residual_gate_head(pooled_for_heads)
            base_wp = base_waypoints.to(device=raw_waypoints.device, dtype=raw_waypoints.dtype)
            base_logits = base_command_logits.to(device=raw_command_logits.device, dtype=raw_command_logits.dtype)
            pred_waypoints = base_wp + residual_gate.view(-1, 1, 1) * raw_waypoints
            command_logits = base_logits + residual_gate * raw_command_logits
            residual_l2_loss = torch.mean(raw_waypoints.pow(2))
            gate_l1_loss = torch.mean(residual_gate)
            waypoint_loss = zero
            if waypoints is not None:
                target_wp = waypoints.to(device=pred_waypoints.device, dtype=pred_waypoints.dtype)
                waypoint_loss = torch.nn.functional.smooth_l1_loss(pred_waypoints, target_wp)
            command_loss = zero
            if command_id is not None:
                command_loss = self._command_cross_entropy(
                    command_logits,
                    command_id,
                )
        elif self.fusion_mode == "geov2x_residual":
            if base_waypoints is None or base_command_logits is None:
                raise ValueError("fusion_mode=geov2x_residual requires base_waypoints and base_command_logits.")
            if geov2x_geometry is None:
                raise ValueError("fusion_mode=geov2x_residual requires geov2x_geometry.")
            if self.geov2x_residual_head is None:
                raise RuntimeError("geov2x_residual_head is missing for fusion_mode=geov2x_residual")
            if next(self.geov2x_residual_head.parameters()).device != head_device:
                self.geov2x_residual_head.to(head_device)
            (
                pred_waypoints,
                command_logits,
                residual_gate,
                raw_waypoints,
                raw_command_logits,
            ) = self.geov2x_residual_head(
                hidden.to(device=head_device),
                head_attention_mask if head_attention_mask is not None else backbone_inputs.get("attention_mask"),
                geov2x_geometry,
                base_waypoints,
                base_command_logits,
            )
            base_wp = base_waypoints.to(device=pred_waypoints.device, dtype=pred_waypoints.dtype)
            base_logits = base_command_logits.to(device=command_logits.device, dtype=command_logits.dtype)
            residual_l2_loss = torch.mean(raw_waypoints.pow(2)) + 0.01 * torch.mean(raw_command_logits.pow(2))
            gate_l1_loss = torch.mean(residual_gate)
            waypoint_loss = zero
            if waypoints is not None:
                target_wp = waypoints.to(device=pred_waypoints.device, dtype=pred_waypoints.dtype)
                if counterfactual_mask is not None and bool(counterfactual_mask.any()):
                    normal_mask = ~counterfactual_mask
                    if bool(normal_mask.any()):
                        waypoint_loss = torch.nn.functional.smooth_l1_loss(
                            pred_waypoints[normal_mask],
                            target_wp[normal_mask],
                        )
                    counterfactual_base_loss = torch.nn.functional.smooth_l1_loss(
                        pred_waypoints[counterfactual_mask],
                        base_wp[counterfactual_mask],
                    )
                    counterfactual_gate_loss = torch.mean(residual_gate[counterfactual_mask].pow(2))
                    counterfactual_cmd_loss = torch.nn.functional.mse_loss(
                        command_logits[counterfactual_mask],
                        base_logits[counterfactual_mask],
                    )
                else:
                    waypoint_loss = torch.nn.functional.smooth_l1_loss(pred_waypoints, target_wp)
                direct_target = target_wp - base_wp
                direct_loss_raw = torch.nn.functional.smooth_l1_loss(
                    raw_waypoints,
                    direct_target,
                    reduction="none",
                )
                if counterfactual_mask is not None and bool((~counterfactual_mask).any()):
                    direct_residual_loss = direct_loss_raw[~counterfactual_mask].mean()
                elif counterfactual_mask is None:
                    direct_residual_loss = direct_loss_raw.mean()
                step_error = torch.norm(base_wp - target_wp, dim=-1)
                benefit_target = torch.clamp((step_error - 1.0) / 4.0, 0.0, 1.0)
                if counterfactual_mask is not None:
                    benefit_target = benefit_target.masked_fill(counterfactual_mask.view(-1, 1), 0.0)
                gate_benefit_loss = torch.nn.functional.binary_cross_entropy(
                    residual_gate.clamp(1e-5, 1.0 - 1e-5),
                    benefit_target.detach(),
                )
            command_loss = zero
            if command_id is not None:
                if counterfactual_mask is not None and bool(counterfactual_mask.any()):
                    normal_mask = ~counterfactual_mask
                    if bool(normal_mask.any()):
                        command_loss = self._command_cross_entropy(
                            command_logits[normal_mask],
                            command_id[normal_mask],
                        )
                else:
                    command_loss = self._command_cross_entropy(
                        command_logits,
                        command_id,
                    )
        elif self.fusion_mode == "predicted_evidence_residual":
            if base_waypoints is None or base_command_logits is None:
                raise ValueError("fusion_mode=predicted_evidence_residual requires base_waypoints and base_command_logits.")
            if geov2x_geometry is None:
                raise ValueError("fusion_mode=predicted_evidence_residual requires geov2x_geometry.")
            if self.predicted_evidence_residual_head is None:
                raise RuntimeError("predicted_evidence_residual_head is missing")
            if next(self.predicted_evidence_residual_head.parameters()).device != head_device:
                self.predicted_evidence_residual_head.to(head_device)
            (
                pred_waypoints,
                command_logits,
                residual_gate,
                raw_waypoints,
                raw_command_logits,
                predicted_evidence,
            ) = self.predicted_evidence_residual_head(
                pooled_for_heads,
                geov2x_geometry,
                base_waypoints,
                base_command_logits,
            )
            base_wp = base_waypoints.to(device=pred_waypoints.device, dtype=pred_waypoints.dtype)
            base_logits = base_command_logits.to(device=command_logits.device, dtype=command_logits.dtype)
            residual_l2_loss = torch.mean(raw_waypoints.pow(2)) + 0.01 * torch.mean(raw_command_logits.pow(2))
            gate_l1_loss = torch.mean(residual_gate)
            waypoint_loss = zero
            if waypoints is not None:
                target_wp = waypoints.to(device=pred_waypoints.device, dtype=pred_waypoints.dtype)
                if counterfactual_mask is not None and bool(counterfactual_mask.any()):
                    normal_mask = ~counterfactual_mask
                    if bool(normal_mask.any()):
                        waypoint_loss = torch.nn.functional.smooth_l1_loss(
                            pred_waypoints[normal_mask],
                            target_wp[normal_mask],
                        )
                    counterfactual_base_loss = torch.nn.functional.smooth_l1_loss(
                        pred_waypoints[counterfactual_mask],
                        base_wp[counterfactual_mask],
                    )
                    counterfactual_gate_loss = torch.mean(residual_gate[counterfactual_mask].pow(2))
                    counterfactual_cmd_loss = torch.nn.functional.mse_loss(
                        command_logits[counterfactual_mask],
                        base_logits[counterfactual_mask],
                    )
                else:
                    waypoint_loss = torch.nn.functional.smooth_l1_loss(pred_waypoints, target_wp)
                direct_target = target_wp - base_wp
                direct_loss_raw = torch.nn.functional.smooth_l1_loss(
                    raw_waypoints,
                    direct_target,
                    reduction="none",
                )
                if counterfactual_mask is not None and bool((~counterfactual_mask).any()):
                    direct_residual_loss = direct_loss_raw[~counterfactual_mask].mean()
                elif counterfactual_mask is None:
                    direct_residual_loss = direct_loss_raw.mean()
                step_error = torch.norm(base_wp - target_wp, dim=-1)
                benefit_target = torch.clamp((step_error - 1.0) / 4.0, 0.0, 1.0)
                if counterfactual_mask is not None:
                    benefit_target = benefit_target.masked_fill(counterfactual_mask.view(-1, 1), 0.0)
                gate_benefit_loss = torch.nn.functional.binary_cross_entropy(
                    residual_gate.clamp(1e-5, 1.0 - 1e-5),
                    benefit_target.detach(),
                )
            command_loss = zero
            if command_id is not None:
                if counterfactual_mask is not None and bool(counterfactual_mask.any()):
                    normal_mask = ~counterfactual_mask
                    if bool(normal_mask.any()):
                        command_loss = self._command_cross_entropy(
                            command_logits[normal_mask],
                            command_id[normal_mask],
                        )
                else:
                    command_loss = self._command_cross_entropy(
                        command_logits,
                        command_id,
                    )
            if route_risk_target is not None:
                risk_target = route_risk_target.to(device=predicted_evidence.device, dtype=predicted_evidence.dtype)
                if counterfactual_mask is not None and bool((~counterfactual_mask).any()):
                    predicted_evidence_loss = torch.nn.functional.mse_loss(
                        predicted_evidence[~counterfactual_mask],
                        risk_target[~counterfactual_mask],
                    )
                elif counterfactual_mask is None:
                    predicted_evidence_loss = torch.nn.functional.mse_loss(predicted_evidence, risk_target)
        elif self.fusion_mode == "predicted_evidence_token_residual":
            if base_waypoints is None or base_command_logits is None:
                raise ValueError(
                    "fusion_mode=predicted_evidence_token_residual requires base_waypoints and base_command_logits."
                )
            if geov2x_geometry is None:
                raise ValueError("fusion_mode=predicted_evidence_token_residual requires geov2x_geometry.")
            if self.predicted_evidence_token_residual_head is None:
                raise RuntimeError("predicted_evidence_token_residual_head is missing")
            if next(self.predicted_evidence_token_residual_head.parameters()).device != head_device:
                self.predicted_evidence_token_residual_head.to(head_device)
            (
                pred_waypoints,
                command_logits,
                residual_gate,
                raw_waypoints,
                raw_command_logits,
                predicted_evidence,
            ) = self.predicted_evidence_token_residual_head(
                hidden.to(device=head_device),
                head_attention_mask if head_attention_mask is not None else backbone_inputs.get("attention_mask"),
                geov2x_geometry,
                base_waypoints,
                base_command_logits,
            )
            base_wp = base_waypoints.to(device=pred_waypoints.device, dtype=pred_waypoints.dtype)
            base_logits = base_command_logits.to(device=command_logits.device, dtype=command_logits.dtype)
            residual_l2_loss = torch.mean(raw_waypoints.pow(2)) + 0.01 * torch.mean(raw_command_logits.pow(2))
            gate_l1_loss = torch.mean(residual_gate)
            waypoint_loss = zero
            if waypoints is not None:
                target_wp = waypoints.to(device=pred_waypoints.device, dtype=pred_waypoints.dtype)
                if counterfactual_mask is not None and bool(counterfactual_mask.any()):
                    normal_mask = ~counterfactual_mask
                    if bool(normal_mask.any()):
                        waypoint_loss = torch.nn.functional.smooth_l1_loss(
                            pred_waypoints[normal_mask],
                            target_wp[normal_mask],
                        )
                    counterfactual_base_loss = torch.nn.functional.smooth_l1_loss(
                        pred_waypoints[counterfactual_mask],
                        base_wp[counterfactual_mask],
                    )
                    counterfactual_gate_loss = torch.mean(residual_gate[counterfactual_mask].pow(2))
                    counterfactual_cmd_loss = torch.nn.functional.mse_loss(
                        command_logits[counterfactual_mask],
                        base_logits[counterfactual_mask],
                    )
                else:
                    waypoint_loss = torch.nn.functional.smooth_l1_loss(pred_waypoints, target_wp)
                direct_target = target_wp - base_wp
                direct_loss_raw = torch.nn.functional.smooth_l1_loss(
                    raw_waypoints,
                    direct_target,
                    reduction="none",
                )
                if counterfactual_mask is not None and bool((~counterfactual_mask).any()):
                    direct_residual_loss = direct_loss_raw[~counterfactual_mask].mean()
                elif counterfactual_mask is None:
                    direct_residual_loss = direct_loss_raw.mean()
                step_error = torch.norm(base_wp - target_wp, dim=-1)
                benefit_target = torch.clamp((step_error - 1.0) / 4.0, 0.0, 1.0)
                if counterfactual_mask is not None:
                    benefit_target = benefit_target.masked_fill(counterfactual_mask.view(-1, 1), 0.0)
                gate_benefit_loss = torch.nn.functional.binary_cross_entropy(
                    residual_gate.clamp(1e-5, 1.0 - 1e-5),
                    benefit_target.detach(),
                )
            command_loss = zero
            if command_id is not None:
                if counterfactual_mask is not None and bool(counterfactual_mask.any()):
                    normal_mask = ~counterfactual_mask
                    if bool(normal_mask.any()):
                        command_loss = self._command_cross_entropy(
                            command_logits[normal_mask],
                            command_id[normal_mask],
                        )
                else:
                    command_loss = self._command_cross_entropy(
                        command_logits,
                        command_id,
                    )
            if route_risk_target is not None:
                risk_target = route_risk_target.to(device=predicted_evidence.device, dtype=predicted_evidence.dtype)
                if counterfactual_mask is not None and bool((~counterfactual_mask).any()):
                    predicted_evidence_loss = torch.nn.functional.mse_loss(
                        predicted_evidence[~counterfactual_mask],
                        risk_target[~counterfactual_mask],
                    )
                elif counterfactual_mask is None:
                    predicted_evidence_loss = torch.nn.functional.mse_loss(predicted_evidence, risk_target)
        elif self.fusion_mode == "evidence_residual":
            if base_waypoints is None or base_command_logits is None:
                raise ValueError("fusion_mode=evidence_residual requires base_waypoints and base_command_logits.")
            if object_evidence_tokens is None or object_evidence_mask is None:
                raise ValueError("fusion_mode=evidence_residual requires object_evidence_tokens and object_evidence_mask.")
            if self.evidence_residual_head is None:
                raise RuntimeError("evidence_residual_head is missing for fusion_mode=evidence_residual")
            if next(self.evidence_residual_head.parameters()).device != head_device:
                self.evidence_residual_head.to(head_device)
            pred_waypoints, command_logits, residual_gate, raw_waypoints = self.evidence_residual_head(
                pooled_for_heads,
                object_evidence_tokens,
                object_evidence_mask,
                base_waypoints,
                base_command_logits,
            )
            raw_command_logits = command_logits - base_command_logits.to(
                device=command_logits.device,
                dtype=command_logits.dtype,
            )
            residual_l2_loss = torch.mean(raw_waypoints.pow(2)) + 0.01 * torch.mean(raw_command_logits.pow(2))
            gate_l1_loss = torch.mean(residual_gate)
            if v2x_relevance_prior is not None:
                prior = v2x_relevance_prior.to(device=residual_gate.device, dtype=residual_gate.dtype).view(-1, 1)
                gate_mean_for_prior = residual_gate.mean(dim=-1, keepdim=True)
                gate_prior_loss = torch.nn.functional.binary_cross_entropy(
                    gate_mean_for_prior.clamp(1e-5, 1.0 - 1e-5),
                    prior.clamp(0.0, 1.0),
                )
            waypoint_loss = zero
            if waypoints is not None:
                target_wp = waypoints.to(device=pred_waypoints.device, dtype=pred_waypoints.dtype)
                waypoint_loss = torch.nn.functional.smooth_l1_loss(pred_waypoints, target_wp)
            command_loss = zero
            if command_id is not None:
                command_loss = self._command_cross_entropy(
                    command_logits,
                    command_id,
                )
        elif self.fusion_mode == "evidence_directed_residual":
            if base_waypoints is None or base_command_logits is None:
                raise ValueError(
                    "fusion_mode=evidence_directed_residual requires base_waypoints and base_command_logits."
                )
            if object_evidence_tokens is None or object_evidence_mask is None:
                raise ValueError(
                    "fusion_mode=evidence_directed_residual requires object_evidence_tokens and object_evidence_mask."
                )
            if self.evidence_directed_residual_head is None:
                raise RuntimeError(
                    "evidence_directed_residual_head is missing for fusion_mode=evidence_directed_residual"
                )
            if next(self.evidence_directed_residual_head.parameters()).device != head_device:
                self.evidence_directed_residual_head.to(head_device)
            (
                pred_waypoints,
                command_logits,
                residual_gate,
                raw_waypoints,
                raw_command_logits,
            ) = self.evidence_directed_residual_head(
                pooled_for_heads,
                object_evidence_tokens,
                object_evidence_mask,
                base_waypoints,
                base_command_logits,
            )
            base_wp = base_waypoints.to(device=pred_waypoints.device, dtype=pred_waypoints.dtype)
            safety_extra: Dict[str, Any] = {}
            if self.safety_mixer == "base_shape_guard":
                safety_extra = self._apply_base_shape_guard(
                    pred_waypoints,
                    base_wp,
                    lateral_shape_residual_scale,
                )
                pred_waypoints = safety_extra["pred_waypoints"]
            elif self.safety_mixer == "turn_right_lateral_guard":
                safety_extra = self._apply_turn_right_lateral_guard(
                    pred_waypoints,
                    base_wp,
                    command_logits,
                    base_command_logits.to(device=command_logits.device, dtype=command_logits.dtype),
                    route_relevance_prior,
                    lateral_shape_residual_scale,
                    route_right_turn_shape,
                )
                pred_waypoints = safety_extra["pred_waypoints"]
            residual_l2_loss = torch.mean(raw_waypoints.pow(2)) + 0.01 * torch.mean(raw_command_logits.pow(2))
            gate_l1_loss = torch.mean(residual_gate)
            prior_source = route_relevance_prior if route_relevance_prior is not None else v2x_relevance_prior
            if prior_source is not None:
                prior = prior_source.to(device=residual_gate.device, dtype=residual_gate.dtype).view(-1, 1)
                gate_mean_for_prior = residual_gate.mean(dim=-1, keepdim=True)
                gate_prior_loss = torch.nn.functional.binary_cross_entropy(
                    gate_mean_for_prior.clamp(1e-5, 1.0 - 1e-5),
                    prior.clamp(0.0, 1.0),
                )
            waypoint_loss = zero
            if waypoints is not None:
                target_wp = waypoints.to(device=pred_waypoints.device, dtype=pred_waypoints.dtype)
                waypoint_loss = torch.nn.functional.smooth_l1_loss(pred_waypoints, target_wp)
                direct_target = target_wp - base_wp
                direct_loss_raw = torch.nn.functional.smooth_l1_loss(
                    raw_waypoints,
                    direct_target,
                    reduction="none",
                )
                direct_residual_loss = direct_loss_raw.mean()
                step_error = torch.norm(base_wp - target_wp, dim=-1)
                relevance = torch.zeros_like(step_error[:, :1])
                if prior_source is not None:
                    relevance = prior_source.to(device=step_error.device, dtype=step_error.dtype).view(-1, 1)
                benefit_target = (
                    torch.clamp((step_error - 1.0) / 4.0, 0.0, 1.0) * relevance.clamp(0.0, 1.0)
                )
                gate_benefit_loss = torch.nn.functional.binary_cross_entropy(
                    residual_gate.clamp(1e-5, 1.0 - 1e-5),
                    benefit_target.detach(),
                )
            command_loss = zero
            if command_id is not None:
                command_loss = self._command_cross_entropy(
                    command_logits,
                    command_id,
                )
        else:
            pred_waypoints, waypoint_loss = self.action_head(pooled_for_heads, waypoints)
            base_command_logits, _ = self.command_head(command_pooled_for_heads, None)
            if self.command_head_mode == "waypoint_shape_fusion":
                if self.shape_command_head is None:
                    raise RuntimeError("shape_command_head is missing for waypoint_shape_fusion")
                if next(self.shape_command_head.parameters()).device != head_device:
                    self.shape_command_head.to(head_device)
                shape_delta_logits, _ = self.shape_command_head(
                    command_pooled_for_heads,
                    pred_waypoints,
                    None,
                )
                command_logits = base_command_logits + shape_delta_logits
            else:
                command_logits = base_command_logits
            command_loss = zero
            if command_id is not None:
                command_loss = self._command_cross_entropy(command_logits, command_id)
            if use_task_query_aux:
                if task_query_output is None:
                    raise RuntimeError("task-query auxiliary losses require task-query output")
                ego_action_state = task_query_output["ego_action_state"].to(dtype=head_dtype)
                ego_command_state = task_query_output["ego_command_state"].to(dtype=head_dtype)
                ego_waypoints, _ = self.action_head(ego_action_state, None)
                ego_command_logits, _ = self.command_head(ego_command_state, None)
                roles = [
                    str(sample.get("trajectory_grounded_infra_role", "")).lower()
                    for sample in samples
                ]
                if any(
                    role
                    not in {
                        "context_only",
                        "no_additional_evidence",
                        "decision_relevant",
                    }
                    for role in roles
                ):
                    raise ValueError(
                        "task-query auxiliary losses require trajectory-grounded roles"
                    )
                conditions = [
                    str(sample.get("infra_condition", "normal")).lower()
                    for sample in samples
                ]
                invariance_mask = torch.tensor(
                    [
                        role in {"context_only", "no_additional_evidence"}
                        or condition in {"blank", "shuffled"}
                        for role, condition in zip(roles, conditions)
                    ],
                    device=head_device,
                    dtype=torch.bool,
                )
                if bool(invariance_mask.any()):
                    if waypoints is not None:
                        target_wp = waypoints.to(
                            device=ego_waypoints.device,
                            dtype=ego_waypoints.dtype,
                        )
                        task_query_ego_wp_loss = torch.nn.functional.smooth_l1_loss(
                            ego_waypoints[invariance_mask],
                            target_wp[invariance_mask],
                        )
                        task_query_consistency_wp_loss = torch.nn.functional.smooth_l1_loss(
                            pred_waypoints[invariance_mask],
                            ego_waypoints[invariance_mask].detach(),
                        )
                    if command_id is not None:
                        task_query_ego_cmd_loss = self._command_cross_entropy(
                            ego_command_logits[invariance_mask],
                            command_id[invariance_mask],
                        )
                        task_query_consistency_cmd_loss = torch.nn.functional.kl_div(
                            torch.nn.functional.log_softmax(
                                command_logits[invariance_mask],
                                dim=-1,
                            ),
                            torch.nn.functional.softmax(
                                ego_command_logits[invariance_mask].detach(),
                                dim=-1,
                            ),
                            reduction="batchmean",
                        )
                    infra_mass = task_query_output["infra_attention_mass"]
                    task_query_infra_attention_loss = infra_mass[invariance_mask].mean()
        lm_loss = getattr(outputs, "loss", None)
        lm_loss = lm_loss if lm_loss is not None else zero
        waypoint_loss = waypoint_loss if waypoint_loss is not None else zero
        command_loss = command_loss if command_loss is not None else zero
        teacher_wp_loss = zero
        teacher_cmd_loss = zero
        if teacher_waypoints is not None and self.lambda_teacher_wp > 0.0:
            teacher_target = teacher_waypoints.to(device=pred_waypoints.device, dtype=pred_waypoints.dtype)
            teacher_loss = torch.nn.functional.smooth_l1_loss(
                pred_waypoints,
                teacher_target,
                reduction="none",
            )
            if teacher_available is not None:
                teacher_wp_loss = self._masked_sample_mean(
                    teacher_loss,
                    teacher_available,
                )
            else:
                teacher_wp_loss = teacher_loss.mean()
        if teacher_command_logits is not None and self.lambda_teacher_cmd > 0.0:
            teacher_logits = teacher_command_logits.to(device=command_logits.device, dtype=command_logits.dtype)
            teacher_cmd_raw = torch.nn.functional.mse_loss(command_logits, teacher_logits, reduction="none").mean(dim=-1)
            if teacher_available is not None:
                weights = teacher_available.to(device=command_logits.device, dtype=command_logits.dtype).view(-1)
                teacher_cmd_loss = (teacher_cmd_raw * weights).sum() / weights.sum().clamp_min(1.0)
            else:
                teacher_cmd_loss = teacher_cmd_raw.mean()
        if waypoints is not None:
            target_wp = waypoints.to(device=pred_waypoints.device, dtype=pred_waypoints.dtype)
            fde_loss = torch.nn.functional.smooth_l1_loss(pred_waypoints[:, -1], target_wp[:, -1])
        else:
            fde_loss = zero
        total_loss = (
            self.lambda_lm * lm_loss
            + self.lambda_wp * waypoint_loss
            + self.lambda_cmd * command_loss
            + self.lambda_fde * fde_loss
            + self.lambda_residual_l2 * residual_l2_loss
            + self.lambda_gate_l1 * gate_l1_loss
            + self.lambda_gate_prior * gate_prior_loss
            + self.lambda_direct_residual * direct_residual_loss
            + self.lambda_gate_benefit * gate_benefit_loss
            + self.lambda_teacher_wp * teacher_wp_loss
            + self.lambda_teacher_cmd * teacher_cmd_loss
            + self.lambda_counterfactual_base * counterfactual_base_loss
            + self.lambda_counterfactual_gate * counterfactual_gate_loss
            + self.lambda_counterfactual_cmd * counterfactual_cmd_loss
            + self.lambda_predicted_evidence * predicted_evidence_loss
            + self.lambda_task_query_ego_wp * task_query_ego_wp_loss
            + self.lambda_task_query_ego_cmd * task_query_ego_cmd_loss
            + self.lambda_task_query_consistency_wp * task_query_consistency_wp_loss
            + self.lambda_task_query_consistency_cmd * task_query_consistency_cmd_loss
            + self.lambda_task_query_infra_attention * task_query_infra_attention_loss
        )
        extra: Dict[str, Any] = {}
        if task_query_output is not None:
            extra.update(
                {
                    "task_query_ego_attention_mass": task_query_output[
                        "ego_attention_mass"
                    ].mean(),
                    "task_query_infra_attention_mass": task_query_output[
                        "infra_attention_mass"
                    ].mean(),
                    "task_query_ego_wp_loss": task_query_ego_wp_loss,
                    "task_query_ego_cmd_loss": task_query_ego_cmd_loss,
                    "task_query_consistency_wp_loss": task_query_consistency_wp_loss,
                    "task_query_consistency_cmd_loss": task_query_consistency_cmd_loss,
                    "task_query_infra_attention_loss": task_query_infra_attention_loss,
                }
            )
        if residual_gate is not None:
            gate_flat = residual_gate.detach().float().view(-1)
            delta_norm = raw_waypoints.detach().float().norm(dim=-1).mean(dim=1)
            extra.update(
                {
                    "gate": residual_gate,
                    "delta_waypoints": raw_waypoints,
                    "delta_command_logits": raw_command_logits,
                    "base_waypoints": base_waypoints,
                    "base_command_logits": base_command_logits,
                    "residual_l2_loss": residual_l2_loss,
                    "gate_l1_loss": gate_l1_loss,
                    "gate_prior_loss": gate_prior_loss,
                    "direct_residual_loss": direct_residual_loss,
                    "gate_benefit_loss": gate_benefit_loss,
                    "teacher_wp_loss": teacher_wp_loss,
                    "teacher_cmd_loss": teacher_cmd_loss,
                    "counterfactual_base_loss": counterfactual_base_loss,
                    "counterfactual_gate_loss": counterfactual_gate_loss,
                    "counterfactual_cmd_loss": counterfactual_cmd_loss,
                    "predicted_evidence_loss": predicted_evidence_loss,
                    "gate_mean": gate_flat.mean(),
                    "gate_p90": torch.quantile(gate_flat, 0.9),
                    "gate_max": gate_flat.max(),
                    "delta_waypoint_norm": delta_norm.mean(),
                }
            )
            if "predicted_evidence" in locals():
                extra["predicted_evidence"] = predicted_evidence
            if self.fusion_mode == "evidence_directed_residual" and "safety_extra" in locals():
                extra.update(safety_extra)
        if return_analysis_hidden:
            extra["analysis_hidden"] = hidden
        return {
            "loss": total_loss,
            "total_loss": total_loss,
            "lm_loss": lm_loss,
            "waypoint_loss": waypoint_loss,
            "command_loss": command_loss,
            "fde_loss": fde_loss,
            "pred_waypoints": pred_waypoints,
            "command_logits": command_logits,
            **extra,
        }

    def save_pretrained(self, output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if hasattr(self.backbone, "save_pretrained"):
            self.backbone.save_pretrained(str(out / "backbone"))
        torch.save(
            {
                "action_head": self.action_head.state_dict(),
                "command_head": self.command_head.state_dict(),
                "shape_command_head": (
                    self.shape_command_head.state_dict() if self.shape_command_head is not None else None
                ),
                "residual_gate_head": (
                    self.residual_gate_head.state_dict() if self.residual_gate_head is not None else None
                ),
                "evidence_residual_head": (
                    self.evidence_residual_head.state_dict() if self.evidence_residual_head is not None else None
                ),
                "evidence_directed_residual_head": (
                    self.evidence_directed_residual_head.state_dict()
                    if self.evidence_directed_residual_head is not None
                    else None
                ),
                "geov2x_residual_head": (
                    self.geov2x_residual_head.state_dict()
                    if self.geov2x_residual_head is not None
                    else None
                ),
                "predicted_evidence_residual_head": (
                    self.predicted_evidence_residual_head.state_dict()
                    if self.predicted_evidence_residual_head is not None
                    else None
                ),
                "predicted_evidence_token_residual_head": (
                    self.predicted_evidence_token_residual_head.state_dict()
                    if self.predicted_evidence_token_residual_head is not None
                    else None
                ),
                "task_query_pooler": (
                    self.task_query_pooler.state_dict()
                    if self.task_query_pooler is not None
                    else None
                ),
            },
            out / "action_heads.pt",
        )
        with (out / "baseline_config.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "lambda_lm": self.lambda_lm,
                    "lambda_wp": self.lambda_wp,
                    "lambda_cmd": self.lambda_cmd,
                    "lambda_fde": self.lambda_fde,
                    "lambda_residual_l2": self.lambda_residual_l2,
                    "lambda_gate_l1": self.lambda_gate_l1,
                    "lambda_gate_prior": self.lambda_gate_prior,
                    "head_pooling": self.head_pooling,
                    "vision_encode_per_view": self.vision_encode_per_view,
                    "fusion_mode": self.fusion_mode,
                    "residual_gate_init_bias": self.residual_gate_init_bias,
                    "evidence_feature_dim": self.evidence_feature_dim,
                    "evidence_hidden_size": self.evidence_hidden_size,
                    "evidence_layers": self.evidence_layers,
                    "evidence_heads": self.evidence_heads,
                    "evidence_gate_mode": self.evidence_gate_mode,
                    "safety_mixer": self.safety_mixer,
                    "base_shape_guard_alpha": self.base_shape_guard_alpha,
                    "base_shape_guard_waypoints_only": self.base_shape_guard_waypoints_only,
                    "base_shape_guard_source": self.base_shape_guard_source,
                    "turn_right_guard_alpha": self.turn_right_guard_alpha,
                    "turn_right_guard_prior_threshold": self.turn_right_guard_prior_threshold,
                    "turn_right_guard_residual_threshold": self.turn_right_guard_residual_threshold,
                    "right_turn_shape_guard_alpha": self.right_turn_shape_guard_alpha,
                    "right_turn_shape_guard_prior_threshold": self.right_turn_shape_guard_prior_threshold,
                    "lambda_teacher_wp": self.lambda_teacher_wp,
                    "lambda_teacher_cmd": self.lambda_teacher_cmd,
                    "geometry_dim": self.geometry_dim,
                    "geov2x_hidden_size": self.geov2x_hidden_size,
                    "geov2x_heads": self.geov2x_heads,
                    "lambda_counterfactual_base": self.lambda_counterfactual_base,
                    "lambda_counterfactual_gate": self.lambda_counterfactual_gate,
                    "lambda_counterfactual_cmd": self.lambda_counterfactual_cmd,
                    "predicted_evidence_dim": self.predicted_evidence_dim,
                    "predicted_evidence_hidden_size": self.predicted_evidence_hidden_size,
                    "lambda_predicted_evidence": self.lambda_predicted_evidence,
                    "task_query_dim": self.task_query_dim,
                    "task_query_heads": self.task_query_heads,
                    "task_query_dropout": self.task_query_dropout,
                    "lambda_task_query_ego_wp": self.lambda_task_query_ego_wp,
                    "lambda_task_query_ego_cmd": self.lambda_task_query_ego_cmd,
                    "lambda_task_query_consistency_wp": (
                        self.lambda_task_query_consistency_wp
                    ),
                    "lambda_task_query_consistency_cmd": (
                        self.lambda_task_query_consistency_cmd
                    ),
                    "lambda_task_query_infra_attention": (
                        self.lambda_task_query_infra_attention
                    ),
                    "lambda_direct_residual": self.lambda_direct_residual,
                    "lambda_gate_benefit": self.lambda_gate_benefit,
                    "command_class_weights": (
                        self.command_class_weights.detach().cpu().tolist()
                        if self.command_class_weights is not None
                        else None
                    ),
                    "command_class_weight_reduction": self.command_class_weight_reduction,
                    "command_class_weight_normalizer": self.command_class_weight_normalizer,
                    "command_head_mode": self.command_head_mode,
                    "command_shape_hidden_size": self.command_shape_hidden_size,
                },
                f,
                indent=2,
            )
