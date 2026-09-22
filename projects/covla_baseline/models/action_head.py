"""Action and command heads for the CoVLM Qwen-VL baseline."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class TaskQueryPooler(nn.Module):
    """Non-causal task readout over the two Qwen visual-token spans.

    The backbone remains unchanged. Two learned queries select evidence for the
    trajectory and command heads after Qwen has encoded both views, avoiding an
    unconditional average over causally incomplete visual states.
    """

    def __init__(
        self,
        hidden_size: int,
        query_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(query_dim) % int(num_heads) != 0:
            raise ValueError("task-query dimension must be divisible by the number of heads")
        self.hidden_size = int(hidden_size)
        self.query_dim = int(query_dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.token_proj = nn.Linear(self.hidden_size, self.query_dim)
        self.view_embeddings = nn.Parameter(torch.zeros(2, self.query_dim))
        self.task_queries = nn.Parameter(torch.randn(2, self.query_dim) * 0.02)
        self.memory_norm = nn.LayerNorm(self.query_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.query_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(self.query_dim)
        self.out_proj = nn.Linear(self.query_dim, self.hidden_size)
        self.out_norm = nn.LayerNorm(self.hidden_size)

    def _attend(
        self,
        memory: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not bool(valid_mask.any(dim=1).all()):
            raise ValueError("TaskQueryPooler received a sample without visual tokens")
        queries = self.task_queries.unsqueeze(0).expand(memory.shape[0], -1, -1)
        attended, weights = self.cross_attention(
            queries,
            memory,
            memory,
            key_padding_mask=~valid_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        attended = self.query_norm(queries + attended)
        states = self.out_norm(self.out_proj(attended))
        return states, weights.mean(dim=1)

    def forward(
        self,
        hidden: torch.Tensor,
        ego_view_mask: torch.Tensor,
        infra_view_mask: torch.Tensor,
        *,
        return_ego_control: bool = False,
    ) -> Dict[str, torch.Tensor]:
        dtype = self.token_proj.weight.dtype
        hidden = hidden.to(dtype=dtype)
        ego = ego_view_mask.to(device=hidden.device, dtype=torch.bool)
        infra = infra_view_mask.to(device=hidden.device, dtype=torch.bool)
        if bool(torch.logical_and(ego, infra).any()):
            raise ValueError("ego and infrastructure view masks must be disjoint")
        memory = self.token_proj(hidden)
        memory = memory + ego.unsqueeze(-1).to(memory.dtype) * self.view_embeddings[0]
        memory = memory + infra.unsqueeze(-1).to(memory.dtype) * self.view_embeddings[1]
        memory = self.memory_norm(memory)
        both_mask = ego | infra
        task_states, weights = self._attend(memory, both_mask)
        result = {
            "action_state": task_states[:, 0],
            "command_state": task_states[:, 1],
            "ego_attention_mass": (weights * ego.unsqueeze(1).to(weights.dtype)).sum(dim=-1),
            "infra_attention_mass": (weights * infra.unsqueeze(1).to(weights.dtype)).sum(dim=-1),
        }
        if return_ego_control:
            ego_states, _ = self._attend(memory, ego)
            result["ego_action_state"] = ego_states[:, 0]
            result["ego_command_state"] = ego_states[:, 1]
        return result


class ActionHead(nn.Module):
    """MLP head predicting 6 future waypoints in ego-local coordinates."""

    def __init__(self, hidden_size: int, num_steps: int = 6, loss_type: str = "smooth_l1") -> None:
        super().__init__()
        mid = max(hidden_size // 2, 128)
        self.num_steps = num_steps
        self.loss_type = loss_type
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mid),
            nn.GELU(),
            nn.LayerNorm(mid),
            nn.Linear(mid, num_steps * 2),
        )

    def forward(self, pooled: torch.Tensor, target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pred = self.net(pooled).view(-1, self.num_steps, 2)
        loss = None
        if target is not None:
            target = target.to(device=pred.device, dtype=pred.dtype)
            if self.loss_type == "mse":
                loss = F.mse_loss(pred, target)
            else:
                loss = F.smooth_l1_loss(pred, target)
        return pred, loss

    def zero_init_output(self) -> None:
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)


class CommandHead(nn.Module):
    """MLP command classifier."""

    def __init__(self, hidden_size: int, num_commands: int) -> None:
        super().__init__()
        mid = max(hidden_size // 2, 128)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mid),
            nn.GELU(),
            nn.LayerNorm(mid),
            nn.Linear(mid, num_commands),
        )

    def forward(
        self,
        pooled: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        class_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        logits = self.net(pooled)
        loss = None
        if target is not None:
            weight = None
            if class_weight is not None:
                weight = class_weight.to(device=logits.device, dtype=logits.dtype)
            loss = F.cross_entropy(logits, target.to(logits.device), weight=weight)
        return logits, loss

    def zero_init_output(self) -> None:
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)


class WaypointShapeCommandHead(nn.Module):
    """Command classifier that consumes deployable features from predicted waypoints."""

    def __init__(
        self,
        hidden_size: int,
        num_commands: int,
        num_steps: int = 6,
        shape_hidden_size: int = 64,
    ) -> None:
        super().__init__()
        self.num_steps = int(num_steps)
        self.shape_dim = self.num_steps * 2 + 5
        self.shape_proj = nn.Sequential(
            nn.Linear(self.shape_dim, int(shape_hidden_size)),
            nn.GELU(),
            nn.LayerNorm(int(shape_hidden_size)),
        )
        mid = max(hidden_size // 2, 128)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + int(shape_hidden_size), mid),
            nn.GELU(),
            nn.LayerNorm(mid),
            nn.Linear(mid, num_commands),
        )
        final = self.fusion[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    @staticmethod
    def shape_features(waypoints: torch.Tensor) -> torch.Tensor:
        wp = waypoints.detach()
        flat = wp.flatten(1)
        lateral = wp[:, :, 0]
        forward = wp[:, :, 1]
        deltas = wp[:, 1:] - wp[:, :-1]
        final_lateral = lateral[:, -1]
        forward_span = forward.max(dim=1).values - forward.min(dim=1).values
        path_length = torch.norm(deltas, dim=-1).sum(dim=1)
        final_heading = torch.atan2(deltas[:, -1, 0], deltas[:, -1, 1].clamp_min(1e-6))
        lateral_range = lateral.max(dim=1).values - lateral.min(dim=1).values
        compact = torch.stack(
            [final_lateral, forward_span, path_length, final_heading, lateral_range],
            dim=-1,
        )
        return torch.cat([flat, compact], dim=-1)

    def forward(
        self,
        pooled: torch.Tensor,
        pred_waypoints: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        class_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        features = self.shape_features(pred_waypoints).to(device=pooled.device, dtype=pooled.dtype)
        shape = self.shape_proj(features)
        logits = self.fusion(torch.cat([pooled, shape], dim=-1))
        loss = None
        if target is not None:
            weight = None
            if class_weight is not None:
                weight = class_weight.to(device=logits.device, dtype=logits.dtype)
            loss = F.cross_entropy(logits, target.to(logits.device), weight=weight)
        return logits, loss


class ResidualGateHead(nn.Module):
    """Predict a bounded scalar residual gate for ego-anchored V2X fusion."""

    def __init__(self, hidden_size: int, init_bias: float = -3.0) -> None:
        super().__init__()
        mid = max(hidden_size // 4, 128)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mid),
            nn.GELU(),
            nn.LayerNorm(mid),
            nn.Linear(mid, 1),
        )
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.constant_(final.bias, float(init_bias))

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(pooled)).view(-1, 1)


class GeoV2XResidualHead(nn.Module):
    """Action-query residual decoder using raw Qwen hidden states and deployable geometry."""

    def __init__(
        self,
        qwen_hidden_size: int,
        geometry_dim: int,
        num_commands: int,
        num_steps: int = 6,
        decoder_hidden_size: int = 256,
        num_heads: int = 4,
        gate_init_bias: float = -2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_steps = int(num_steps)
        self.num_commands = int(num_commands)
        self.decoder_hidden_size = int(decoder_hidden_size)
        self.qwen_proj = nn.Sequential(
            nn.Linear(qwen_hidden_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.decoder_hidden_size),
        )
        self.geometry_proj = nn.Sequential(
            nn.Linear(int(geometry_dim), self.decoder_hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.decoder_hidden_size),
        )
        base_dim = self.num_steps * 2 + self.num_commands
        self.base_proj = nn.Sequential(
            nn.Linear(base_dim, self.decoder_hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.decoder_hidden_size),
        )
        self.queries = nn.Parameter(torch.randn(self.num_steps + 1, self.decoder_hidden_size) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.decoder_hidden_size,
            num_heads=max(1, int(num_heads)),
            dropout=float(dropout),
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(self.decoder_hidden_size),
            nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.decoder_hidden_size * 2, self.decoder_hidden_size),
        )
        self.out_norm = nn.LayerNorm(self.decoder_hidden_size)
        self.delta_wp = nn.Linear(self.decoder_hidden_size, 2)
        self.step_gate = nn.Linear(self.decoder_hidden_size, 1)
        self.delta_cmd = nn.Linear(self.decoder_hidden_size, self.num_commands)
        self.cmd_gate = nn.Linear(self.decoder_hidden_size, 1)
        for layer in (self.delta_wp, self.delta_cmd, self.step_gate, self.cmd_gate):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.constant_(self.step_gate.bias, float(gate_init_bias))
        nn.init.constant_(self.cmd_gate.bias, float(gate_init_bias))

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        geometry: torch.Tensor,
        base_waypoints: torch.Tensor,
        base_command_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = hidden.to(dtype=next(self.parameters()).dtype)
        memory = self.qwen_proj(hidden)
        if attention_mask is not None:
            valid = attention_mask.to(device=hidden.device, dtype=torch.bool)
        else:
            valid = torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.bool)
        geometry = geometry.to(device=hidden.device, dtype=memory.dtype)
        base_wp = base_waypoints.to(device=hidden.device, dtype=memory.dtype)
        base_logits = base_command_logits.to(device=hidden.device, dtype=memory.dtype)
        geometry_token = self.geometry_proj(geometry).unsqueeze(1)
        base_flat = torch.cat([base_wp.flatten(1), base_logits], dim=-1)
        base_token = self.base_proj(base_flat).unsqueeze(1)
        memory = torch.cat([memory, geometry_token, base_token], dim=1)
        extra_valid = torch.ones((valid.shape[0], 2), device=valid.device, dtype=torch.bool)
        valid = torch.cat([valid, extra_valid], dim=1)
        queries = self.queries.unsqueeze(0).expand(hidden.shape[0], -1, -1)
        attended, _ = self.cross_attn(queries, memory, memory, key_padding_mask=~valid)
        decoded = self.out_norm(attended + self.ffn(attended))
        step_tokens = decoded[:, : self.num_steps]
        cmd_token = decoded[:, self.num_steps]
        delta_wp = self.delta_wp(step_tokens)
        step_gate = torch.sigmoid(self.step_gate(step_tokens))
        delta_cmd = self.delta_cmd(cmd_token)
        cmd_gate = torch.sigmoid(self.cmd_gate(cmd_token))
        pred_wp = base_wp + step_gate * delta_wp
        pred_cmd = base_logits + cmd_gate * delta_cmd
        return pred_wp, pred_cmd, step_gate.squeeze(-1), delta_wp, delta_cmd


class PredictedEvidenceResidualHead(nn.Module):
    """Predict compact route-risk evidence from Qwen state, then plan from it.

    The planner consumes only the predicted evidence vector, route geometry, and
    frozen base plan/logits.  L1-derived route-risk can supervise the predictor
    during training, but is never concatenated into the planner input.
    """

    def __init__(
        self,
        qwen_hidden_size: int,
        geometry_dim: int,
        num_commands: int,
        risk_dim: int = 10,
        num_steps: int = 6,
        hidden_size: int = 256,
        gate_init_bias: float = -1.0,
    ) -> None:
        super().__init__()
        self.num_steps = int(num_steps)
        self.num_commands = int(num_commands)
        self.risk_dim = int(risk_dim)
        self.risk_predictor = nn.Sequential(
            nn.Linear(qwen_hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.risk_dim),
        )
        planner_dim = self.risk_dim + int(geometry_dim) + self.num_steps * 2 + self.num_commands
        self.planner = nn.Sequential(
            nn.Linear(planner_dim, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        self.delta_wp = nn.Linear(hidden_size, self.num_steps * 2)
        self.delta_cmd = nn.Linear(hidden_size, self.num_commands)
        self.gate = nn.Linear(hidden_size, self.num_steps)
        nn.init.zeros_(self.delta_wp.weight)
        nn.init.zeros_(self.delta_wp.bias)
        nn.init.zeros_(self.delta_cmd.weight)
        nn.init.zeros_(self.delta_cmd.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_init_bias))

    def forward(
        self,
        pooled: torch.Tensor,
        geometry: torch.Tensor,
        base_waypoints: torch.Tensor,
        base_command_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = pooled.to(dtype=next(self.parameters()).dtype)
        predicted_risk = torch.sigmoid(self.risk_predictor(pooled))
        geometry = geometry.to(device=pooled.device, dtype=pooled.dtype)
        base_wp = base_waypoints.to(device=pooled.device, dtype=pooled.dtype)
        base_logits = base_command_logits.to(device=pooled.device, dtype=pooled.dtype)
        features = torch.cat([predicted_risk, geometry, base_wp.flatten(1), base_logits], dim=-1)
        hidden = self.planner(features)
        raw_waypoints = self.delta_wp(hidden).view(-1, self.num_steps, 2)
        raw_command_logits = self.delta_cmd(hidden)
        gate = torch.sigmoid(self.gate(hidden))
        pred_wp = base_wp + gate.unsqueeze(-1) * raw_waypoints
        pred_cmd = base_logits + gate.mean(dim=-1, keepdim=True) * raw_command_logits
        return pred_wp, pred_cmd, gate, raw_waypoints, raw_command_logits, predicted_risk


class PredictedEvidenceTokenResidualHead(nn.Module):
    """Predict route-risk evidence by querying Qwen tokens, then plan from predictions only."""

    def __init__(
        self,
        qwen_hidden_size: int,
        geometry_dim: int,
        num_commands: int,
        risk_dim: int = 10,
        num_steps: int = 6,
        hidden_size: int = 256,
        num_heads: int = 4,
        gate_init_bias: float = -1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_steps = int(num_steps)
        self.num_commands = int(num_commands)
        self.risk_dim = int(risk_dim)
        self.hidden_size = int(hidden_size)
        self.qwen_proj = nn.Sequential(
            nn.Linear(qwen_hidden_size, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
        )
        self.geometry_proj = nn.Sequential(
            nn.Linear(int(geometry_dim), self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
        )
        base_dim = self.num_steps * 2 + self.num_commands
        self.base_proj = nn.Sequential(
            nn.Linear(base_dim, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
        )
        self.risk_query = nn.Parameter(torch.randn(1, self.hidden_size) * 0.02)
        self.risk_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=max(1, int(num_heads)),
            dropout=float(dropout),
            batch_first=True,
        )
        self.risk_ffn = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )
        self.risk_norm = nn.LayerNorm(self.hidden_size)
        self.risk_out = nn.Linear(self.hidden_size, self.risk_dim)
        planner_dim = self.risk_dim + int(geometry_dim) + self.num_steps * 2 + self.num_commands
        self.planner = nn.Sequential(
            nn.Linear(planner_dim, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
        )
        self.delta_wp = nn.Linear(self.hidden_size, self.num_steps * 2)
        self.delta_cmd = nn.Linear(self.hidden_size, self.num_commands)
        self.gate = nn.Linear(self.hidden_size, self.num_steps)
        nn.init.zeros_(self.delta_wp.weight)
        nn.init.zeros_(self.delta_wp.bias)
        nn.init.zeros_(self.delta_cmd.weight)
        nn.init.zeros_(self.delta_cmd.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_init_bias))

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        geometry: torch.Tensor,
        base_waypoints: torch.Tensor,
        base_command_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = hidden.to(dtype=next(self.parameters()).dtype)
        memory = self.qwen_proj(hidden)
        if attention_mask is not None:
            valid = attention_mask.to(device=hidden.device, dtype=torch.bool)
        else:
            valid = torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.bool)
        geometry = geometry.to(device=hidden.device, dtype=memory.dtype)
        base_wp = base_waypoints.to(device=hidden.device, dtype=memory.dtype)
        base_logits = base_command_logits.to(device=hidden.device, dtype=memory.dtype)
        geometry_token = self.geometry_proj(geometry).unsqueeze(1)
        base_flat = torch.cat([base_wp.flatten(1), base_logits], dim=-1)
        base_token = self.base_proj(base_flat).unsqueeze(1)
        memory = torch.cat([memory, geometry_token, base_token], dim=1)
        extra_valid = torch.ones((valid.shape[0], 2), device=valid.device, dtype=torch.bool)
        valid = torch.cat([valid, extra_valid], dim=1)
        query = self.risk_query.unsqueeze(0).expand(hidden.shape[0], -1, -1)
        attended, _ = self.risk_attn(query, memory, memory, key_padding_mask=~valid)
        risk_state = self.risk_norm(attended + self.risk_ffn(attended)).squeeze(1)
        predicted_risk = torch.sigmoid(self.risk_out(risk_state))
        planner_features = torch.cat(
            [predicted_risk, geometry, base_wp.flatten(1), base_logits],
            dim=-1,
        )
        plan_state = self.planner(planner_features)
        raw_waypoints = self.delta_wp(plan_state).view(-1, self.num_steps, 2)
        raw_command_logits = self.delta_cmd(plan_state)
        gate = torch.sigmoid(self.gate(plan_state))
        pred_wp = base_wp + gate.unsqueeze(-1) * raw_waypoints
        pred_cmd = base_logits + gate.mean(dim=-1, keepdim=True) * raw_command_logits
        return pred_wp, pred_cmd, gate, raw_waypoints, raw_command_logits, predicted_risk


class L1EvidenceEncoder(nn.Module):
    """Small transformer encoder for current-frame L1 object evidence tokens."""

    def __init__(
        self,
        feature_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=max(1, int(num_heads)),
            dim_feedforward=max(hidden_size * 2, 128),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(num_layers)))
        self.out_norm = nn.LayerNorm(hidden_size)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"object_evidence_tokens must be [B, K, F], got {tuple(tokens.shape)}")
        if mask.ndim != 2:
            raise ValueError(f"object_evidence_mask must be [B, K], got {tuple(mask.shape)}")
        x = self.input_proj(tokens)
        valid = mask.to(device=tokens.device, dtype=torch.bool)
        encoded = self.encoder(x, src_key_padding_mask=~valid)
        valid_f = valid.to(dtype=encoded.dtype).unsqueeze(-1)
        denom = valid_f.sum(dim=1).clamp_min(1.0)
        pooled = (encoded * valid_f).sum(dim=1) / denom
        return self.out_norm(pooled)

    def encode_tokens(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"object_evidence_tokens must be [B, K, F], got {tuple(tokens.shape)}")
        if mask.ndim != 2:
            raise ValueError(f"object_evidence_mask must be [B, K], got {tuple(mask.shape)}")
        x = self.input_proj(tokens)
        valid = mask.to(device=tokens.device, dtype=torch.bool)
        encoded = self.encoder(x, src_key_padding_mask=~valid)
        return self.out_norm(encoded)


class EvidenceResidualFusionHead(nn.Module):
    """Predict residual planning deltas from Qwen state, base plan, and L1 evidence."""

    def __init__(
        self,
        qwen_hidden_size: int,
        evidence_feature_dim: int,
        num_commands: int,
        num_steps: int = 6,
        evidence_hidden_size: int = 256,
        evidence_layers: int = 2,
        evidence_heads: int = 4,
        gate_init_bias: float = -1.5,
        gate_mode: str = "per_step",
    ) -> None:
        super().__init__()
        self.num_steps = int(num_steps)
        self.num_commands = int(num_commands)
        self.gate_mode = str(gate_mode or "per_step").lower()
        if self.gate_mode not in {"scalar", "per_step"}:
            raise ValueError("gate_mode must be 'scalar' or 'per_step'")
        self.evidence_encoder = L1EvidenceEncoder(
            feature_dim=evidence_feature_dim,
            hidden_size=evidence_hidden_size,
            num_layers=evidence_layers,
            num_heads=evidence_heads,
        )
        base_dim = self.num_steps * 2 + self.num_commands
        fusion_dim = qwen_hidden_size + evidence_hidden_size + base_dim
        mid = max(qwen_hidden_size // 2, 256)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, mid),
            nn.GELU(),
            nn.LayerNorm(mid),
            nn.Linear(mid, mid),
            nn.GELU(),
            nn.LayerNorm(mid),
        )
        self.delta_wp = nn.Linear(mid, self.num_steps * 2)
        self.delta_cmd = nn.Linear(mid, self.num_commands)
        gate_out_dim = self.num_steps if self.gate_mode == "per_step" else 1
        self.gate = nn.Linear(mid, gate_out_dim)
        nn.init.zeros_(self.delta_wp.weight)
        nn.init.zeros_(self.delta_wp.bias)
        nn.init.zeros_(self.delta_cmd.weight)
        nn.init.zeros_(self.delta_cmd.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_init_bias))

    def forward(
        self,
        pooled: torch.Tensor,
        object_tokens: torch.Tensor,
        object_mask: torch.Tensor,
        base_waypoints: torch.Tensor,
        base_command_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        object_tokens = object_tokens.to(device=pooled.device, dtype=pooled.dtype)
        object_mask = object_mask.to(device=pooled.device)
        base_wp = base_waypoints.to(device=pooled.device, dtype=pooled.dtype)
        base_logits = base_command_logits.to(device=pooled.device, dtype=pooled.dtype)
        evidence = self.evidence_encoder(object_tokens, object_mask)
        base_flat = torch.cat([base_wp.flatten(1), base_logits], dim=-1)
        fused = self.fusion(torch.cat([pooled, evidence, base_flat], dim=-1))
        delta_wp = self.delta_wp(fused).view(-1, self.num_steps, 2)
        delta_cmd = self.delta_cmd(fused)
        gate = torch.sigmoid(self.gate(fused))
        if self.gate_mode == "scalar":
            wp_gate = gate.view(-1, 1, 1)
            cmd_gate = gate
        else:
            wp_gate = gate.view(-1, self.num_steps, 1)
            cmd_gate = gate.mean(dim=-1, keepdim=True)
        pred_wp = base_wp + wp_gate * delta_wp
        pred_cmd = base_logits + cmd_gate * delta_cmd
        return pred_wp, pred_cmd, gate, delta_wp


class EvidenceDirectedResidualFusionHead(nn.Module):
    """Object-query residual head with route-conditioned evidence cross-attention."""

    def __init__(
        self,
        qwen_hidden_size: int,
        evidence_feature_dim: int,
        num_commands: int,
        num_steps: int = 6,
        evidence_hidden_size: int = 256,
        evidence_layers: int = 2,
        evidence_heads: int = 4,
        gate_init_bias: float = -1.5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_steps = int(num_steps)
        self.num_commands = int(num_commands)
        self.evidence_encoder = L1EvidenceEncoder(
            feature_dim=evidence_feature_dim,
            hidden_size=evidence_hidden_size,
            num_layers=evidence_layers,
            num_heads=evidence_heads,
            dropout=dropout,
        )
        self.qwen_proj = nn.Sequential(
            nn.Linear(qwen_hidden_size, evidence_hidden_size),
            nn.GELU(),
            nn.LayerNorm(evidence_hidden_size),
        )
        self.base_context = nn.Sequential(
            nn.Linear(self.num_steps * 2 + self.num_commands, evidence_hidden_size),
            nn.GELU(),
            nn.LayerNorm(evidence_hidden_size),
        )
        self.query_embed = nn.Parameter(torch.zeros(self.num_steps + 1, evidence_hidden_size))
        nn.init.normal_(self.query_embed, mean=0.0, std=0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=evidence_hidden_size,
            num_heads=max(1, int(evidence_heads)),
            dropout=float(dropout),
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(evidence_hidden_size)
        self.delta_wp = nn.Linear(evidence_hidden_size, 2)
        self.delta_cmd = nn.Linear(evidence_hidden_size, self.num_commands)
        self.gate = nn.Linear(evidence_hidden_size, 1)
        nn.init.zeros_(self.delta_wp.weight)
        nn.init.zeros_(self.delta_wp.bias)
        nn.init.zeros_(self.delta_cmd.weight)
        nn.init.zeros_(self.delta_cmd.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_init_bias))

    def forward(
        self,
        pooled: torch.Tensor,
        object_tokens: torch.Tensor,
        object_mask: torch.Tensor,
        base_waypoints: torch.Tensor,
        base_command_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        object_tokens = object_tokens.to(device=pooled.device, dtype=pooled.dtype)
        object_mask = object_mask.to(device=pooled.device)
        base_wp = base_waypoints.to(device=pooled.device, dtype=pooled.dtype)
        base_logits = base_command_logits.to(device=pooled.device, dtype=pooled.dtype)
        encoded_objects = self.evidence_encoder.encode_tokens(object_tokens, object_mask)
        qwen_token = self.qwen_proj(pooled).unsqueeze(1)
        base_token = self.base_context(torch.cat([base_wp.flatten(1), base_logits], dim=-1)).unsqueeze(1)
        memory = torch.cat([encoded_objects, qwen_token, base_token], dim=1)
        valid = object_mask.to(device=pooled.device, dtype=torch.bool)
        memory_mask = torch.cat(
            [
                valid,
                torch.ones((valid.shape[0], 2), dtype=torch.bool, device=valid.device),
            ],
            dim=1,
        )
        queries = self.query_embed.unsqueeze(0).expand(pooled.shape[0], -1, -1)
        attended, _ = self.cross_attn(
            queries,
            memory,
            memory,
            key_padding_mask=~memory_mask,
            need_weights=False,
        )
        attended = self.query_norm(attended + queries)
        waypoint_queries = attended[:, : self.num_steps]
        command_query = attended[:, self.num_steps]
        delta_wp = self.delta_wp(waypoint_queries)
        delta_cmd = self.delta_cmd(command_query)
        gate = torch.sigmoid(self.gate(waypoint_queries)).squeeze(-1)
        pred_wp = base_wp + gate.unsqueeze(-1) * delta_wp
        pred_cmd = base_logits + gate.mean(dim=-1, keepdim=True) * delta_cmd
        return pred_wp, pred_cmd, gate, delta_wp, delta_cmd
