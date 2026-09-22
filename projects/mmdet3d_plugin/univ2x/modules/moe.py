import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.activation import build_activation_layer


class Expert(nn.Module):
    def __init__(self, d_model, d_ff, act_cfg=dict(type='ReLU', inplace=True), ffn_dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.act = build_activation_layer(act_cfg)
        self.dropout = nn.Dropout(ffn_dropout) if ffn_dropout > 0 else None
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.act(self.w1(x))
        if self.dropout is not None:
            x = self.dropout(x)
        return self.w2(x)


class MoE(nn.Module):
    """Small UniMM-V2X-style sparse FFN used as a drop-in transformer FFN."""

    def __init__(self,
                 d_model,
                 num_experts,
                 top_k,
                 d_ff=2048,
                 act_cfg=dict(type='ReLU', inplace=True),
                 ffn_dropout=0.0,
                 load_balance_loss_weight=0.0):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_loss_weight = load_balance_loss_weight
        self.gate = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, act_cfg=act_cfg, ffn_dropout=ffn_dropout)
            for _ in range(num_experts)
        ])
        self.moe_load_balance_loss = None

    def forward(self, x, residual=None):
        original_shape = x.shape
        x_flat = x.reshape(-1, self.d_model)
        num_tokens = x_flat.shape[0]
        if num_tokens == 0:
            self.moe_load_balance_loss = x_flat.sum() * 0.0
            return x

        gate_logits = self.gate(x_flat)
        if self.training:
            gumbel_noise = -torch.empty_like(gate_logits).exponential_().log()
            routing_logits = gate_logits + gumbel_noise
        else:
            routing_logits = gate_logits
        topk_weights, topk_indices = torch.topk(routing_logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1)

        output_flat = torch.zeros_like(x_flat)
        flat_topk_indices = topk_indices.reshape(-1)
        repeated_x = x_flat.unsqueeze(1).expand(-1, self.top_k, -1).reshape(-1, self.d_model)
        token_indices = torch.arange(num_tokens, device=x.device)
        token_indices = token_indices.unsqueeze(1).expand(-1, self.top_k).reshape(-1)

        for expert_idx, expert in enumerate(self.experts):
            expert_mask = flat_topk_indices == expert_idx
            if not torch.any(expert_mask):
                continue
            expert_out = expert(repeated_x[expert_mask])
            expert_weight = topk_weights.reshape(-1)[expert_mask].unsqueeze(-1)
            output_flat.scatter_add_(
                0,
                token_indices[expert_mask].unsqueeze(-1).expand(-1, self.d_model),
                expert_out * expert_weight,
            )

        probs = F.softmax(gate_logits, dim=-1)
        expert_load = torch.zeros(self.num_experts, device=x.device, dtype=x.dtype)
        expert_load.scatter_add_(0, flat_topk_indices, topk_weights.reshape(-1))
        importance = probs.mean(dim=0)
        load = expert_load / float(num_tokens)
        aux = ((importance - importance.mean()) ** 2).mean() + ((load - load.mean()) ** 2).mean()
        self.moe_load_balance_loss = self.load_balance_loss_weight * aux

        if residual is not None:
            identity = residual
        else:
            identity = x
        output = output_flat.view(original_shape)
        return output + identity

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        old_w1 = state_dict.get(prefix + 'layers.0.0.weight')
        old_b1 = state_dict.get(prefix + 'layers.0.0.bias')
        old_w2 = state_dict.get(prefix + 'layers.1.weight')
        old_b2 = state_dict.get(prefix + 'layers.1.bias')
        if old_w1 is not None and old_b1 is not None and old_w2 is not None and old_b2 is not None:
            for idx in range(self.num_experts):
                state_dict[prefix + f'experts.{idx}.w1.weight'] = old_w1.detach().clone()
                state_dict[prefix + f'experts.{idx}.w1.bias'] = old_b1.detach().clone()
                state_dict[prefix + f'experts.{idx}.w2.weight'] = old_w2.detach().clone()
                state_dict[prefix + f'experts.{idx}.w2.bias'] = old_b2.detach().clone()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
