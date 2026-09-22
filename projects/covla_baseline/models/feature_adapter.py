"""Reserved V2X-feature adapter for Phase 2."""
from __future__ import annotations

import torch
from torch import nn


class InfraFeatureAdapter(nn.Module):
    """Compress infrastructure-side visual tokens into K learned tokens.

    This is a minimal interface placeholder. Phase 1 does not use it; Phase 2
    can replace the pooling/projection with a Perceiver-style compressor.
    """

    def __init__(self, input_dim: int, output_dim: int, num_tokens: int = 16) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.query_tokens = nn.Parameter(torch.randn(num_tokens, output_dim) * 0.02)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, infra_tokens: torch.Tensor) -> torch.Tensor:
        if infra_tokens.ndim != 3:
            raise ValueError(f"infra_tokens must be [B, N, C], got shape={tuple(infra_tokens.shape)}")
        pooled = self.proj(infra_tokens.mean(dim=1))
        return pooled[:, None, :] + self.query_tokens[None, :, :]
