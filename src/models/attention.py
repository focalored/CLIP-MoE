#todo get rid of the eric written attn and steal it from someone who knows what they're doing

from __future__ import annotations

import torch
import torch.nn as nn


class SelfAttentionBlock(nn.Module):
    """Pre-norm transformer block (LN -> MHSA -> residual -> LN -> FFN -> residual)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ProjectionHead(nn.Module):
    """Stack of SelfAttentionBlocks with a final layer norm."""

    def __init__(
        self, hidden_dim: int, num_heads: int, num_layers: int,
        mlp_ratio: float = 4.0, dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [SelfAttentionBlock(hidden_dim, num_heads, mlp_ratio, dropout)
             for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
