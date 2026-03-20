from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.models.attention import ProjectionHead
from src.models.backbone import VLMBackbone


class MultiIndexRetriever(nn.Module):
    def __init__(
        self,
        model_name: str,
        freeze_vision: bool = True,
        freeze_text: bool = True,
        num_db_indices: int = 8,
        db_dim: int = 256,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_attn_layers: int = 4,
        num_cls_tokens: int = 1,
        cls_mode: str = "append",
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if cls_mode not in ("project", "append"):
            raise ValueError(f"cls_mode must be 'project' or 'append', got '{cls_mode}'")

        self.num_db_indices = num_db_indices
        self.num_cls_tokens = num_cls_tokens
        self.cls_mode = cls_mode
        self.hidden_dim = hidden_dim

        self.backbone = VLMBackbone(model_name, freeze_vision, freeze_text)

        if cls_mode == "project":
            self.vision_cls_proj = nn.Linear(
                self.backbone.vision_hidden_size, num_cls_tokens * hidden_dim
            )
            self.text_cls_proj = nn.Linear(
                self.backbone.text_hidden_size, num_cls_tokens * hidden_dim
            )
        else:
            self.vision_cls_proj = nn.Linear(self.backbone.vision_hidden_size, hidden_dim)
            self.text_cls_proj = nn.Linear(self.backbone.text_hidden_size, hidden_dim)
            if num_cls_tokens > 1:
                self.vision_aux_cls = nn.Parameter(
                    torch.empty(num_cls_tokens - 1, hidden_dim)
                )
                self.text_aux_cls = nn.Parameter(
                    torch.empty(num_cls_tokens - 1, hidden_dim)
                )

        self.db_queries = nn.Parameter(torch.empty(num_db_indices, hidden_dim))
        self.router_token = nn.Parameter(torch.empty(1, hidden_dim))

        self.vision_head = ProjectionHead(
            hidden_dim, num_heads, num_attn_layers, mlp_ratio, dropout
        )
        self.text_head = ProjectionHead(
            hidden_dim, num_heads, num_attn_layers, mlp_ratio, dropout
        )

        self.vision_db_proj = nn.Linear(hidden_dim, db_dim)
        self.text_db_proj = nn.Linear(hidden_dim, db_dim)
        self.router_head = nn.Linear(hidden_dim, num_db_indices)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / 0.07))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.db_queries, std=0.02)
        nn.init.trunc_normal_(self.router_token, std=0.02)

        if self.cls_mode == "append" and self.num_cls_tokens > 1:
            nn.init.trunc_normal_(self.vision_aux_cls, std=0.02)
            nn.init.trunc_normal_(self.text_aux_cls, std=0.02)

        for module in (
            self.vision_cls_proj, self.text_cls_proj,
            self.vision_db_proj, self.text_db_proj, self.router_head,
        ):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        for head in (self.vision_head, self.text_head):
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def _make_cls_tokens(
        self,
        pooled: torch.Tensor,
        proj: nn.Linear,
        aux_cls: nn.Parameter | None,
    ) -> torch.Tensor:
        """(B, backbone_dim) -> (B, num_cls_tokens, hidden_dim)"""
        B = pooled.shape[0]
        if self.cls_mode == "project":
            return proj(pooled).view(B, self.num_cls_tokens, self.hidden_dim)
        backbone_tok = proj(pooled).unsqueeze(1)  # (B, 1, D)
        if self.num_cls_tokens == 1:
            return backbone_tok
        aux = aux_cls.unsqueeze(0).expand(B, -1, -1)  # (B, K-1, D)
        return torch.cat([backbone_tok, aux], dim=1)  # (B, K, D)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, N, db_dim)"""
        B = pixel_values.shape[0]
        pooled = self.backbone.encode_vision(pixel_values)
        aux = getattr(self, "vision_aux_cls", None)
        cls_tokens = self._make_cls_tokens(pooled, self.vision_cls_proj, aux)

        db_q = self.db_queries.unsqueeze(0).expand(B, -1, -1)
        seq = torch.cat([db_q, cls_tokens], dim=1)

        seq = self.vision_head(seq)
        return self.vision_db_proj(seq[:, :self.num_db_indices])

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, L), (B, L) -> (B, N, db_dim), (B, N)"""
        B = input_ids.shape[0]
        pooled = self.backbone.encode_text(input_ids, attention_mask)
        aux = getattr(self, "text_aux_cls", None)
        cls_tokens = self._make_cls_tokens(pooled, self.text_cls_proj, aux)

        db_q = self.db_queries.unsqueeze(0).expand(B, -1, -1)
        router = self.router_token.unsqueeze(0).expand(B, -1, -1)
        seq = torch.cat([db_q, router, cls_tokens], dim=1)

        seq = self.text_head(seq)
        text_embeds = self.text_db_proj(seq[:, :self.num_db_indices])
        router_logits = self.router_head(seq[:, self.num_db_indices])
        return text_embeds, router_logits

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (image_embeds, text_embeds, router_logits, logit_scale)."""
        image_embeds = self.encode_image(pixel_values)
        text_embeds, router_logits = self.encode_text(input_ids, attention_mask)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        return image_embeds, text_embeds, router_logits, logit_scale
