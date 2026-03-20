from __future__ import annotations

from typing import Any

import lightning as L
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR

from src.models.losses import multi_index_contrastive_loss
from src.models.multi_index_retriever import MultiIndexRetriever


class MultiIndexRetrievalModule(L.LightningModule):
    def __init__(
        self,
        # backbone
        model_name: str = "openai/clip-vit-large-patch14",
        freeze_vision: bool = True,
        freeze_text: bool = True,
        # retriever
        num_db_indices: int = 8,
        db_dim: int = 256,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_attn_layers: int = 4,
        num_cls_tokens: int = 1,
        cls_mode: str = "append",
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        # loss
        router_temperature: float = 1.0,
        label_smoothing: float = 0.0,
        gather_distributed: bool = False,
        # optimizer
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        # scheduler
        t_max: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = MultiIndexRetriever(
            model_name=model_name,
            freeze_vision=freeze_vision,
            freeze_text=freeze_text,
            num_db_indices=num_db_indices,
            db_dim=db_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_attn_layers=num_attn_layers,
            num_cls_tokens=num_cls_tokens,
            cls_mode=cls_mode,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.router_temperature = router_temperature
        self.label_smoothing = label_smoothing
        self.gather_distributed = gather_distributed
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.t_max = t_max

    def forward(self, pixel_values, input_ids, attention_mask):
        return self.model(pixel_values, input_ids, attention_mask)

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str):
        image_embeds, text_embeds, router_logits, logit_scale = self.model(
            batch["pixel_values"], batch["input_ids"], batch["attention_mask"]
        )
        loss, metrics = multi_index_contrastive_loss(
            image_embeds,
            text_embeds,
            router_logits,
            logit_scale,
            router_temperature=self.router_temperature,
            label_smoothing=self.label_smoothing,
            gather=self.gather_distributed and stage == "train",
        )
        for k, v in metrics.items():
            self.log(
                f"{stage}/{k}",
                v,
                on_step=(stage == "train"),
                on_epoch=True,
                prog_bar=(k in ("loss", "acc_t2i")),
                sync_dist=(stage != "train"),
                batch_size=batch["pixel_values"].shape[0],
            )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self) -> dict[str, Any]:
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=self.warmup_steps)
        cosine = CosineAnnealingWarmRestarts(optimizer, T_0=self.t_max)
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[self.warmup_steps]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
