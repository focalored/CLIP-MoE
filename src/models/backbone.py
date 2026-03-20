from __future__ import annotations

import torch.nn as nn
from transformers import AutoModel


class VLMBackbone(nn.Module):
    """Loads paired vision+text encoders from a HuggingFace VLM checkpoint."""

    SUPPORTED_ARCHITECTURES = {"CLIPModel", "SiglipModel", "Siglip2Model"}

    def __init__(
        self,
        model_name: str,
        freeze_vision: bool = True,
        freeze_text: bool = True,
    ):
        super().__init__()
        model = AutoModel.from_pretrained(model_name)
        arch = type(model).__name__
        if arch not in self.SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Architecture {arch} from {model_name} not supported. "
                f"Expected one of {self.SUPPORTED_ARCHITECTURES}."
            )

        self.vision_encoder = model.vision_model
        self.text_encoder = model.text_model
        self.vision_hidden_size: int = self.vision_encoder.config.hidden_size
        self.text_hidden_size: int = self.text_encoder.config.hidden_size

        self._set_grad(self.vision_encoder, not freeze_vision)
        self._set_grad(self.text_encoder, not freeze_text)

    @staticmethod
    def _set_grad(module: nn.Module, requires_grad: bool) -> None:
        for p in module.parameters():
            p.requires_grad = requires_grad

    def encode_vision(self, pixel_values):
        """(B, C, H, W) -> (B, vision_hidden_size)"""
        return self.vision_encoder(pixel_values=pixel_values).pooler_output

    def encode_text(self, input_ids, attention_mask):
        """(B, L), (B, L) -> (B, text_hidden_size)"""
        return self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).pooler_output
