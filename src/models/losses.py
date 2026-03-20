from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F


def gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather across GPUs, preserving gradients on the local shard."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    gathered[dist.get_rank()] = tensor
    return torch.cat(gathered, dim=0)


def multi_index_contrastive_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    router_logits: torch.Tensor,
    logit_scale: torch.Tensor,
    router_temperature: float = 1.0,
    label_smoothing: float = 0.0,
    gather: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Symmetric contrastive loss with per-query routed index aggregation.

    sim(i, j) = logit_scale * sum_k[ w_k^i * cos(text_k^i, image_k^j) ]
    where w^i = softmax(router_logits^i / temperature)
    """
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    if gather:
        image_embeds = gather_with_grad(image_embeds)
        text_embeds = gather_with_grad(text_embeds)
        router_logits = gather_with_grad(router_logits)

    B = image_embeds.shape[0]

    router_weights = F.softmax(router_logits / router_temperature, dim=-1)

    # sims[i, k, j] = cos(text[i,k], image[j,k])
    sims = torch.einsum("bnd,jnd->bnj", text_embeds, image_embeds)

    # weighted aggregation per query -> (B, B)
    weighted_sims = (router_weights.unsqueeze(-1) * sims).sum(dim=1) * logit_scale

    labels = torch.arange(B, device=weighted_sims.device)
    loss_t2i = F.cross_entropy(weighted_sims, labels, label_smoothing=label_smoothing)
    loss_i2t = F.cross_entropy(weighted_sims.t(), labels, label_smoothing=label_smoothing)
    loss = (loss_t2i + loss_i2t) / 2.0

    with torch.no_grad():
        t2i_acc = (weighted_sims.argmax(-1) == labels).float().mean()
        i2t_acc = (weighted_sims.t().argmax(-1) == labels).float().mean()
        router_entropy = -(router_weights * (router_weights + 1e-8).log()).sum(-1).mean()

    metrics = {
        "loss": loss.detach(),
        "loss_t2i": loss_t2i.detach(),
        "loss_i2t": loss_i2t.detach(),
        "acc_t2i": t2i_acc,
        "acc_i2t": i2t_acc,
        "logit_scale": logit_scale.detach(),
        "router_entropy": router_entropy,
    }
    return loss, metrics
