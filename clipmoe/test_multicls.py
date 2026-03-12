import torch
import torch.nn as nn
import torch.distributed as dist
import tempfile
import clip
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import os
from model_clipmoe import build_model
from coco_loader2 import CocoForClipMoE, clip_collate_fn, data_path, ann_path
from model_clipmoe import CLIP
from torch.utils.data import DataLoader, Dataset, Subset

def recall_at_k(sim, k):
    ranks = torch.argsort(sim, dim=1, descending=True) # [N, N]
    targets = torch.arange(sim.size(0)).unsqueeze(1) # [N, 1]
    correct = (ranks[:, :k] == targets).any(dim=1) # [N]

    return correct.float().mean().item()

def evaluate(model, val_loader, device, epoch, num_experts=8):
    """Runs a quick forward pass on the validation set to check expert assignment."""
    model.eval()
    all_assignments = []
    all_img = []
    all_txt = []
    all_entropy = []

    print(f"Evaluating epoch {epoch}")
    with torch.no_grad():
        for images, captions, _ in val_loader:
            if num_experts > 0:
                image_features, router_logits = model.encode_image(images.to(device), router_output=True)
                router_probs = torch.softmax(router_logits, dim=-1) # [B, k]
                entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-10), dim=-1) # per image router entropy, [B]
                all_entropy.append(entropy.cpu()) # list of tensors [B]
                
                assignments = torch.argmax(router_logits, dim=-1) # per image expert assignment, [B]
                all_assignments.append(assignments.cpu()) # list of tensors [B]
            else:
                image_features = model.encode_image(images.to(device), router_output=False)
            
            texts = clip.tokenize(captions, context_length=model.context_length)
            text_features = model.encode_text(texts.to(device), router_output=False)

            # img and txt embeddings
            all_img.append(torch.nn.functional.normalize(image_features, dim=-1).cpu()) # list of tensors [B, embed_dim]
            all_txt.append(torch.nn.functional.normalize(text_features, dim=-1).cpu())

        img = torch.cat(all_img) # [128, embed_dim]
        txt = torch.cat(all_txt) # [128, embed_dim]
        sim = img @ txt.T

        metrics = {
            "eval/i2t_Recall@1": recall_at_k(sim, 1),
            "eval/i2t_Recall@10": recall_at_k(sim, 10),
            "eval/t2i_Recall@1": recall_at_k(sim.T, 1),
            "eval/t2i_Recall@10": recall_at_k(sim.T, 10),
            "epoch": epoch
        }

        if num_experts > 0:
            entropy_concat = torch.cat(all_entropy) # [128]
            assignments_concat = torch.cat(all_assignments) # [128]
            counts = assignments_concat.view(-1).bincount(minlength=num_experts)

            usage_table = wandb.Table(columns=["epoch", "expert", "count"])
            for i, c in enumerate(counts):
                usage_table.add_data(epoch, f"expert_{i}", c.item())

            metrics.update({
                "eval/expert_usage_histogram": wandb.Histogram(assignments_concat.numpy()),
                "eval/expert_usage_over_time": usage_table,
                "eval/router_entropy_histogram": wandb.Histogram(entropy_concat.numpy()),
            })
            print(f"Validation expert assignments: {counts.tolist()}")

        wandb.log(metrics)
        model.train()


def train_on_coco(model, train_loader, val_loader, device, epochs=5):
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.visual.cls_tokens: param.requires_grad = True
    for param in model.visual.proj.parameters(): param.requires_grad = True

    params_to_train = [
        {'params': model.visual.cls_tokens, 'lr': 2e-4, 'weight_decay': 0.01},
        {'params': model.visual.proj.parameters(), 'lr': 1e-4, 'weight_decay': 0.01}
    ]

    if model.num_experts > 0:
        params_to_train.append({'params': model.visual.router_head.parameters(), 'lr': 1e-4})
        for param in model.visual.router_head.parameters(): param.requires_grad = True

    optimizer = torch.optim.AdamW(params_to_train)
    
    print(f"\n--- Training on ({device}) | Epochs: {epochs} ---")

    for epoch in range(epochs):
        model.train()
        for i, (images, captions, captions_short) in enumerate(train_loader):
            optimizer.zero_grad()
            
            images = images.to(device, non_blocking=True)
            texts = clip.tokenize(captions, context_length=model.context_length).to(device, non_blocking=True) # tokenize on cpu, move to gpu
            texts_short = clip.tokenize(captions_short, context_length=model.context_length).to(device, non_blocking=True)

            # forward pass
            alpha = 0.01
            outputs = model(images, texts, texts_short, rank=0)

            if model.num_experts > 0:
                loss_itcl, loss_itcs, loss_div = outputs
            else:
                loss_itcl, loss_itcs = outputs
                loss_div = torch.tensor(0.0)
            
            loss_contr = loss_itcl + loss_itcs
            loss = loss_contr + alpha * loss_div

            loss.backward()

            if i % 5 == 0:
                log_data = {
                    "epoch": epoch,
                    "batch": i,
                    "train/loss": loss.item(),
                    "train/loss_contrastive": loss_contr.item(),
                }

                if model.num_experts > 0:
                    with torch.no_grad():
                        _, router_logits = model.encode_image(images, router_output=True)
                        router_probs = torch.softmax(router_logits, dim=-1)
                        max_probs, _ = torch.max(router_probs, dim=-1)
                        r_param = next(model.visual.router_head.parameters())

                        log_data["train/loss_diversity"] = loss_div.item()
                        log_data["train/router_mean_max_prob"] = max_probs.mean().item()
                        log_data["train/router_mean_entropy"] = -torch.sum(router_probs * torch.log(router_probs + 1e-10), dim=-1).mean().item()
                        log_data["train/router_grad_norm"] = r_param.grad.norm().item() if r_param.grad is not None else 0.0

                wandb.log(log_data)
                
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    
        evaluate(model, val_loader, device, epoch, num_experts=model.num_experts)

        checkpoint_dir = "/project/osprey/scratch/liv/repos/CLIP-MoE/clipmoe/checkpoints"
        checkpoint_path = os.path.join(checkpoint_dir, f"moe_v1_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")


def build_baseline_clip(device):
    """Builds a standard CLIP model using the same class structure but 1 CLS token."""
    print("Building baseline CLIP model (1 CLS)...")
    model = CLIP(
        embed_dim=768, image_resolution=224, vision_layers=24, vision_width=1024, vision_patch_size=14,
        context_length=77, vocab_size=49408, transformer_width=768, transformer_heads=12, transformer_layers=12,
        load_from_clip=False, num_experts=0, head_type="linear"
    ).to(device)
    model.initialize_parameters()

    # load official weights
    official_clip, _ = clip.load("ViT-L/14", device="cpu")
    official_sd = official_clip.state_dict()
    model_sd = model.state_dict()

    # 1-to-1 weight copy, exclude cls token and proj head
    load_count = 0
    for k, v in official_sd.items():
        if k in model_sd and v.shape == model_sd[k].shape:
            if "visual.cls_tokens" in k or "visual.proj" in k or "visual.positional_embedding" in k:
                continue
            model_sd[k].copy_(v)
            load_count += 1    

    if "visual.positional_embedding" in official_sd:
        v = official_sd["visual.positional_embedding"]
        model_sd["visual.positional_embedding"].data[1:257].copy_(v.data[1:257])
        print("  [Fixed PE] Loaded patch embeddings 1-256; CLS PE (0) remains random")

    print(f"Baseline backbone loaded with {load_count} layers. CLS token and Proj are RANDOM.")
    return model


def run_multicls_test():
    # mock distributed environment - create a temporary file for the process group to "sync"
    if not dist.is_initialized():
        tmp_file = tempfile.NamedTemporaryFile(delete=False).name
        file_url = "file:///" + tmp_file.replace("\\", "/")
        dist.init_process_group(backend='gloo', init_method=file_url, rank=0, world_size=1)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_experts = 8

    if num_experts > 0:
        # bypass build_model() with no state_dict
        print("1. Building CLIP model with Multi-CLS...")
        model = CLIP(
            embed_dim=768, image_resolution=224, vision_layers=24, vision_width=1024, vision_patch_size=14,
            context_length=77, vocab_size=49408, transformer_width=768, transformer_heads=12, transformer_layers=12,
            load_from_clip=False, MoE_args=None, num_experts=num_experts, head_type="", use_short_text=False
        ).to(device)
        model.initialize_parameters()

        # load official weights into a temporary object
        print("--- Manually injecting openai weights ---")
        official_clip, _ = clip.load("ViT-L/14", device="cpu")
        official_sd = official_clip.state_dict()
        model_sd = model.state_dict()

        # filtered dictionary of only matching layers
        load_count = 0
        for k, v in official_sd.items():
            if k in model_sd:
                if v.shape == model_sd[k].shape:
                    model_sd[k].copy_(v)
                    load_count += 1
                elif "visual.positional_embedding" in k:
                    # v is [257, 1024], model_sd[k] is [265, 1024]
                    model_sd[k].data[num_experts+1:].copy_(v.data[1:])
                    print(f"  [Fixed PE] {k}: Loaded patches to indices {num_experts+1}-264")
                    load_count += 1
                else:
                    print(f"  [Skip] {k}: Shape mismatch ({v.shape} vs {model_sd[k].shape})")
    else:
        model = build_baseline_clip(device)

    # verify the backbone is not random
    conv_mean = model.visual.conv1.weight.mean().item()
    print(f"Final Pre-train Check (Conv1): {conv_mean:.6f}")

    _, preprocess = clip.load("ViT-L/14", device="cpu", jit=False)
    dataset = CocoForClipMoE(root=data_path, annFile=ann_path, transform=preprocess)
    indices = torch.randperm(len(dataset)).tolist()

    train_loader = DataLoader(Subset(dataset, indices[:1000]), batch_size=32, shuffle=True, collate_fn=clip_collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(Subset(dataset, indices[1000:1128]), batch_size=32, shuffle=False, collate_fn=clip_collate_fn)

    wandb.init(
        project="CLIP-MoE-COCO",
        name="moe-8-experts-0.01-lb",
        config={
            "learning_rate_experts": 2e-4,
            "learning_rate_router": 1e-4,
            "learning_rate_proj": 1e-4,
            "num_experts": 8,
            "top_k": 1,
            "alpha": 0.01,
            "epochs": 10,
            "batch_size": 32,
        }
    )

    train_on_coco(model, train_loader, val_loader, device, epochs=10)

if __name__ == "__main__":
    run_multicls_test()