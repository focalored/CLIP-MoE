import torch
import torch.nn as nn
import torch.distributed as dist
import tempfile
import clip
from model_clipmoe import build_model
from coco_loader2 import CocoForClipMoE, clip_collate_fn, data_path, ann_path
from torch.utils.data import DataLoader, Subset

def train_on_coco(model, num_samples=1000, batch_size=32, epochs=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    _, preprocess = clip.load("ViT-L/14", device="cpu", jit=False) 
    full_dataset = CocoForClipMoE(root=data_path, annFile=ann_path, transform=preprocess)
    indices = torch.arange(min(num_samples, len(full_dataset)))
    coco_subset = Subset(full_dataset, indices)
    
    loader = DataLoader(
        coco_subset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=clip_collate_fn,
        num_workers=4,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    
    print(f"\n--- Training on ({device}) | {num_samples} samples ---")

    for epoch in range(epochs):
        model.train()
        for i, (images, captions, captions_short) in enumerate(loader):
            optimizer.zero_grad()
            
            images = images.to(device, non_blocking=True)
            # tokenize on CPU, move to GPU
            texts = clip.tokenize(captions).to(device, non_blocking=True)
            texts_short = clip.tokenize(captions_short).to(device, non_blocking=True)

            # forward pass
            loss, _ = model(images, texts, texts_short, rank=0)
            
            if torch.isnan(loss):
                print(f"Batch {i}: NaN Loss detected. Check init scales.")
                return

            loss.backward()
            
            # log gradient norms
            if i % 5 == 0:
                with torch.no_grad():
                    # check the router head gradients
                    r_grad = next(model.visual.router_head.parameters()).grad.norm().item()
                    print(f"Ep {epoch} | Bt {i:02d} | Loss: {loss.item():.4f} | Router Grad Norm: {r_grad:.6f}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

def router_sanity_check(model):
    # train everything
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
  
    # dummy inputs
    fixed_img = torch.zeros(2, 3, 224, 224)
    fixed_img[0] += torch.randn(3, 224, 224) * 0.5 # noise
    fixed_img[1] += 1.0 # pure white image
    fixed_txt = torch.zeros(2, 248, dtype=torch.long)
    fixed_txt[0, :5] = torch.tensor([49406, 123, 456, 789, 49407])
    fixed_txt[1, :5] = torch.tensor([49406, 987, 654, 321, 49407]) 
    
    print("\n--- Starting scratch-training (50 Steps) ---")
    
    for step in range(51):
        optimizer.zero_grad()
        
        loss, _ = model(fixed_img, fixed_txt, fixed_txt, rank=0)
        # loss = image_features.pow(2).mean()
        if torch.isnan(loss):
            print(f"CRASHED at step {step}. Loss is NaN. Check initialization scales")
            return 0.0
        
        loss.backward()
        print(f"Loss Grad Check: {loss.grad_fn}")
        
        if step % 10 == 0:
            with torch.no_grad():
                image_features, router_logits = model.visual(fixed_img)
            # check difference between the router's decision for the two images
                diff = (router_logits[0] - router_logits[1]).abs().mean().item()
                print(f"Step {step:02d} | Loss: {loss.item():.4f} | Router Divergence: {diff:.6f}")
            
            print(f"\n--- Gradient Flow Check (Step {step}) ---")
            conv_grad = model.visual.conv1.weight.grad.abs().mean().item()
            cls_grad = model.visual.cls_tokens.grad.abs().mean().item()
            router_grad = next(model.visual.router_head.parameters()).grad.abs().mean().item()
            proj_grad = model.text_projection.grad.abs().mean().item()

            print(f"  Conv1 Grad:   {conv_grad:.16f}")
            print(f"  CLS Token Grad: {cls_grad:.8f}")
            print(f"  Router Grad:  {router_grad:.8f}")
            print(f"  Text Proj Grad: {proj_grad:.8f}")
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
    return diff

def run_multicls_test():
    # mock distributed environment - create a temporary file for the process group to "sync"
    if not dist.is_initialized():
        # tmp_file = tempfile.NamedTemporaryFile(delete=False).name
        # file_url = "file:///" + tmp_file.replace("\\", "/")
        dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', rank=0, world_size=1)

    template_sd = {
        "visual.proj": torch.randn(1024, 768),
        "visual.conv1.weight": torch.randn(1024, 3, 14, 14),
        "visual.positional_embedding": torch.randn(257, 1024),
        "visual.class_embedding": torch.randn(1024),
        "text_projection": torch.randn(768, 768),
        "positional_embedding": torch.randn(248, 768),
        "token_embedding.weight": torch.randn(49408, 768),
        "ln_final.weight": torch.randn(768),
        "ln_final.bias": torch.randn(768),
        "logit_scale": torch.ones([]),
    } 

    # build model with no state_dict
    print("1. Building CLIP model with Multi-CLS...")
    model = build_model(state_dict=template_sd, load_from_clip=False, MoE_args=None, multi_cls=True, head_type="linear")
    model.initialize_parameters()

    # verify initial state of ViT
    # dummy_img = torch.randn(2, 3, 224, 224)
    # with torch.no_grad():
    #     image_embedding, router_logits = model.visual(dummy_img)
    #     norm = image_embedding.norm(dim=1)

    # print(f"✅ ViT initialization check:")
    # print(f"   Embed norms: {norm.tolist()} (should be non-zero)")
    # if torch.any(norm == 0):
    #     print("   ❌ ERROR: Zero norms detected! Check projection layer initialization.")
    #     return
    
    train_on_coco(model)

if __name__ == "__main__":
    run_multicls_test()