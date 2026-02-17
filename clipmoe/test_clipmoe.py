import torch
import torch.nn.functional as F
import os
import torch.distributed as dist
import tempfile
from model_clipmoe import build_model

def run_test():
    # Mock distributed environment - create a temporary file for the process group to "sync"
    if not dist.is_initialized():
        tmp_file = tempfile.NamedTemporaryFile(delete=False).name
        file_url = "file:///" + tmp_file.replace("\\", "/")
        try:
            dist.init_process_group(backend='gloo', init_method=file_url, rank=0, world_size=1)
        except Exception as e:
            print(f"Failed to init distributed: {e}")
            return

    # Mock state_dict (ViT-L/14: width 1024, patch 14)
    mock_sd = {
        "visual.proj": torch.randn(1024, 768),
        "visual.conv1.weight": torch.randn(1024, 3, 14, 14),
        "visual.positional_embedding": torch.randn(257, 1024),
        "text_projection": torch.randn(768, 768),
        "positional_embedding": torch.randn(77, 768),
        "token_embedding.weight": torch.randn(49408, 768),
        "ln_final.weight": torch.randn(768),
        "ln_final.bias": torch.randn(768),
        "transformer.resblocks.0.attn.in_proj_weight": torch.randn(2304, 768), 
    }

    # [num_experts, top_k, dropout, moe_layers]
    moe_args = [4, 2, 0.1, 1] 
    print("1. Building model...")
    model = build_model(mock_sd, load_from_clip=False, MoE_args=moe_args)
    model.train() # Enable gradients

    # Dummy inputs
    batch_size = 2
    dummy_img = torch.randn(batch_size, 3, 224, 224)
    dummy_txt = torch.randint(0, 49408, (batch_size, 248))
    rank = 0 # Mock rank for distributed logic

    # A. Forward pass
    print("2. Running forward pass...")
    try:
        # forward() in model_clipmoe.py takes (image, text_long, text_short, rank)
        outputs = model(dummy_img, dummy_txt, dummy_txt, rank)
        
        loss_itcl, loss_itcs, img_bal, txt_bal, _ = outputs
        
        # Combine losses to test backprop
        total_loss = loss_itcl + img_bal + txt_bal
        
        print(f"✅ Forward pass successful!")
        print(f"   Contrastive Loss: {loss_itcl.item():.4f}")
        print(f"   MoE Balance Loss: {img_bal.item():.4f}")

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # B. Backward pass
    print("3. Running backward pass...")
    try:
        total_loss.backward()
        print("✅ Backward pass successful! Gradients computed.")
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")

if __name__ == "__main__":
    run_test()