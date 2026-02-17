import torch
import torch.distributed as dist
import tempfile
import os
from model_clipmoe import build_model

def run_original_test():
    # Mock distributed environment - create a temporary file for the process group to "sync"
    if not dist.is_initialized():
        tmp_file = tempfile.NamedTemporaryFile(delete=False).name
        file_url = "file:///" + tmp_file.replace("\\", "/")
        dist.init_process_group(backend='gloo', init_method=file_url, rank=0, world_size=1)

    # Mock state_dict (standard CLIP)
    mock_sd = {
        "visual.proj": torch.randn(1024, 768),
        "visual.conv1.weight": torch.randn(1024, 3, 14, 14),
        "visual.positional_embedding": torch.randn(257, 1024),
        "visual.class_embedding": torch.randn(1024),
        "text_projection": torch.randn(768, 768),
        "positional_embedding": torch.randn(248, 768),
        "positional_embedding_res": torch.randn(248, 768),
        "token_embedding.weight": torch.randn(49408, 768),
        "ln_final.weight": torch.randn(768),
        "ln_final.bias": torch.randn(768),
        "logit_scale": torch.ones([]), # Added
        "transformer.resblocks.0.attn.in_proj_weight": torch.randn(2304, 768), 
    }

    # 2. Build model with MoE_args=None
    print("1. Building CLIP model...")
    model = build_model(mock_sd, load_from_clip=False, MoE_args=None)
    model.train()

    # Dummy inputs
    batch_size = 2
    dummy_img = torch.randn(batch_size, 3, 224, 224).type(model.dtype)
    dummy_txt = torch.randint(0, 49408, (batch_size, 248))

    # A. Forward Pass
    print("2. Running forward pass...")
    outputs = model(dummy_img, dummy_txt, dummy_txt, rank=0)
    loss_itcl, loss_itcs = outputs
    
    print(f"✅ Forward Pass Successful!")
    print(f"   Contrastive Loss: {loss_itcl.item():.4f}")

    # B. Backward Pass
    print("3. Running backward pass...")
    loss_itcl.backward()
    print("✅ Backward Pass Successful!")

    # C. Verify visual output shape
    with torch.no_grad():
        print("4. Verifying model.visual() output shape...")
        visual_features = model.visual(dummy_img)
        print(f"Image input shape:  {dummy_img.shape}")
        print(f"Visual output shape: {visual_features.shape}")
        
        expected_dim = 768
        if visual_features.shape == (batch_size, expected_dim):
            print(f"✅ Status: standard CLIP output (single CLS token pooled)")
        else:
            print(f"❓ Status: Unexpected shape {visual_features.shape}")

if __name__ == "__main__":
    run_original_test()