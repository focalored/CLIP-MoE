import torch
import torch.nn as nn
import torch.distributed as dist
import tempfile
from model_clipmoe import build_model

def run_multicls_test():
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
        "logit_scale": torch.ones([]),
        "transformer.resblocks.0.attn.in_proj_weight": torch.randn(2304, 768), 
    }

    # 2. Build model with MoE_args=None
    print("1. Building CLIP model with Multi-CLS...")
    model = build_model(mock_sd, load_from_clip=False, MoE_args=None, multi_cls=True)
    model.train()

    # Dummy inputs
    dummy_img = torch.randn(1, 3, 224, 224).half()
    model.half()

    # 3. Forward pass
    print("2. Running Multi-CLS visual forward pass...")
    with torch.no_grad():
        router_token, expert_tokens = model.visual(dummy_img)
    
    print(f"✅ Visual Wrapper Forward Pass Successful!")
    print(f"Router token shape: {router_token.shape} (Expected: [1, 1, 1024])")
    print(f"Expert tokens shape: {expert_tokens.shape} (Expected: [1, 8, 1024])")

if __name__ == "__main__":
    run_multicls_test()