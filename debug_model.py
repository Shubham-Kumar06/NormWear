import torch
import sys
import os
from modules.normwear import NormWear

def debug():
    print("Checking CUDA...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    print("Initializing model...")
    model = NormWear(
        img_size=(387, 65),
        patch_size=(9, 5),
        in_chans=3,
        target_len=388,
        nvar=4,
        embed_dim=768,
        decoder_embed_dim=512,
        depth=12,
        num_heads=12,
        decoder_depth=2,
        mlp_ratio=4.0,
        fuse_freq=2,
        is_pretrain=True,
        mask_t_prob=0.6,
        mask_f_prob=0.5,
        mask_prob=0.8,
        mask_scheme='random',
        use_cwt=True
    )
    print("Model initialized.")
    
    if torch.cuda.is_available():
        model.cuda()
        print("Model moved to GPU.")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

if __name__ == "__main__":
    debug()
