import sys, os
sys.path.insert(0, '/home/ug24/FoundationalModel')

import torch
import torch.nn as nn

# Auto-find checkpoint
pth_files = []
for root, dirs, files in os.walk('NormWear/data/results/'):
    for f in files:
        if f.endswith('.pth'):
            pth_files.append(os.path.join(root, f))

if not pth_files:
    print("No .pth files found! Run: find . -name '*.pth'")
    sys.exit(1)

# Use the latest checkpoint
checkpoint_path = sorted(pth_files)[-1]
print(f"Using checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

from NormWear.pretrain.pipeline.misc import NormWear
model = NormWear()
model.load_state_dict(checkpoint['model'], strict=False)
model.eval()

print("\n=== ATTENTION LAYERS ===")
for name, m in model.named_modules():
    if isinstance(m, nn.Linear) and any(k in name for k in
        ['attn', 'query', 'key', 'value', 'qkv', 'proj', 'q_', 'v_']):
        print(f"{name}  |  in={m.in_features}  out={m.out_features}")