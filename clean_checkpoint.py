"""One-time script: convert the raw HuggingFace SimCLR v1 weights
(SauravMaheshkar/simclrv1-imagenet1k-resnet50-1x) into a PyTorch-native
safetensors file that loads with strict=True against resnet_wider.py.

Ran once on 2026-05-24. Output uploaded to project Google Drive folder
at: https://drive.google.com/drive/folders/1JJlZhH3GBWErN__-TTWysngcUYXeHzgY?usp=drive_link

Not part of the runtime pipeline — kept for reproducibility/provenance.
"""

import os
from safetensors.torch import load_file, save_file

SRC = "simclr-converter/model.safetensors"
DST = "checkpoints/simclr_v1_r50_1x_pytorch.safetensors"

os.makedirs("checkpoints", exist_ok=True)

raw = load_file(SRC)
print(f"Loaded {len(raw)} tensors from {SRC}")

# Fix 1: drop the extra ".layers" in key names
remapped = {k.replace(".layers.", "."): v for k, v in raw.items()}

# Fix 2: transpose 4D conv weights from [out, H, W, in] to [out, in, H, W]
fixed = {
    k: (v.permute(0, 3, 1, 2).contiguous() if v.ndim == 4 else v)
    for k, v in remapped.items()
}

save_file(fixed, DST)
print(f"Saved cleaned weights to {DST}")
print(f"File size: {os.path.getsize(DST) / 1e6:.1f} MB")