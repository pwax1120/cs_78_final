import os
import torch
import torch.nn as nn
from safetensors.torch import load_file

from resnet_wider import resnet50x1

# Saved google drive ID for the cleaned checkpoint
PRETRAINED_DRIVE_ID = "https://drive.google.com/file/d/1HWMIWLcZlQ_zERiPPtUB2Pd6JRpoCPfD/view?usp=drive_link"

# Where to save the file if it doesn't exist locally (will be downloaded from Google Drive)
CHECKPOINT_PATH = "checkpoints/simclr_v1_r50_1x_pytorch.safetensors"

EXPECTED_SIZE_MB = 102  # Expected file size in MB (for sanity check after download)

# Internal helper function to download the checkpoint from Google Drive using gdown
def _download_if_needed():

    # Do nothing if the file already exists locally
    if os.path.exists(CHECKPOINT_PATH):
        return
    
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    try:
        import gdown
    except ImportError:
        raise ImportError("gdown is required to download the checkpoint. Please install it via 'pip install gdown.'"
        "Or manually place the file from Google Drive at https://drive.google.com/drive/folders/1JJlZhH3GBWErN__-TTWysngcUYXeHzgY?usp=drive_link.")
    
    print(f"Downloading pretrained model to --> {CHECKPOINT_PATH} <--")
    gdown.download(PRETRAINED_DRIVE_ID, CHECKPOINT_PATH, quiet=False)

    # SANITY CHECK: verify file size after download
    actual_size_mb = os.path.getsize(CHECKPOINT_PATH) / 1e6
    if abs(actual_size_mb - EXPECTED_SIZE_MB) > 5:
        print(f"Warning: downloaded file size {actual_size_mb:.1f} MB differs significantly from expected {EXPECTED_SIZE_MB} MB. Please verify the download.")


# Public function to load the pretrained ResNet-50x1 checkpoint into a PyTorch model
def load_pretrained_resnet50x1(num_classes=None, freeze_encoder=False):

    """ 
    Args: if num_classes is provided, the final fully connected layer will be replaced to match that number of classes (useful for downstream classification tasks).
    If freeze_encoder is True, all layers except the final fully connected layer will have requires_grad=False (useful for linear evaluation or fine-tuning).

    Returns: a PyTorch nn.Module with the pretrained weights loaded. The model architecture is defined in resnet_wider.py (ResNet-50x1).
            
    """

    _download_if_needed()

    state = load_file(CHECKPOINT_PATH)
    model = resnet50x1()
    model.load_state_dict(state, strict=True)

    if num_classes is not None:
        model.fc = nn.Linear(2048, num_classes)  # Replace final layer to match num_classes
    
    if freeze_encoder:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc")  # Only the final layer is trainable

    return model


''' This is tester code for the file, it attempts to load the model and then checks that it has the correct formatting for the SimCLR format '''
if __name__ == "__main__":
    print("Loading model with original ImageNet head...")
    m = load_pretrained_resnet50x1()
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, 224, 224))
    print(f"  output shape: {out.shape}")
    assert out.shape == (2, 1000)

    print(f"Loading model for 10-class fine-tuning, encoder frozen...")
    m = load_pretrained_resnet50x1(num_classes=10, freeze_encoder=False)
    print(f"  fc layer: {m.fc}")
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total = sum(p.numel() for p in m.parameters())
    print(f"  trainable params: {trainable:,} / {total:,}")

    print("\nAll checks passed.")
