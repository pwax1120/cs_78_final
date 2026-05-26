#!/bin/bash
set -e

# Clone the converter from GitHub because the original repo doesn't support GPU training
git clone https://github.com/tonylins/simclr-converter.git
cd simclr-converter

# Download the official SimCLR checkpoints
python convert.py --depth 50 --width_multiplier 1.0 \
    --sk_ratio 0 --checkpoint <tf_checkpoint_path> \
    --output ../checkpoints/simclr_v1_r50_1x.pth

cd ..
echo "SimCLR setup completed. Checkpoints are saved in the 'checkpoints' directory under /simclr_v1_r50_1x.pth."