#!/usr/bin/bash

# source ./venv_mediaverse/bin/activate
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mediaverse

python evaluate/compute_runtime_coco.py \
    --coco_visual cache/coco_visual_cache.pth \
    --coco_text cache/coco_text_cache.pth

conda deactivate