#!/usr/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mediaverse

python evaluate/compute_runtime_filter.py \
    --coco_retrieval $COCO_PATH/annotations/retrieval/coco_i-retrieval_5k_karpathy_objects.json \
    --coco_captions $COCO_PATH/annotations/retrieval/all_captions.json \
    --concepts_table cache/concepts_table.json \
    --txt_cache cache/coco_text_cache.pth \
    --visual_cache cache/coco_visual_cache.pth

conda deactivate