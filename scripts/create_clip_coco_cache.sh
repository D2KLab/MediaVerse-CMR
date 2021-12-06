#!/usr/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mediaverse

python prepare/create_clip_coco_cache.py \
    --data_root $COCO_PATH \
    --images_dir images \
    --image_retrieval_annotations annotations/retrieval/coco_i-retrieval_5k_karpathy.json \
    --text_retrieval_annotations annotations/retrieval/coco_t-retrieval_5k_karpathy.json \
    --out_dir cache \
    --cuda

conda deactivate