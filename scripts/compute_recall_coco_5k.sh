#!/usr/bin/bash

source ~/mediaverse/MediaVerse-CMR/venv_mediaverse/bin/activate

python evaluate/compute_recall_coco_5k.py \
    --data_root $COCO_PATH \
    --images_dir images \
    --image_retrieval_annotations annotations/retrieval/coco_i-retrieval_5k_karpathy.json \
    --text_retrieval_annotations annotations/retrieval/coco_t-retrieval_5k_karpathy.json \
    --images_cache cache/coco_visual_cache.pth \
    --text_cache cache/coco_text_cache.pth \
    --cuda

deactivate