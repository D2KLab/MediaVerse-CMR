#!/usr/bin/bash

source ./venv_mediaverse/bin/activate

python evaluate/compute_recall_coco_1k.py \
    --data_root $COCO_PATH \
    --images_dir images \
    --image_retrieval_dir annotations/retrieval \
    --text_retrieval_dir annotations/retrieval \
    --images_cache cache/coco_visual_cache.pth \
    --text_cache cache/coco_text_cache.pth \
    --cuda

deactivate