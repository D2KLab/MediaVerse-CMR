#!/usr/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mediaverse

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=5


# python evaluate/compute_recall_coco_1k.py \
#     --data_root $COCO_PATH \
#     --images_dir images \
#     --image_retrieval_dir annotations/retrieval \
#     --text_retrieval_dir annotations/retrieval \
#     --images_cache cache/coco_visual_cache.pth \
#     --text_cache cache/coco_text_cache.pth \
#     --cuda

python evaluate/compute_recall_filter_1k.py \
    --annotations_dir cache/filtered \
    --images_cache cache/coco_visual_cache.pth \
    --text_cache cache/coco_text_cache.pth \
    --cuda

# python evaluate/compute_recall_coco_5k.py \
#     --data_root $COCO_PATH \
#     --images_dir images \
#     --image_retrieval_annotations annotations/retrieval/coco_i-retrieval_5k_karpathy.json \
#     --text_retrieval_annotations annotations/retrieval/coco_t-retrieval_5k_karpathy.json \
#     --images_cache cache/coco_visual_cache.pth \
#     --text_cache cache/coco_text_cache.pth \
#     --cuda

conda deactivate