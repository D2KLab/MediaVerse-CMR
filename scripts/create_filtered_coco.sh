#!/usr/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mediaverse

# coco 1k with 5 folds
python prepare/create_filtered_coco_1k.py \
    --concepts_table cache/concepts_table.json \
    --coco_retrieval_root $COCO_PATH/annotations/retrieval \
    --img_to_objects_mapping cache/image_to_tags.json \
    --out_dir cache/filtered

# coco 5k
python prepare/create_filtered_coco_5k.py \
    --concepts_table cache/concepts_table.json \
    --coco_retrieval_root $COCO_PATH/annotations/retrieval \
    --img_to_objects_mapping cache/image_to_tags.json \
    --out_dir cache/filtered

conda deactivate