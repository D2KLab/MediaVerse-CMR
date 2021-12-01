#!/usr/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mediaverse


python prepare/create_filtered_coco_1k.py \
    --concepts_table cache/concepts_table.json \
    --coco_retrieval_root $COCO_PATH/annotations/retrieval \
    --img_to_objects_mapping cache/image_to_tags.json \
    --out_dir cache/filtered

python prepare/create_filtered_coco_5k.py \
    --concepts_table cache/concepts_table.json \
    --coco_retrieval_root $COCO_PATH/annotations/retrieval \
    --img_to_objects_mapping cache/image_to_tags.json \
    --out_dir cache/filtered

conda deactivate