#!/usr/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mediaverse

python prepare/create_image_to_objects_mapping.py \
    --coco_annotations_dir $COCO_PATH/annotations

conda deactivate