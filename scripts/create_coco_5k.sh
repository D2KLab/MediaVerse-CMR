#!/usr/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mediaverse

python prepare/create_coco_5k.py \
    --data_root $COCO_PATH \
    --annotations_file annotations/karpathy_coco_split.json

conda deactivate