#!/usr/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mediaverse

# coco 1k with 5 folds
python prepare/create_coco_1k.py \
    --data_root $COCO_PATH \
    --annotations_file annotations/karpathy_coco_split.json \
    --nfolds 5

# coco 5k
python prepare/create_coco_5k.py \
    --data_root $COCO_PATH \
    --annotations_file annotations/karpathy_coco_split.json

conda deactivate