#!/usr/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mediaverse

python prepare/create_concepts_table.py \
    --source_file $COCO_PATH/annotations/instances_val2017.json \
    --out_dir cache

conda deactivate