#!/usr/bin/bash

source ./venv_mediaverse/bin/activate

python prepare/create_concepts_table.py \
    --source_file $COCO_PATH/annotations/instances_val2017.json \
    --out_dir cache

deactivate