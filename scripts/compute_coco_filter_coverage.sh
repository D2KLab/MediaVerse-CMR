#!/usr/bin/bash

source ./venv_mediaverse/bin/activate

python evaluate/compute_coco_filter_coverage.py \
    --coco_retrieval $COCO_PATH/annotations/retrieval/coco_i-retrieval_5k_karpathy_objects.json \
    --coco_captions $COCO_PATH/annotations/retrieval/all_captions.json \
    --concepts_table cache/concepts_table.json

deactivate