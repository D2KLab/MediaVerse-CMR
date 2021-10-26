#!/usr/bin/bash

source ~/mediaverse/MediaVerse-CMR/venv_mediaverse/bin/activate

python prepare/create_coco_1k.py \
    --data_root $COCO_PATH \
    --annotations_file annotations/karpathy_coco_split.json \
    --nfolds 5

deactivate