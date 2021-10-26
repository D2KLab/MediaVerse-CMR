#!/usr/bin/bash

source ~/mediaverse/MediaVerse-CMR/venv_mediaverse/bin/activate

python prepare/create_random_cache.py \
    --nb 1000000 \
    --nq 100000 \
    --d 512 \
    --out_dir cache

deactivate