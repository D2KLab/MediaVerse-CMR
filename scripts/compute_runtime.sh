#!/usr/bin/bash

source ~/mediaverse/MediaVerse-CMR/venv_mediaverse/bin/activate

python evaluate/compute_runtime_cosine.py \
    --pool cache/random_pool.pth \
    --queries cache/random_queries.pth \
    --cuda

deactivate