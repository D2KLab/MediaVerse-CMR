#!/usr/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mediaverse

python evaluate/compute_runtime_cosine.py \
    --pool cache/random_pool.pth \
    --queries cache/random_queries.pth \
    --cuda

conda deactivate