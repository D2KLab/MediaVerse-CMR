#!/usr/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mediaverse

python prepare/create_random_cache.py \
    --nb 1000000 \
    --nq 100000 \
    --d 512 \
    --out_dir cache

conda deactivate