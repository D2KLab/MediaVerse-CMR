#!/usr/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mediaverse

python table_stats.py \
    --table_file concepts_table.json \
    --top_k 100

conda deactivate