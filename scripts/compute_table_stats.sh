#!/usr/bin/bash

source ~/mediaverse/MediaVerse-CMR/venv_mediaverse/bin/activate

python table_stats.py \
    --table_file concepts_table.json \
    --top_k 100

deactivate