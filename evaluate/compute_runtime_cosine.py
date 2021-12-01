"""
@author: Matteo A. Senese

This scripts computes the run time of retrieval approach based on cosine similarity for generic query-pool dataset.
It is used to get run times on very large randomly generated datasets to have an idea of the scalability of the tool.
(take a look at `prepare/create_random_cache.py`)
"""


import torch
import argparse
from datetime import datetime
from common.utils import compute_cosine_similarity
from tqdm import tqdm
from typing import TypeVar

Tensor = TypeVar('torch.Tensor')
time   = TypeVar('datetime.datetime') 

BATCH_SIZE = 1000


#TODO include also top-k

def get_batching_runtime(pool: Tensor, queries: Tensor, batch_size: int, device: torch.device) -> time:
    pool      = pool.to(device)
    nq        = queries.shape[0]
    start     = datetime.now()
    for start_idx in tqdm(range(0, nq-batch_size, batch_size)):
        end_idx    = start_idx + batch_size
        q          = queries[start_idx:end_idx, :].to(device)
        compute_cosine_similarity(q, pool)
    end       = datetime.now()
    return end-start



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pool',
        type=str,
        help='Path to pool file'
    )
    parser.add_argument(
        '--queries',
        type=str,
        help='Path to queries file'
    )
    parser.add_argument(
        '--cuda',
        action='store_true',
        default=False
    )
    args = parser.parse_args()


    with open(args.pool, 'rb') as fp:
        pool  = torch.load(fp)
    with open(args.queries, 'rb') as fp:
        queries  = torch.load(fp)
    device = torch.device('cuda:0' if args.cuda else 'cpu')

    print(args)
    print('Pool size {}\nQuery size {}'.format(pool.shape, queries.shape))

    t = get_batching_runtime(pool, queries, BATCH_SIZE, device)
    
    print('Runtime: {}'.format(t))