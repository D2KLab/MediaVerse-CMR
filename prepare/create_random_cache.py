"""
@author: Matteo A. Senese

This script creates random vectors to create fake pools and queries. It is used to compute FAISS run times on very large pools.
Since the vectors are random generated, the resulting dataset can not be used to compute quality scores (e.g., Recall@K).
"""


import argparse
import torch
import os
from common.utils import cache_random_tensor





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--nb',
        type=int,
        help='Number of vectors in the database'
    )
    parser.add_argument(
        '--nq',
        type=int,
        help='Number of query vectors'
    )
    parser.add_argument(
        '--d',
        type=int,
        help='Size of each vector'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        help='Directory where to store the cached features'
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cache_random_tensor(nvec=args.nb, dim=args.d, fpath=os.path.join(args.out_dir, 'random_pool.pth'))
    cache_random_tensor(nvec=args.nq, dim=args.d, fpath=os.path.join(args.out_dir, 'random_queries.pth'))