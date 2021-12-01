import torch
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
import time
from common.utils import compute_cosine_similarity, top_k, top_k_unsorted
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import faiss
from typing import TypeVar

Tensor = TypeVar('torch.Tensor')
# time   = TypeVar('datetime.datetime') 


def compute_runtime_faiss(pool: Tensor, queries: Tensor, K: int, device: torch.device) -> time:
    assert(pool.shape[-1] == queries.shape[-1]), "Error! Pool and queries do not have the same dimensions!"
    # Create FAISS index and load data.
    dim = pool.shape[-1]
    pool = pool.squeeze(1)
    pool = pool.numpy().astype(np.float32)
    queries = queries.numpy().astype(np.float32)
    index = faiss.IndexFlatL2(dim)   # build the index.
    index.add(pool)                  # add vectors to the index.

    start = time.time()
    for q in queries:
        D, I = index.search(q, K)     # Find top K nearest neighbourhs of q.  I= indexes of top K, D= distances of the top K
        I = I.squeeze()
    end = time.time()
    return end-start

def compute_runtime_cosine(
    pool: Tensor,
    queries: Tensor,
    K: int,
    device: torch.device) -> time:
    nq        = queries.shape[0]
    start     = time.time()
    for query in tqdm(queries, desc='cosine'):
        sim = compute_cosine_similarity(query, pool)
        top_k(sim.unsqueeze(0).numpy(), k=K, descending=True)
    end       = time.time()
    return end-start

def compute_runtime_cosine_1(
    pool: Tensor,
    queries: Tensor,
    K: int,
    device: torch.device) -> time:
    queries, pool = queries.double(), pool.double()
    nq        = queries.shape[0]
    start     = time.time()
    for query in queries:
        sim = cosine_similarity(query, pool).unsqueeze(0)
        top_k_unsorted(sim, k=K, descending=True)
    end       = time.time()
    return end-start


def scan_parameters(pool, queries):
    pool = pool.repeat(10, 1)  # Increase numper of pools to 50K
    queries = queries[: 1000]  # Decrease number of queries to 1K

    K_size = [10, 100, 200, 400, 600, 800, 1000]
    pool_size = [2000, 5000, 10000, 20000, 30000, 40000, 50000]

    res = []
    # Scan runtime vs pool size, fixing topK.
    topK = 800
    for size in pool_size:
        t_cos = compute_runtime_cosine(pool[: size], queries, topK, device)
        t_faiss = compute_runtime_faiss(pool[: size], queries, topK, device)
        print('Runtime: size: {} K:{}   --   cosine: {}     faiss:{}'.format(size, topK, t_cos, t_faiss))
        res.append((size, topK, t_cos, t_faiss))
    # Scan runtime vs top K size, fixing pool size.
    size = 50000
    for topK in K_size:
        t_cos = compute_runtime_cosine(pool[: size], queries, topK, device)
        t_faiss = compute_runtime_faiss(pool[: size], queries, topK, device)
        print('Runtime: size: {} K:{}   --   cosine: {}     faiss:{}'.format(size, topK, t_cos, t_faiss))
        res.append((size, topK, t_cos, t_faiss))
    
    return res



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--coco_visual',
        type=str,
        help='Path to images tensor pool'
    )
    parser.add_argument(
        '--coco_text',
        type=str,
        help='Path to text tensor pool'
    )
    parser.add_argument(
        '--cuda',
        action='store_true',
        default=False
    )
    args = parser.parse_args()


    with open(args.coco_visual, 'rb') as fp:
        pool  = torch.load(fp)
    with open(args.coco_text, 'rb') as fp:
        queries  = torch.load(fp)
    device = torch.device('cuda:0' if args.cuda else 'cpu')

    print(args)

    pool      = torch.stack([v for v in pool.values()]).to(device)
    queries   = torch.stack([v for v in queries.values()]).to(device)

    res = scan_parameters(pool, queries)
    res = pd.DataFrame(res, columns=['db_size', 'top K', 'cosine', 'faiss'])
    res.to_csv('evaluate/results/exec_runtimes_scan.csv')
