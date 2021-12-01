import argparse
import time
from datetime import datetime
from typing import TypeVar

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import cosine_similarity

from common.utils import top_k_unsorted

Tensor = TypeVar('torch.Tensor')


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
        import pdb; pdb.set_trace()
    end = time.time()
    return (end-start) / len(queries)

def compute_runtime_cosine(
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
    return (end - start) / len(queries)

def scan_parameters(pool: Tensor, queries: Tensor) -> pd.DataFrame:
    pool = pool.repeat(200, 1)  # Increase numper of pools to 1M
    queries = queries[: 1000]  # Decrease number of queries to 1K

    K_size = [10, 100, 200, 400, 600, 800, 1000]
    pool_size = [
        2_000, 5_000, 10_000, 20_000, 30_000, 40_000, 50_000, 100_000, 1_000_000
    ]

    res = []
    # Scan runtime vs pool size, fixing topK.
    topK = 800
    for size in pool_size:
        t_cos = compute_runtime_cosine(pool[: size], queries, topK, device)
        t_faiss = compute_runtime_faiss(pool[: size], queries, topK, device)
        print('Runtime: size: {} K:{}   --   cosine: {}     faiss:{}'.format(size, topK, t_cos, t_faiss))
        res.append((size, topK, t_cos, t_faiss))
    # Scan runtime vs top K size, fixing pool size.
    size = 1_000_000
    for topK in K_size:
        t_cos = compute_runtime_cosine(pool[: size], queries, topK, device)
        t_faiss = compute_runtime_faiss(pool[: size], queries, topK, device)
        print('Runtime: size: {} K:{}   --   cosine: {}     faiss:{}'.format(size, topK, t_cos, t_faiss))
        res.append((size, topK, t_cos, t_faiss))

    res = pd.DataFrame(res, columns=['db_size', 'top K', 'cosine', 'faiss'])
    return res

def plot_results(data: pd.DataFrame) -> None:
    ax = plt.gca()
    df = data.iloc[: 9]
    df.plot(kind='line',x='db_size',y='cosine', color='blue', ax=ax)
    df.plot(kind='line',x='db_size',y='faiss', color='red', ax=ax)
    ax.set_title('Top K retrieval time VS pool size\n(K=800)')
    ax.set_xlabel('pool size')
    ax.set_ylabel('time (sec)')
    plt.savefig('evaluate/results/runtime_vs_poolsize.png')
    plt.close()
    ax = plt.gca()
    df = data.iloc[9: ]
    df.plot(kind='line',x='top K',y='cosine', color='blue', ax=ax)
    df.plot(kind='line',x='top K',y='faiss', color='red', ax=ax)
    ax.set_title('Top K retrieval time VS K\n(pool size=1M)')
    ax.set_xlabel('top K nearest neighbours')
    ax.set_ylabel('time (sec)')
    plt.savefig('evaluate/results/runtime_vs_K.png')





if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--coco_visual',
    #     type=str,
    #     default='cache/coco_visual_cache.pth',
    #     help='Path to images tensor pool'
    # )
    # parser.add_argument(
    #     '--coco_text',
    #     type=str,
    #     default='cache/coco_text_cache.pth',
    #     help='Path to text tensor pool'
    # )
    # parser.add_argument(
    #     '--cuda',
    #     action='store_true',
    #     default=False
    # )
    # args = parser.parse_args()


    # with open(args.coco_visual, 'rb') as fp:
    #     pool  = torch.load(fp)
    # with open(args.coco_text, 'rb') as fp:
    #     queries  = torch.load(fp)
    # device = torch.device('cuda:0' if args.cuda else 'cpu')

    # print(args)

    # pool      = torch.stack([v for v in pool.values()]).squeeze(1).to(device)
    # queries   = torch.stack([v for v in queries.values()]).to(device)

    # torch.set_num_threads(1)
    # res = scan_parameters(pool, queries)
    # res.to_csv('evaluate/results/exec_runtimes_scan.csv')
    # # res = pd.read_csv('evaluate/results/exec_runtimes_scan.csv')
    # plot_results(res)

    pool = np.random.rand(5000, 512).astype(np.float32)
    query = np.random.rand(1, 512).astype(np.float32)

    index = faiss.index_factory(512, "Flat", faiss.METRIC_INNER_PRODUCT)
    faiss.normalize_L2(query)
    faiss.normalize_L2(pool)
    index.add(pool)
    sims, res       = index.search(query, 5000)
    
    sims = sims.squeeze(0)
    sims2 = np.matmul(query, pool.T).squeeze(0)
    sims2.sort()
    sims2 = sims2[::-1]

    equals = sims == sims2

    i = 0
    for x, y, e in zip(sims, sims2, equals):
        print(x, y, e)
        i += 1
        if i >= 20:
            break

    print(sum(equals))


    
    
