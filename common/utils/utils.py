import os
from typing import Dict, List, Tuple, TypeVar

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

COCODataset = TypeVar('COCODataset')
CLIP        = TypeVar('CLIP')
Tensor      = TypeVar('torch.Tensor')
T           = TypeVar('T') #generic



def compute_recalls_vanilla(query_cache: Dict[int, Tensor], pool_cache: Dict[int, Tensor], dataset: COCODataset, device: torch.device) -> Dict[str, float]:
    """
    it assumes that the ground truth for each pool is located in the first position (index = 0)
    """
    recall  = {'R@1': 0, 'R@5': 0, 'R@10': 0}
    loader  = tqdm(dataset)
    curr_it = 1
    for query, pool in loader:
        pool_feats   = torch.stack([pool_cache[item_id] for item_id in pool]).squeeze(1).to(device)
        query_feats  = query_cache[query].to(device)
        similarity   = F.cosine_similarity(query_feats, pool_feats)
        _update_recalls(similarity.tolist(), recall)
        loader.set_description(str({k: round(v/curr_it, 2) for k, v in recall.items()}))
        curr_it += 1
    for rec_type in recall.keys():
        recall[rec_type] /= len(loader)
    return recall


def compute_recalls_filter(query_cache: Dict[int, Tensor], pool_cache: Dict[int, Tensor], dataset: Dict, device: torch.device) -> Dict[str, float]:
    """
    it assumes that the ground truth for each pool is located in the first position (index = 0)
    """
    recall  = {'R@1': 0, 'R@5': 0, 'R@10': 0}
    loader  = tqdm(dataset)
    curr_it = 1
    for row in loader:
        query, pool, gt = row['query'], row['pool'], row['gt']
        try:
            #gt is not always located at index 0 in the 5k IR split. Fix that
            gt_idx = pool.index(gt)
            if gt_idx != 0:
                pool[0], pool[gt_idx] = pool[gt_idx], pool[0] 
        except:
            # ground truth was filtered out
            curr_it += 1
            continue
        pool_feats   = torch.stack([pool_cache[item_id] for item_id in pool]).squeeze(1).to(device)
        query_feats  = query_cache[query].to(device)
        similarity   = F.cosine_similarity(query_feats, pool_feats)
        _update_recalls(similarity.tolist(), recall)
        loader.set_description(str({k: round(v/curr_it, 2) for k, v in recall.items()}))
        curr_it += 1
    for rec_type in recall.keys():
        recall[rec_type] /= len(loader)
    return recall

def compute_recalls_faiss(query_cache: Dict[int, Tensor], pool_cache: Dict[int, Tensor], dataset: COCODataset) -> Dict[str, float]:
    """
    it assumes that the ground truth for each pool is located in the first position (index = 0)
    """
    recall  = {'R@1': 0, 'R@5': 0, 'R@10': 0}
    index   = faiss.index_factory(512, "Flat", faiss.METRIC_INNER_PRODUCT)
    loader  = tqdm(dataset)
    curr_it = 1
    for query, pool in loader:
        index.reset()
        pool_feats   = torch.stack([pool_cache[item_id].squeeze(0) for item_id in pool]).numpy().astype(np.float32)
        query_feats  = query_cache[query].numpy().astype(np.float32)
        faiss.normalize_L2(pool_feats)
        faiss.normalize_L2(query_feats)
        index.add(pool_feats)
        _, res       = index.search(query_feats, 5000)
        try:
            res        = res.squeeze(0).tolist()
            pos        = res.index(0) #gt is always the first element in the pool
            recall['R@1']  += 1 if pos < 1  else 0
            recall['R@5']  += 1 if pos < 5  else 0
            recall['R@10'] += 1 if pos < 10 else 0
        except:
            pass
        loader.set_description(str({k: round(v/curr_it, 2) for k, v in recall.items()}))
        curr_it += 1
    for rec_type in recall.keys():
        recall[rec_type] /= len(dataset)
    return recall


def _update_recalls(scores: List[float], recall: Dict[str, int]) -> None:
    ref = scores[0]
    pos = 1
    for score in scores[1:]:
        pos += 1 if score > ref else 0
        if pos > 10:
            break
    recall['R@1']  += 1 if pos <= 1  else 0
    recall['R@5']  += 1 if pos <= 5  else 0
    recall['R@10'] += 1 if pos <= 10 else 0


"""
def old_compute_cosine_similarity(query: Tensor, pool: Tensor):
    pool   /= pool.norm(dim=-1, keepdim=True)
    query  /= query.norm(dim=-1, keepdim=True)
    similarity   = (pool @ query.T).view(-1) #(100.0 * v_feats @ t_feats.T).softmax(dim=0)
    return similarity
"""


def compute_cosine_similarity(query: Tensor, pool: Tensor):
    return F.cosine_similarity(query, pool)
    #return torch.mm(query, pool.T) / (query.norm(dim=-1, keepdim=True) * pool.norm(dim=-1, keepdim=True))


def top_k(values: 'np.ndarray', k: int, descending: bool) -> Tuple['np.ndarray', 'np.ndarray']:
    """
    O(N+KlogK)
    """
    if descending:
        values = -values

    if k >= values.shape[1]:
        idx = values.argsort(axis=1)[:, :k]
        values = np.take_along_axis(values, idx, axis=1)
    else:
        idx_ps = values.argpartition(kth=k, axis=1)[:, :k]
        values = np.take_along_axis(values, idx_ps, axis=1)
        idx_fs = values.argsort(axis=1)
        idx = np.take_along_axis(idx_ps, idx_fs, axis=1)
        values = np.take_along_axis(values, idx_fs, axis=1)

    if descending:
        values = -values

    return values, idx


def top_k_unsorted(values: 'np.ndarray', k: int, descending: bool) -> np.ndarray:
    """
    O(N)
    """
    if descending:
        values = -values
    if k >= values.shape[1]:
        return np.tile(np.arange(values.shape[1]), (values.shape[0], 1))
    else:
        return torch.topk(values, k, dim=-1).indices.numpy()


def create_cache(datasets: COCODataset, content: str, model: CLIP) -> Dict[int, Tensor]:
    cache = {}
    assert content in ['image', 'text']
    if content == 'image':
        print('caching images features')
        img_folder = datasets.imgs
        for img_id in tqdm(datasets.data['images']):
            if img_id not in cache:
                cache[img_id] = model.clipVisualEncode(os.path.join(img_folder, str(img_id)+'.jpg')).to('cpu')
    else:
        print('caching captions features')
        for capt_id, caption in tqdm(datasets.data['captions'].items()):
            if capt_id not in cache:
                cache[int(capt_id)] = model.clipTextEncode(caption).to('cpu')
    return cache


def cache_random_tensor(nvec: int, dim: int, fpath: str):
    x        = torch.rand((nvec, dim))
    with open(fpath, "wb") as fp:
        torch.save(x, fp)
