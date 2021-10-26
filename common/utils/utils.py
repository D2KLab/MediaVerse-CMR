import os
from typing import Dict, List, TypeVar

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

COCODataset = TypeVar('COCODataset')
CLIP        = TypeVar('CLIP')
Tensor      = TypeVar('torch.Tensor')
T           = TypeVar('T') #generic


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


def compute_recalls(query_cache: Dict[int, Tensor], pool_cache: Dict[int, Tensor], dataset: COCODataset, device: torch.device) -> Dict[str, float]:
    """
    it assumes that the ground truth for each pool is located in the first position (index = 0)
    """
    recall = {'R@1': 0, 'R@5': 0, 'R@10': 0}
    loader = tqdm(dataset)
    for query, pool in tqdm(loader):
        pool_feats   = torch.stack([pool_cache[item_id] for item_id in pool]).to(device)
        query_feats  = query_cache[query].to(device)
        similarity   = compute_cosine_similarity(query_feats, pool_feats)
        _update_recalls(similarity, recall)
    for rec_type in recall.keys():
        recall[rec_type] /= len(loader)
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


def compute_cosine_similarity(query: Tensor, pool: Tensor):
    pool   /= pool.norm(dim=-1, keepdim=True)
    query  /= query.norm(dim=-1, keepdim=True)
    similarity   = (pool @ query.T).view(-1) #(100.0 * v_feats @ t_feats.T).softmax(dim=0)
    return similarity


def cache_random_tensor(nvec: int, dim: int, fpath: str):
    x        = torch.rand((nvec, dim))
    with open(fpath, "wb") as fp:
        torch.save(x, fp)
