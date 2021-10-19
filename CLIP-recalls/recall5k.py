import argparse
import os
import pdb
from typing import Dict, List, TypeVar

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import IRETR_5k as IRETR
from config import TRETR_5k as TRETR
from dataset import COCOCaptions5k
from model import CLIP
from utils import create_cache, update_recalls


Tensor      = TypeVar('torch.Tensor')
COCODataset = TypeVar('COCODataset')


DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLIP_VIS  = 'ViT-B/32'
print('# DEVICE={}'.format(DEVICE))


def compute_recalls(query_cache: Dict[int, Tensor], pool_cache: Dict[int, Tensor], dataset: COCODataset) -> Dict[str, float]:
    recall = {'R@1': 0, 'R@5': 0, 'R@10': 0}
    loader = tqdm(dataset)
    for query, pool in tqdm(loader):
        pool_feats   = torch.stack([pool_cache[item_id] for item_id in pool]).to(DEVICE)
        query_feats  = query_cache[query].to(DEVICE)
        pool_feats  /= pool_feats.norm(dim=-1, keepdim=True)
        query_feats /= query_feats.norm(dim=-1, keepdim=True)
        similarity   = (pool_feats @ query_feats.T).view(-1) #(100.0 * v_feats @ t_feats.T).softmax(dim=0)
        update_recalls(similarity, recall)
    for rec_type in recall.keys():
        recall[rec_type] /= len(loader)
    return recall


model    = CLIP(vis_model=CLIP_VIS, device=DEVICE)
dataset = {'IR': COCOCaptions5k(IRETR['DATA_ROOT'], IRETR['ANN_FILES'], IRETR['IMG_DIR'], retrieval='image'),
           'TR': COCOCaptions5k(TRETR['DATA_ROOT'], TRETR['ANN_FILES'], TRETR['IMG_DIR'], retrieval='text')}

v_cache  = create_cache(dataset['IR'], model = model, content = 'image')
t_cache  = create_cache(dataset['TR'], model = model, content = 'text')

print('~~~ Image Retrieval (5k) ~~~')
recall = compute_recalls(query_cache=t_cache, pool_cache=v_cache, dataset=dataset['IR'])
print('[IR]: R@1: {}, R@5: {}, R@10: {}'.format(recall['R@1'], recall['R@5'], recall['R@10']))

print('~~~ Text Retrieval (5k) ~~~')
recall = compute_recalls(query_cache=v_cache, pool_cache=t_cache, dataset=dataset['TR'])
print('[TR]: R@1: {}, R@5: {}, R@10: {}'.format(recall['R@1'], recall['R@5'], recall['R@10']))