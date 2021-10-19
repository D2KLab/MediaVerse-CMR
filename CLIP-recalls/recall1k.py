import argparse
import os
import pdb
from typing import Dict, List, TypeVar

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import IRETR_1K as IRETR
from config import TRETR_1K as TRETR
from dataset import COCOCaptions1k
from model import CLIP
from utils import create_cache, update_recalls


Tensor      = TypeVar('torch.Tensor')
COCODataset = TypeVar('COCODataset')


DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLIP_VIS  = 'ViT-B/32'
print('# DEVICE={}'.format(DEVICE))


def compute_recalls(query_cache: Dict[int, Tensor], pool_cache: Dict[int, Tensor], datasets: COCODataset) -> Dict[str, float]:
    recall = [{'R@1': 0, 'R@5': 0, 'R@10': 0} for _ in range(5)]
    for fold_idx, fold in enumerate(datasets):
        loader = DataLoader(fold)
        for query, pool in tqdm(loader):
            pool_feats   = torch.stack([pool_cache[item_id.item()] for item_id in pool]).to(DEVICE)
            query_feats  = query_cache[query.item()].to(DEVICE)
            pool_feats  /= pool_feats.norm(dim=-1, keepdim=True)
            query_feats /= query_feats.norm(dim=-1, keepdim=True)
            similarity   = (pool_feats @ query_feats.T).view(-1) #(100.0 * v_feats @ t_feats.T).softmax(dim=0)
            update_recalls(similarity, recall[fold_idx])
        for rec_type in recall[fold_idx].keys():
            recall[fold_idx][rec_type] /= len(loader)
    recall_1  = sum(curr_recall['R@1'] for curr_recall in recall)  / len(recall)
    recall_5  = sum(curr_recall['R@5'] for curr_recall in recall)  / len(recall)
    recall_10 = sum(curr_recall['R@10'] for curr_recall in recall) / len(recall)
    return {'R@1': recall_1, 'R@5': recall_5, 'R@10': recall_10}


model    = CLIP(vis_model=CLIP_VIS, device=DEVICE)
datasets = {'IR': [], 'TR': []}
for fold in range(1, 6):
    datasets['IR'].append(COCOCaptions1k(IRETR['DATA_ROOT'], IRETR['ANN_FILES'].format(fold), IRETR['IMG_DIR'], retrieval='image'))
    datasets['TR'].append(COCOCaptions1k(TRETR['DATA_ROOT'], TRETR['ANN_FILES'].format(fold), TRETR['IMG_DIR'], retrieval='text'))

v_cache  = create_cache(datasets['IR'][0], model = model, content = 'image')
t_cache  = create_cache(datasets['TR'][0], model = model, content = 'text')

print('~~~ Image Retrieval (1k) ~~~')
recall = compute_recalls(query_cache=t_cache, pool_cache=v_cache, datasets=datasets['IR'])
print('[IR]: R@1: {}, R@5: {}, R@10: {}'.format(recall['R@1'], recall['R@5'], recall['R@10']))

print('~~~ Text Retrieval (1k) ~~~')
recall = compute_recalls(query_cache=v_cache, pool_cache=t_cache, datasets=datasets['TR'])
print('[TR]: R@1: {}, R@5: {}, R@10: {}'.format(recall['R@1'], recall['R@5'], recall['R@10']))