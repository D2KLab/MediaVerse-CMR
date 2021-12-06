"""
@author: Matteo A. Senese

This script computes the recall on MSCOCO-5K split using the cosine similarity on CLIP embeddings.

The recalls are computed for: vanilla cosine, filter+cosine, faiss.
"""


import argparse
import collections
import json
import os
from typing import Dict, List, TypeVar

import torch
from common import COCOCaptions5k
from common.utils import compute_recalls_filter

Tensor      = TypeVar('torch.Tensor')
COCODataset = TypeVar('COCODataset')


N_FOLDS             = 5
COCO_1K_IR_TEMPLATE = 'filtered_coco_i-retrieval_1k_karpathy_fold{}.json'
COCO_1K_TR_TEMPLATE = 'filtered_coco_t-retrieval_1k_karpathy_fold{}.json'
COCO_5K_IR          = 'filtered_coco_i-retrieval_5k_karpathy.json'
COCO_5K_TR          = 'filtered_coco_t-retrieval_5k_karpathy.json'



def compute_folds_recalls(query_cache: Dict[int, Tensor],
                          pool_cache: Dict[int, Tensor],
                          datasets: Dict,
                          device: torch.device) -> Dict[str, float]:
    recalls = []
    for fold in datasets:
        recalls.append(compute_recalls_filter(query_cache, pool_cache, dataset=fold, device=device))
    recall_1  = sum(curr_recall['R@1'] for curr_recall in recalls)  / len(recalls)
    recall_5  = sum(curr_recall['R@5'] for curr_recall in recalls)  / len(recalls)
    recall_10 = sum(curr_recall['R@10'] for curr_recall in recalls) / len(recalls)
    return {'R@1': recall_1, 'R@5': recall_5, 'R@10': recall_10}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--annotations_dir',
        type=str,
        help='Directory containing the filtered annotations'
    )
    parser.add_argument(
        '--images_cache',
        type=str,
        help='Path to file containing images features'
    )
    parser.add_argument(
        '--text_cache',
        type=str,
        help='Path to file containing text features'
    )
    parser.add_argument(
        '--cuda',
        action='store_true',
        default=False
    )

    args = parser.parse_args()

    # loading dataset folds for coco 1k
    dataset_1k = collections.defaultdict(list)
    for fold_n in range(1, N_FOLDS+1):
        with open(os.path.join(args.annotations_dir, COCO_1K_IR_TEMPLATE.format(fold_n))) as fp:
            coco = json.load(fp)
            dataset_1k['IR'].append(coco['annotations'])
        with open(os.path.join(args.annotations_dir, COCO_1K_TR_TEMPLATE.format(fold_n))) as fp:
            coco = json.load(fp)
            dataset_1k['TR'].append(coco['annotations'])
    # loading coco 5k
    dataset_5k = {}
    with open(os.path.join(args.annotations_dir, COCO_5K_IR)) as fp:
        dataset_5k['IR'] = json.load(fp)['annotations']
    with open(os.path.join(args.annotations_dir, COCO_5K_TR)) as fp:
        dataset_5k['TR'] = json.load(fp)['annotations']

    t_cache = torch.load(args.text_cache)
    v_cache = torch.load(args.images_cache)
    device  = torch.device('cuda:0' if args.cuda else 'cpu')

    print('~~~ [FILTER] Image Retrieval (1k) ~~~')
    recall = compute_folds_recalls(query_cache=t_cache, pool_cache=v_cache, datasets=dataset_1k['IR'], device=device)
    print('[IR]: R@1: {}, R@5: {}, R@10: {}'.format(recall['R@1'], recall['R@5'], recall['R@10']))

    print('~~~ [FILTER] Text Retrieval (1k) ~~~')
    recall = compute_folds_recalls(query_cache=v_cache, pool_cache=t_cache, datasets=dataset_1k['TR'], device=device)
    print('[TR]: R@1: {}, R@5: {}, R@10: {}'.format(recall['R@1'], recall['R@5'], recall['R@10']))

    print('~~~ [FILTER] Image Retrieval (5k) ~~~')
    recall = compute_recalls_filter(query_cache=t_cache, pool_cache=v_cache, dataset=dataset_5k['IR'], device=device)
    print('[TR]: R@1: {}, R@5: {}, R@10: {}'.format(recall['R@1'], recall['R@5'], recall['R@10']))

    print('~~~ [FILTER] Text Retrieval (5k) ~~~')
    recall = compute_recalls_filter(query_cache=v_cache, pool_cache=t_cache, dataset=dataset_5k['TR'], device=device)
    print('[TR]: R@1: {}, R@5: {}, R@10: {}'.format(recall['R@1'], recall['R@5'], recall['R@10']))
