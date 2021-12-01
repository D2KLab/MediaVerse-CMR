"""
@author: Matteo A. Senese

This script computes the recall on MSCOCO-1K split using the cosine similarity on CLIP embeddings.

The recalls are computed for: vanilla cosine, filter+cosine, faiss.
"""


import argparse
import os
from typing import Dict, TypeVar

import torch
from common import COCOCaptions1k
from common.utils import compute_recalls_vanilla, compute_recalls_faiss

Tensor      = TypeVar('torch.Tensor')
COCODataset = TypeVar('COCODataset')



def compute_folds_recalls(query_cache: Dict[int, Tensor],
                          pool_cache: Dict[int, Tensor],
                          datasets: COCODataset,
                          device: torch.device) -> Dict[str, float]:
    recalls = []
    for fold in datasets:
        recalls.append(compute_recalls_vanilla(query_cache, pool_cache, dataset=fold, device=device))
    recall_1  = sum(curr_recall['R@1'] for curr_recall in recalls)  / len(recalls)
    recall_5  = sum(curr_recall['R@5'] for curr_recall in recalls)  / len(recalls)
    recall_10 = sum(curr_recall['R@10'] for curr_recall in recalls) / len(recalls)
    return {'R@1': recall_1, 'R@5': recall_5, 'R@10': recall_10}


def compute_folds_recalls_faiss(query_cache: Dict[int, Tensor],
                                pool_cache: Dict[int, Tensor],
                                datasets: COCODataset) -> Dict[str, float]:
    recalls = []
    for fold in datasets:
        recalls.append(compute_recalls_faiss(query_cache, pool_cache, dataset=fold))
    recall_1  = sum(curr_recall['R@1'] for curr_recall in recalls)  / len(recalls)
    recall_5  = sum(curr_recall['R@5'] for curr_recall in recalls)  / len(recalls)
    recall_10 = sum(curr_recall['R@10'] for curr_recall in recalls) / len(recalls)
    return {'R@1': recall_1, 'R@5': recall_5, 'R@10': recall_10}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root',
        type=str,
        help='Path to MSCOCO root directory'
    )
    parser.add_argument(
        '--images_dir',
        type=str,
        help='Path to MSCOCO images directory wrt to specified data root'
    )
    parser.add_argument(
        '--image_retrieval_dir',
        type=str,
        help='Path to directory containing image retrieval annotations files wrt to specified data root'
    )
    parser.add_argument(
        '--text_retrieval_dir',
        type=str,
        help='Path to file containing text retrieval annotations files wrt to specified data root'
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


    img_retrieval_format = os.path.join(args.image_retrieval_dir, 'coco_i-retrieval_1k_karpathy_fold{}.json')
    txt_retrieval_format = os.path.join(args.text_retrieval_dir, 'coco_t-retrieval_1k_karpathy_fold{}.json')
    print('Loading different dataset folds ...')
    dataset = { 'IR': [COCOCaptions1k(args.data_root,
                                     img_retrieval_format.format(fold_idx),
                                     args.images_dir) for fold_idx in range(1, 6)],
                'TR': [COCOCaptions1k(args.data_root,
                                     txt_retrieval_format.format(fold_idx),
                                     args.images_dir) for fold_idx in range(1, 6)]
            }

    t_cache = torch.load(args.text_cache)
    v_cache = torch.load(args.images_cache)
    device  = torch.device('cuda:0' if args.cuda else 'cpu')

    print('~~~ [VANILLA] Image Retrieval (1k) ~~~')
    recall = compute_folds_recalls(query_cache=t_cache, pool_cache=v_cache, datasets=dataset['IR'], device=device)
    print('[IR]: R@1: {}, R@5: {}, R@10: {}'.format(recall['R@1'], recall['R@5'], recall['R@10']))

    print('~~~ [VANILLA] Text Retrieval (1k) ~~~')
    recall = compute_folds_recalls(query_cache=v_cache, pool_cache=t_cache, datasets=dataset['TR'], device=device)
    print('[TR]: R@1: {}, R@5: {}, R@10: {}'.format(recall['R@1'], recall['R@5'], recall['R@10']))

    print('~~~ [FAISS] Image Retrieval (1k) ~~~')
    recall = compute_folds_recalls_faiss(query_cache=t_cache, pool_cache=v_cache, datasets=dataset['IR'])
    print('[IR]: R@1: {}, R@5: {}, R@10: {}'.format(recall['R@1'], recall['R@5'], recall['R@10']))

    print('~~~ [FAISS] Text Retrieval (1k) ~~~')
    recall = compute_folds_recalls_faiss(query_cache=v_cache, pool_cache=t_cache, datasets=dataset['TR'])
    print('[TR]: R@1: {}, R@5: {}, R@10: {}'.format(recall['R@1'], recall['R@5'], recall['R@10']))
