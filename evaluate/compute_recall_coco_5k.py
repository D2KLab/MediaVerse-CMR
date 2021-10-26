import argparse
import os
import pdb
from typing import Dict, List, TypeVar

import torch
from tqdm import tqdm


from common import COCOCaptions5k
from common.utils import compute_recalls


Tensor      = TypeVar('torch.Tensor')
COCODataset = TypeVar('COCODataset')



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
        '--image_retrieval_annotations',
        type=str,
        help='Path to file containing image retrieval annotations wrt to specified data root'
    )
    parser.add_argument(
        '--text_retrieval_annotations',
        type=str,
        help='Path to file containing text retrieval annotations wrt to specified data root'
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

    dataset = { 'IR': COCOCaptions5k(args.data_root,
                                     args.image_retrieval_annotations,
                                     args.images_dir),
                'TR': COCOCaptions5k(args.data_root,
                                     args.text_retrieval_annotations,
                                     args.images_dir)
            }

    t_cache = torch.load(args.text_cache)
    v_cache = torch.load(args.images_cache)
    device  = torch.device('cuda:0' if args.cuda else 'cpu')

    print('~~~ Image Retrieval (5k) ~~~')
    recall = compute_recalls(query_cache=t_cache, pool_cache=v_cache, dataset=dataset['IR'], device=device)
    print('[IR]: R@1: {}, R@5: {}, R@10: {}'.format(recall['R@1'], recall['R@5'], recall['R@10']))

    print('~~~ Text Retrieval (5k) ~~~')
    recall = compute_recalls(query_cache=v_cache, pool_cache=t_cache, dataset=dataset['TR'], device=device)
    print('[TR]: R@1: {}, R@5: {}, R@10: {}'.format(recall['R@1'], recall['R@5'], recall['R@10']))