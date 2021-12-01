"""
@author: Matteo A. Senese

This script produces CLIP embeddings for images and captions and cache them on disk
"""



import argparse

import os
import torch

from common import CLIP, COCOCaptions5k
from common.utils import create_cache


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
        help='Path to MSCOCO images directory wrt specified data root'
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
        '--out_dir',
        type=str,
        help='Directory where to store the cached features'
    )
    parser.add_argument(
        '--cuda',
        action='store_true',
        default=False
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device  = 'cuda:0' if args.cuda else 'cpu'
    model   = CLIP(device=device)
    dataset = { 'IR': COCOCaptions5k(args.data_root,
                                     args.image_retrieval_annotations,
                                     args.images_dir),
                'TR': COCOCaptions5k(args.data_root,
                                     args.text_retrieval_annotations,
                                     args.images_dir)
            }
    v_cache = create_cache(dataset['IR'], model = model, content = 'image')
    t_cache = create_cache(dataset['TR'], model = model, content = 'text')
    print('Writing on disk ...')
    with open(os.path.join(args.out_dir, 'coco_visual_cache.pth'), 'wb') as fp:
        torch.save(v_cache, fp)
    with open(os.path.join(args.out_dir, 'coco_text_cache.pth'), 'wb') as fp:
        torch.save(t_cache, fp)