"""
@author: Matteo A. Senese

This script creates the filtered version of MSCOCO-1K split (both text and image retrieval).

"""


import argparse
import json
import os
from typing import Dict, List

from common import Filter
from tqdm import tqdm

N_FOLDS   = 5


def load_capt_to_txt(args):
    coco_path = os.path.join(args.coco_retrieval_root, 'coco_t-retrieval_1k_karpathy_fold{}.json'.format(1))
    with open(coco_path) as fp:
        coco1k = json.load(fp)
    return coco1k['captions']


def enrich_image_pool(pool: List[int], img2tags: Dict[str, List[str]]):
    """
    enrich pool by adding tags of images
    """
    enriched_pool = []
    for img in pool:
        img_id = str(img)
        tags   = img2tags[img_id] if img_id in img2tags else []
        enriched_pool.append({'id': img_id, 'tags': tags})
    return enriched_pool


def filter_image_retrieval(annotations: List[int], img2tags: Dict[int, int], filter: Filter, capt2txt: Dict[str, str]):
    """
    each row of the new dataset will be composed of
        {'query': str, 'pool': List[int], 'gt': int}
    """
    data = {'info': 'MSCOCO filtered image retrieval 1k', 'annotations': []}
    pbar   = tqdm(annotations['annotations'])
    for row in pbar:
        query      = capt2txt[str(row['query'])]
        pool       = enrich_image_pool(row['pool'], img2tags)
        gt         = row['pool'][0]
        filt_pool  = filter.filter(query, pool)
        annotation = {'query': row['query'],
                      'pool': [int(item['id']) for item in filt_pool],
                      'gt': gt
                    }
        data['annotations'].append(annotation)
    return data


def filter_txt_retrieval(annotations: List[int], img2tags: Dict[int, int], filter: Filter, capt2txt: Dict[str, str]):
    """
    each row of the new dataset will be composed of
        {'query': str, 'pool': List[int], 'gt': int}
    """
    data = {'info': 'MSCOCO filtered image retrieval 1k', 'annotations': []}
    pbar   = tqdm(annotations['annotations'])
    for row in pbar:
        img_id    = str(row['query'])
        query     = {'id': row['query'], 'tags': img2tags[img_id] if img_id in img2tags else []}
        pool      = [{'id': capt_id, 'txt': capt2txt[str(capt_id)]} for capt_id in row['pool']]
        gt        = row['pool'][0]
        filt_pool = filter.filter(query, pool)
        annotation = {'query': row['query'],
                      'pool': [int(item['id']) for item in filt_pool],
                      'gt': gt
                    }
        data['annotations'].append(annotation)
    return data 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--concepts_table',
        type=str,
        help='Path to concepts table for the filter',
    )
    parser.add_argument(
        '--coco_retrieval_root',
        type=str,
        help="Directory containing 5 folds for coco 1k"
    )
    parser.add_argument(
        '--img_to_objects_mapping',
        type=str,
        help='File containing the mapping from the image to the coco objects tags'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        help='Directory where to store the filtered datasets'
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.concepts_table) as fp:
        table = json.load(fp)
    with open(args.img_to_objects_mapping, 'r') as fp:
        img2tags = json.load(fp)

    filter = Filter(table)
    capt2txt = load_capt_to_txt(args)

    # Filter 5-folds COCO 1k for Image Retrieval
    print('Creating filtered image retrieval 1k')
    pbar = tqdm(range(1, N_FOLDS+1))
    for fold_n in pbar:
        coco_path = os.path.join(args.coco_retrieval_root, 'coco_i-retrieval_1k_karpathy_fold{}.json'.format(fold_n))
        with open(coco_path) as fp:
            coco1k = json.load(fp) 
        filtered_coco = filter_image_retrieval(coco1k, img2tags, filter, capt2txt)

        out_file  = os.path.join(args.out_dir, 'filtered_coco_i-retrieval_1k_karpathy_fold{}.json'.format(fold_n))
        with open(out_file, 'w') as fp:
            json.dump(filtered_coco, fp)


    # Filter 5-folds COCO 1k for Text Retrieval
    print('Creating filtered text retrieval 1k')
    pbar = tqdm(range(1, N_FOLDS+1))
    for fold_n in pbar:
        coco_path = os.path.join(args.coco_retrieval_root, 'coco_t-retrieval_1k_karpathy_fold{}.json'.format(fold_n))
        with open(coco_path) as fp:
            coco1k = json.load(fp) 
        filtered_coco = filter_txt_retrieval(coco1k, img2tags, filter, capt2txt)

        out_file  = os.path.join(args.out_dir, 'filtered_coco_t-retrieval_1k_karpathy_fold{}.json'.format(fold_n))
        with open(out_file, 'w') as fp:
            json.dump(filtered_coco, fp)

    print('Finished. Results saved at {}'.format(args.out_dir))
