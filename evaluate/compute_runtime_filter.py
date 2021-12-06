"""
@author: Matteo A. Senese

This script compute the retrieval run time on MSCOCO using the filtering approach.
The filtering approach consists on reducing the size of the retrieval pool before doing the ranking of contents.
The pool size reduction is made using query expansion from the concepts table.
"""


import argparse
import json
from datetime import datetime
from typing import Dict

import torch
from torch.utils.data import Dataset
from common import Filter
from common.utils import compute_cosine_similarity, top_k
from tqdm import tqdm



class MyDataset(Dataset):
    def __init__(self, ann):
        super().__init__()
        self._data = ann

    def __getitem__(self, index):
        return self._data[index]['query'], self._data[index]['pool']

    def __len__(self):
        return len(self._data)


def create_filtered_pools(data, filter):
    new_ann = []
    for row in tqdm(data['annotations']):
        queryid = str(row['query'])
        query   = id2cap[queryid]
        curr_pool = filter.filter(query, pool)
        new_ann.append({'query': queryid, 'pool': curr_pool})
    return new_ann


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--coco_retrieval',
        type=str,
        help='MSCOCO coco retrieval with objects file'
    )
    parser.add_argument(
        '--coco_captions',
        type=str,
        help='File containing the mapping from caption id to text'
    )
    parser.add_argument(
        '--concepts_table',
        type=str,
        help='JSON file containing the concepts table'
    )
    parser.add_argument(
        '--txt_cache',
        type=str,
        help=''
    )
    parser.add_argument(
        '--visual_cache',
        type=str,
        help=''
    )
    args = parser.parse_args()

    with open(args.concepts_table, 'r') as fp:
        table = json.load(fp)
    with open(args.coco_retrieval, 'r') as fp:
        data  = json.load(fp)
    with open(args.coco_captions, 'r') as fp:
        id2cap = json.load(fp)
    t_cache = torch.load(args.txt_cache)
    v_cache = torch.load(args.visual_cache)

    filter   = Filter(table)
    # 48 images missing
    pool  = [{'id': imgid, 'tags': data['objects'][str(imgid)]} for imgid in data['images'] if str(imgid) in data['objects']]
    t_start = datetime.now()
    print('Filtering pools ...')
    new_ann      = create_filtered_pools(data, filter)
    t_filter_end = datetime.now()
    print('Doing retrieval ...')
    k = 5000
    for row in tqdm(new_ann):
        queryid     = row['query']
        pool        = row['pool']
        query_feats = t_cache[int(queryid)].type(torch.float32)
        pool_feats  = torch.stack([v_cache[item['id']] for item in pool]).type(torch.float32)
        scores      = compute_cosine_similarity(query_feats, pool_feats)
        top_k(scores.unsqueeze(0).numpy(), k=k, descending=True)

    t_end = datetime.now()
    print('filtering pool: {} s per query'.format((t_filter_end-t_start)/len(new_ann)))
    print('filtering pool + cosine + top_k (k={}): {} s per query'.format(k, (t_end-t_start)/len(new_ann)))    