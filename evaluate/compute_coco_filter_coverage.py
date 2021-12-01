"""
@author: Matteo A. Senese

This script compute the coverage of the filter on the MSCOCO dataset.
The coverage is computed as the ratio between the number of captions that can be expanded with the filter and the total number of captions.
A caption can be expanded if and only if exists at least on token in the captions that is also present in the concept table.
"""

import argparse
import json

from common import Filter
from tqdm import tqdm

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
    args = parser.parse_args()

    with open(args.concepts_table, 'r') as fp:
        table = json.load(fp)
    with open(args.coco_retrieval, 'r') as fp:
        data  = json.load(fp)
    with open(args.coco_captions, 'r') as fp:
        id2cap = json.load(fp)

    filter   = Filter(table)
    missings = 0
    for row in tqdm(data['annotations']):
        queryid = str(row['query'])
        query = id2cap[queryid]
        res = filter.query_expansion(query)
        if len(res) == 0:
            missings += 1
    tot = len(data['annotations'])
    print('Coverage: {}'.format(100*(tot-missings)/tot))



