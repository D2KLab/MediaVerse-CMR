"""
@author: Matteo A. Senese

This script creates the MSCOCO-5K split (both text and image retrieval).
Each row of this split consists in 1 query and a retrieval pool of 5000 contents.

Image retrieval: caption as query, 5000 images in the retrieval pool
Text retrieval: image as query, 5000 captions in the retrieval pool
"""


import json
import random
from typing import Dict, List

import os
from tqdm import tqdm
import argparse

K         = 5000


def create_image_retrieval(images: List[int], cap2img: Dict[int, int]):
    data = {'info': 'MSCOCO image retrieval 5k', 'images': images, 'annotations': []}
    pbar   = tqdm(cap2img.items())
    for capid, imgid in pbar:
        gt_img   = imgid
        curr_ann = {'query': capid, 'gt': gt_img}
        data['annotations'].append(curr_ann)
    return data


def create_text_retrieval(images: List[int], img2cap: Dict[int, List[int]], captions: List[int]):
    capt_list = [capt_id for capt_id in captions.keys()]
    data      = {'info': 'MSCOCO text retrieval 5k', 'captions': captions, 'annotations': []}
    pbar      = tqdm(images)
    for img in pbar:
        img_id   = img
        gt_capts = img2cap[img_id] # ~ 5 ground truth captions for each image
        curr_ann = {'query': img, 'pool': [random.choice(gt_capts)]}
        while len(curr_ann['pool']) < K:
            rand_idx  = random.randint(0, len(captions)-1)
            rand_capt = capt_list[rand_idx] 
            if rand_capt in gt_capts:
                continue
            curr_ann['pool'].append(rand_capt)
        data['annotations'].append(curr_ann)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root',
        type=str,
        help="Path to MSCOCO root folder"
    )
    parser.add_argument(
        '--annotations_file',
        type=str,
        help='Relative path of the annotation file wrt specified data root'
    )
    args = parser.parse_args()

    with open(os.path.join(args.data_root, args.annotations_file)) as fp:
        data = json.load(fp)

    #collect captions
    captions = {inner['sentid']: inner['raw'] for item in data['images'] if item['split'] == 'test' for inner in item['sentences']} #{row['id']: row['caption'] for row in dataset}
    images   = [item['cocoid'] for item in data['images'] if item['split'] == 'test'] #[row['id'] for row in dataset.data['images']]
    #collect images and related captions
    img2cap = {}
    for row in data['images']:
        if row['split'] != 'test':
            continue
        img_id = row['cocoid']
        if img_id not in img2cap:
            img2cap[img_id] = []
        img2cap[img_id].extend([item['sentid'] for item in row['sentences']])
    cap2img = {}
    for imgid in img2cap.keys():
        for captid in img2cap[imgid]:
            cap2img[captid] = imgid

    print('Creating image retrieval 5k')
    data = create_image_retrieval(images, cap2img)
    with open('coco_i-retrieval_5k_karpathy.json', 'w') as fp:
        json.dump(data, fp)

    print('Creating text retrieval 5k')
    data = create_text_retrieval(images, img2cap, captions)
    with open('coco_t-retrieval_5k_karpathy.json', 'w') as fp:
        json.dump(data, fp)