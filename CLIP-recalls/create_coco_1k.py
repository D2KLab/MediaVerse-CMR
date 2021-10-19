import json
import os
import pdb
import random
from typing import Dict, List

from tqdm import tqdm

from dataset import COCOCaptions

N_FOLDS   = 5
K         = 1000
DATA_ROOT = 'data/MSCOCO'
ANN_FILES = 'annotations/captions_val2017.json'
IMG_DIR   = 'images'

dataset = COCOCaptions(DATA_ROOT, ANN_FILES, IMG_DIR)

#collect captions
captions = {row['id']: row['caption'] for row in dataset}
images   = [row['id'] for row in dataset.data['images']]
#collect images and related captions
img2cap = {}
for row in dataset:
    img_id = row['image_id']
    row_id = row['id']
    if img_id not in img2cap:
        img2cap[img_id] = [row_id]
    else:
        img2cap[img_id].append(row_id)
gt_annotations = [row for row in dataset]


def create_image_retrieval(images: List[int], gt_annotations: Dict, k: int, desc: str):
    data   = {'info': 'MSCOCO image retrieval 1k', 'images': images, 'annotations': []}
    pbar   = tqdm(gt_annotations)
    pbar.set_description(desc)
    for row in pbar:
        gt_img   = row['image_id']
        curr_ann = {'id': row['id'], 'caption': row['caption'], 'pool': [gt_img]}
        while len(curr_ann['pool']) < k:
            rand_idx = random.randint(0, len(images)-1)
            rand_img = images[rand_idx]
            if rand_img == gt_img:
                continue
            curr_ann['pool'].append(rand_img)
        data['annotations'].append(curr_ann)
    return data

def create_text_retrieval(images: List[int], img2cap: Dict[int, List[int]], captions: List[int], k: int, desc: str):
    capt_list = [capt_id for capt_id in captions.keys()]
    data      = {'info': 'MSCOCO text retrieval 1k', 'captions': captions, 'annotations': []}
    pbar      = tqdm(images)
    pbar.set_description(desc)
    for img in pbar:
        img_id   = img
        gt_capts = img2cap[img_id] # ~ 5 ground truth captions for each image
        curr_ann = {'image_id': img, 'pool': [random.choice(gt_capts)]}
        while len(curr_ann['pool']) < k:
            rand_idx  = random.randint(0, len(captions)-1)
            rand_capt = capt_list[rand_idx] 
            if rand_capt in gt_capts:
                continue
            curr_ann['pool'].append(rand_capt)
        data['annotations'].append(curr_ann)
    return data


print('Creating image retrieval 1k')
for fold in range(N_FOLDS):
    random.seed(fold)
    data = create_image_retrieval(images, gt_annotations, K, desc='{}/{}'.format(fold+1, N_FOLDS))
    with open('coco_i-retrieval_1k_val2017_fold{}.json'.format(fold+1), 'w') as fp:
        json.dump(data, fp)

print('Creating text retrieval 1k')
for fold in range(N_FOLDS):
    random.seed(N_FOLDS+fold)
    data = create_text_retrieval(images, img2cap, captions, K, desc='{}/{}'.format(fold+1, N_FOLDS))
    with open('coco_t-retrieval_1k_val2017_fold{}.json'.format(fold+1), 'w') as fp:
        json.dump(data, fp)