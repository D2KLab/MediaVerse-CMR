import json
import os
import pdb
import random
from typing import Dict, List

from tqdm import tqdm
import argparse


K         = 1000


def create_image_retrieval(images: List[int], cap2img: Dict[int, int], k: int, desc: str):
    data   = {'info': 'MSCOCO image retrieval 1k', 'images': images, 'annotations': []}
    pbar   = tqdm(cap2img.items())
    pbar.set_description(desc)
    for capid, imgid in pbar:
        gt_img   = imgid
        curr_ann = {'query': capid, 'pool': [gt_img]}
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
        curr_ann = {'query': img, 'pool': [random.choice(gt_capts)]}
        while len(curr_ann['pool']) < k:
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
        help="Path to MSCOCO's root folder"
    )
    parser.add_argument(
        '--annotations_file',
        type=str,
        help='Relative path of the karpathy split file wrt specified data root'
    )
    parser.add_argument(
        '--nfolds',
        type=int,
        help="Number of folds to generate"
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

    print('Creating image retrieval 1k')
    for fold in range(args.nfolds):
        random.seed(fold)
        data = create_image_retrieval(images, cap2img, K, desc='{}/{}'.format(fold+1, args.nfolds))
        with open('coco_i-retrieval_1k_karpathy_fold{}.json'.format(fold+1), 'w') as fp:
            json.dump(data, fp)

    print('Creating text retrieval 1k')
    for fold in range(args.nfolds):
        random.seed(args.nfolds+fold)
        data = create_text_retrieval(images, img2cap, captions, K, desc='{}/{}'.format(fold+1, args.nfolds))
        with open('coco_t-retrieval_1k_karpathy_fold{}.json'.format(fold+1), 'w') as fp:
            json.dump(data, fp)