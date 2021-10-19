from model import Tensor
from dataset import T
import os
from typing import Dict, List, TypeVar

from tqdm import tqdm

COCODataset = TypeVar('COCODataset')
CLIP        = TypeVar('CLIP')
Tensor      = TypeVar('torch.Tensor')
T           = TypeVar('T') #generic



def create_cache(datasets: COCODataset, content: str, model: CLIP) -> Dict:
    cache = {}
    assert content in ['image', 'text']
    print('caching {}'.format(content))
    if content == 'image':
        img_folder = datasets.imgs
        for img_id in tqdm(datasets.data['images']):
            if img_id not in cache:
                cache[img_id] = model.clipVisualEncode(os.path.join(img_folder, str(img_id)+'.jpg')).to('cpu')
    else:
        for capt_id, caption in tqdm(datasets.data['captions'].items()):
            if capt_id not in cache:
                cache[int(capt_id)] = model.clipTextEncode(caption).to('cpu')
    return cache

def update_recalls(scores: List[float], recall: Dict[str, int]) -> None:
    ref = scores[0]
    pos = 1
    for score in scores[1:]:
        pos += 1 if score > ref else 0
        if pos > 10:
            break
    recall['R@1']  += 1 if pos <= 1  else 0
    recall['R@5']  += 1 if pos <= 5  else 0
    recall['R@10'] += 1 if pos <= 10 else 0
