import json
import pdb
import random
from typing import List, Dict, Tuple

import nltk
import torch
from numpy import pad
from torch.utils.data import Dataset


class CocoCaptions(Dataset):
    """COCO class that return matching (capt, objects) pairs 50% of times
    """
    def __init__(self, captions_json: str, instances_json: str, max_tokens_n: int, max_instances_n: int, pad_token: str='<pad>'):
        super(CocoCaptions, self).__init__()
        self.max_tokens_n    = max_tokens_n
        self.max_instances_n = max_instances_n
        self.pad_token       = pad_token
        with open(captions_json, 'r') as fp:
            capt_data = json.load(fp)
        with open(instances_json, 'r') as fp:
            inst_data = json.load(fp)
        assert len(capt_data['images']) == len(inst_data['images'])
        self.data   = [item for item in capt_data['annotations']]
        self.images = {item['id']: {'file_name': item['file_name'], 'instances': []} for item in capt_data['images']}
        for item in inst_data['annotations']:
            self.images[item['image_id']]['instances'].append(item['category_id'])
        self.id2inst = {item['id']: item['name'] for item in inst_data['categories']}
    
    def __getitem__(self, index: int) -> Tuple[str, str, int]:
        capt, inst = self._get_sample(index)
        if random.uniform(0, 1) > .5:
            return capt, inst, 1
        else:
            #negative sample
            neg_idx = index
            while neg_idx == index:
                neg_idx = random.randint(0, len(self)-1)
            neg_capt, neg_inst = self._get_sample(neg_idx) 
            if random.uniform(0, 1) > .5:
                return neg_capt, inst, 0
            return capt, neg_inst, 0

    def _get_sample(self, index: int) -> Tuple[str, str]:
        item  = self.data[index]
        capt  = item['caption']
        #create a set: remove duplicates instances
        inst  = {self.id2inst[id] for id in self.images[item['image_id']]['instances']}
        return capt, inst

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch: List) -> Dict:
        captions  = [nltk.word_tokenize(item[0]) for item in batch]
        instances = [' '.join(item[1]) for item in batch]
        instances = [nltk.word_tokenize(inst) for inst in instances]
        labels    = [item[-1] for item in batch]
        #padding
        for idx in range(len(captions)):
            captions[idx] = captions[idx][:self.max_tokens_n]
            pad_amount    = self.max_tokens_n - len(captions[idx])
            captions[idx].extend([self.pad_token] * pad_amount)
        for idx in range(len(instances)):
            instances[idx] = instances[idx][:self.max_instances_n]
            pad_amount     = self.max_instances_n - len(instances[idx])
            instances[idx].extend([self.pad_token] * pad_amount)
        labels = torch.tensor(labels)
        return {'captions': captions, 'instances': instances, 'labels': labels}