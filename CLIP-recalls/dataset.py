import json
import os
import pdb
from typing import List, Tuple, TypeVar

import torch
from torch.utils.data import Dataset

T = TypeVar('T')




class COCOCaptions(Dataset):
    def __init__(self, root: str, annotations_file: str, images_dir: str) -> None:
        super(COCOCaptions, self).__init__()
        self.root = root
        self.anns = os.path.join(root, annotations_file)
        self.imgs = os.path.join(root, images_dir)
        with open(self.anns, 'r') as fp:
            self.data = json.load(fp)

    def __getitem__(self, index: int):
        return self.data['annotations'][index]

    def __len__(self):
        return len(self.data['annotations'])


class COCOCaptions1k(COCOCaptions):
    def __init__(self,
                root: str,
                annotations_file: str,
                images_dir: str,
                retrieval: str) -> None:
        super(COCOCaptions1k, self).__init__(root, annotations_file, images_dir)
        assert retrieval in ['image', 'text']
        self.retrieval = retrieval

    def __getitem__(self, index: int) -> Tuple[str, str, List[int]]:
        row       = self.data['annotations'][index] #todo call parent method
        if self.retrieval == 'image':
            return row['id'], row['pool']
        else:
            return row['image_id'], row['pool']

class COCOCaptions5k(COCOCaptions):
    def __init__(self,
                root: str,
                annotations_file: str,
                images_dir: str,
                retrieval: str) -> None:
        super(COCOCaptions5k, self).__init__(root, annotations_file, images_dir)
        assert retrieval in ['image', 'text']
        self.retrieval = retrieval

    def __getitem__(self, index: int) -> Tuple[str, str, List[int]]:
        row       = self.data['annotations'][index] #todo call parent method
        if self.retrieval == 'image':
            gt_id = row['image_id']
            pool  = [row['image_id']]
            pool.extend([img_id for img_id in self.data['images'] if img_id != gt_id])
            return row['id'], pool
        else:
            return row['image_id'], row['pool']
