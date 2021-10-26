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
                images_dir: str) -> None:
        super(COCOCaptions1k, self).__init__(root, annotations_file, images_dir)
        self.info = self.data['info']

    def __getitem__(self, index: int) -> Tuple[str, str, List[int]]:
        row       = self.data['annotations'][index] #todo call parent method
        return row['query'], row['pool']


class COCOCaptions5k(COCOCaptions):
    def __init__(self,
                root: str,
                annotations_file: str,
                images_dir: str) -> None:
        super(COCOCaptions5k, self).__init__(root, annotations_file, images_dir)
        self.info = self.data['info']

    def __getitem__(self, index: int) -> Tuple[str, str, List[int]]:
        row       = self.data['annotations'][index] #todo call parent method
        if 'pool' in row: #text retrieval
            pool = row['pool']
        else: #img retrieval
            pool = [row['gt']] + [imgid for imgid in self.data['images'] if imgid != row['gt']]
        return row['query'], pool
