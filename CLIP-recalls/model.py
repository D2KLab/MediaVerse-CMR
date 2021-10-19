import pdb
from typing import TypeVar

import clip
import torch
from PIL import Image

Tensor = TypeVar('torch.Tensor')


class CLIP():
    def __init__(self, vis_model: str, device: str = 'cpu') -> None:
        self.device = torch.device(device)
        try:
            print('loading model ...')
            self.model, self.vis_preprocess = clip.load(vis_model, device)
        except:
            print(clip.available_models())
        self.model.eval()

    def clipVisualEncode(self, img_path: str) -> Tensor:
        with torch.no_grad():
            image = self.vis_preprocess(Image.open(img_path)).unsqueeze(0).to(self.device)
            return self.model.encode_image(image)

    def clipTextEncode(self, text: str) -> Tensor:
        with torch.no_grad():
            tok_text = clip.tokenize(text).to(self.device)
            return self.model.encode_text(tok_text)
