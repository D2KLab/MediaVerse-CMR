import pdb

import nltk
import numpy as np
from tqdm import tqdm

from data import CocoCaptions

t_captions_json  ='data/MSCOCO/annotations/captions_train2017.json'
t_instances_json ='data/MSCOCO/annotations/instances_train2017.json'

tdata  = CocoCaptions(captions_json=t_captions_json, instances_json=t_instances_json, max_instances_n=1000, max_tokens_n=1000, pad_token='<pad>')

print('Computing captions average length (in tokens)')
capt_lens = [len(nltk.word_tokenize(data['caption'])) for data in tqdm(tdata.data)]
print('Computing instance number per image')
inst_lens = [len(data['instances']) for data in tqdm(tdata.images.values())]
print('Mean and std over #tokens: {}, {}'.format(round(np.mean(capt_lens), 2), round(np.std(capt_lens), 2)))
print('Mean and std over #instances: {}, {}'.format(round(np.mean(inst_lens), 2), round(np.std(inst_lens), 2)))
