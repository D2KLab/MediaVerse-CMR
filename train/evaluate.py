#import libraries and modules
import json
import os
import pdb
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import CocoCaptions
from model import CMRNet

SEED = 1234
torch.backends.cudnn.deterministic = True
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


VALSET      = 'data/dataval_np.csv'
INSTANCES   = 'data/coco_object_features/features'
EMBEDDINGS  = 'data/glove.6B.300d.txt'
PRETRAINED  = 'model_50_correct.pth'
USE_CUDA    = False
GPU_N       = '5' 



def evaluate_prec_rec_f1(model, iterator):

    '''
    calculate precision, recall, F1 for validation set

    '''
    targets = []
    preds   = []
    
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(iterator):
            logits     = model(batch['captions'], batch['instances'])
            curr_preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1).tolist()

            targets.extend(batch['labels'].view(-1).tolist())
            preds.extend(curr_preds)
        
        acc = sum([i == j for i, j in zip(preds, targets)]) / len(preds) #binary accuracy
        prec = precision_score(preds, targets)
        rec  = recall_score(preds, targets)
        f1   = f1_score(preds, targets)

    return acc, prec,  rec, f1




def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed = 26
seed_everything(seed)

#parameters
v_captions_json  = 'data/MSCOCO/annotations/captions_val2017.json'
v_instances_json = 'data/MSCOCO/annotations/instances_val2017.json'
glove_embeddings = 'glove/glove.6B.300d.txt'
pad_token    = '<pad>'
unk_token    = '<unk>'
hidden_state = 768
batch_size   = 32
n_workers    = 4
use_cuda     = False
device       = torch.device('cuda:5' if torch.cuda.is_available() and use_cuda else 'cpu')
MAX_N_INSTANCES = 12
MAX_N_TOKENS    = 15


vdata  = CocoCaptions(captions_json=v_captions_json, instances_json=v_instances_json, max_instances_n=MAX_N_INSTANCES, max_tokens_n=MAX_N_TOKENS, pad_token=pad_token)
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': n_workers,
          'pin_memory': True if device.type == 'cuda' else False}
vloader = DataLoader(vdata, **params, collate_fn=vdata.collate_fn)

model     = CMRNet(embeddings_file=glove_embeddings, hidden_state=hidden_state, unk_token=unk_token, pad_token=pad_token).to(device)
model.load_state_dict(torch.load('outs/checkpoints/best_model.pth', map_location=device))



#yield scores for create a dataframe
val_loss_   = []
val_acc_    = []
val_prec_   = []
val_recall_ = []
val_F1_     = []

#evaluate model
val_acc, val_prec, val_recall, val_F1 = evaluate_prec_rec_f1(model, vloader)

val_acc_.append(f'{val_acc*100:.2f}')
val_prec_.append(f'{val_prec*100:.2f}')
val_recall_.append(f'{val_recall*100:.2f}')
val_F1_.append(f'{val_F1*100:.2f}')

"""
#create dataframe with metrics
evaluation_df = pd.DataFrame(val_loss_)
evaluation_df = evaluation_df.rename(columns={0:'val_loss_'})
evaluation_df['val_acc']= val_acc_
evaluation_df['val_prec']= val_prec_
evaluation_df['val_rec'] = val_recall_
evaluation_df['val_F1']= val_F1_
"""


print(f'| Val. Acc: {val_acc*100:.2f}% | Val precision: {val_prec*100:.2f}% | Val recall: {val_recall *100:.2f}% | Val F1:{val_F1*100:.2f}% |')
