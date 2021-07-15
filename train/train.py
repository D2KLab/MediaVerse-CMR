import json
import os
import pdb
import random
from sys import setprofile

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import CocoCaptions
from model import CMRNet


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
t_captions_json  = 'data/MSCOCO/annotations/captions_train2017.json'
t_instances_json = 'data/MSCOCO/annotations/instances_train2017.json'
v_captions_json  = 'data/MSCOCO/annotations/captions_val2017.json'
v_instances_json = 'data/MSCOCO/annotations/instances_val2017.json'
glove_embeddings = 'glove/glove.6B.300d.txt'
out_dir          = 'outs/checkpoints/'
os.makedirs(out_dir, exist_ok=True)
pad_token    = '<pad>'
unk_token    = '<unk>'
hidden_state = 768
batch_size   = 32
lr           = 1e-6
w_decay      = 0 #12e-7
n_epochs     = 15
n_workers    = 4
device       = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
MAX_N_INSTANCES = 12
MAX_N_TOKENS    = 15


tdata  = CocoCaptions(captions_json=t_captions_json, instances_json=t_instances_json, max_instances_n=MAX_N_INSTANCES, max_tokens_n=MAX_N_TOKENS, pad_token=pad_token)
vdata  = CocoCaptions(captions_json=v_captions_json, instances_json=v_instances_json, max_instances_n=MAX_N_INSTANCES, max_tokens_n=MAX_N_TOKENS, pad_token=pad_token)
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': n_workers,
          'pin_memory': True if device.type == 'cuda' else False}
tloader = DataLoader(tdata, **params, collate_fn=tdata.collate_fn)
vloader = DataLoader(vdata, **params, collate_fn=vdata.collate_fn)

model     = CMRNet(embeddings_file=glove_embeddings, hidden_state=hidden_state, unk_token=unk_token, pad_token=pad_token).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
step      = 0
best_loss = np.inf
step_loss = []
t_losses  = []
v_losses  = []
writer    = SummaryWriter()
for curr_ep in tqdm(range(1, n_epochs+1)):
    curr_losses = []
    model.train()
    #? soft labels with "difference" between instances vectors? Check the batch
    for batch in tloader:
        step += 1
        out  = model(batch['captions'], batch['instances'])
        loss = criterion(out, batch['labels'].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        curr_losses.append(loss.item())
        writer.add_scalar('Loss/step', loss.item(), step)
    t_avg_loss = sum(curr_losses) / len(curr_losses)
    t_losses.append(t_avg_loss)
    step_loss.extend(curr_losses)
    curr_losses = []
    model.eval()
    with torch.no_grad():
        for batch in vloader:
            out  = model(batch['captions'], batch['instances'])
            loss = criterion(out, batch['labels'].to(device))
            curr_losses.append(loss.item())
    v_avg_loss = sum(curr_losses) / len(curr_losses)
    v_losses.append(v_avg_loss)
    if v_avg_loss < best_loss:
        best_loss = v_avg_loss
        torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pth'))
        with open(os.path.join(out_dir, 'word2id.json'), 'w') as fp:
            json.dump(model.word2id, fp)
    writer.add_scalars('Loss/epochs', {'train': t_avg_loss, 'val': v_avg_loss}, curr_ep)
    print('Epoch {}, tloss = {}, vloss = {}'.format(curr_ep, t_avg_loss, v_avg_loss))

print('Finished')
        


