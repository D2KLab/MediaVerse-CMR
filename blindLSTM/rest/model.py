import pdb
from typing import List

import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F

N_EMBEDDINGS    = 400002
EMBEDDINGS_SIZE = 300 

class CMRNet(nn.Module):

    def __init__(self, word2id: dict, hidden_state: int=768, pad_token: str='<pad>', unk_token: str='<unk>'):
        super(CMRNet, self).__init__()
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.emb_size  = EMBEDDINGS_SIZE

        self.word2id            = word2id
        self.word2id[pad_token] = 0
        self.word2id[unk_token] = 1
        #initialize layers
        self.embedding = nn.Embedding(num_embeddings=N_EMBEDDINGS,
                                      embedding_dim=EMBEDDINGS_SIZE,
                                      padding_idx=0)
        self.caption_rnn    = nn.LSTM(self.emb_size, hidden_state, batch_first=True, bidirectional=True, num_layers=2)
        self.instances_rnn  = nn.LSTM(self.emb_size, hidden_state, batch_first=True, bidirectional=True, num_layers=2)
        self.out_layer      = nn.Sequential(nn.Linear(4*hidden_state, 2*hidden_state),
                                            nn.LayerNorm(2*hidden_state),
                                            nn.ReLU(),
                                            nn.Linear(2*hidden_state, hidden_state),
                                            nn.LayerNorm(hidden_state),
                                            nn.ReLU(),
                                            nn.Linear(hidden_state, 2)
                                            )

    def forward(self, captions: List[str], instances: List[str]) -> torch.Tensor:
        batch_size    = len(captions)
        device        = self.embedding.weight.device
        captions_in   = self.text_to_ids(captions).to(device)
        instances_in  = self.text_to_ids(instances).to(device)
        captions_emb  = self.embedding(captions_in)
        instances_emb = self.embedding(instances_in)
        capt_out      = self.caption_rnn(captions_emb)[1][1].view(self.caption_rnn.num_layers,
                                                                  2 if self.caption_rnn.bidirectional else 1,
                                                                  batch_size,
                                                                  self.caption_rnn.hidden_size)[-1] #last layer
        inst_out      = self.instances_rnn(instances_emb)[1][1].view(self.instances_rnn.num_layers,
                                                                     2 if self.instances_rnn.bidirectional else 1,
                                                                     batch_size,
                                                                     self.instances_rnn.hidden_size)[-1] #last layer
        if self.caption_rnn.bidirectional:
            capt_out = torch.cat((capt_out[0, :, :], capt_out[1, :, :]), dim=-1)
        if self.instances_rnn.bidirectional:
            inst_out = torch.cat((inst_out[0, :, :], inst_out[1, :, :]), dim=-1)

        out           = self.out_layer(torch.cat((capt_out, inst_out), dim=-1))
        return out

    def infer(self, query: str, instances: List[str]) -> float:
        query  = nltk.word_tokenize(query)
        query  = [query] * len(instances)
        out    = self(query, instances)
        scores = F.softmax(out, dim=-1)[:, -1]
        return scores.tolist()

    def text_to_ids(self, text_l: List[str]) -> torch.Tensor:
        ids_l = []
        for text in text_l:
            ids_l.append([self.word2id[w.lower() if w.lower() in self.word2id else self.unk_token] for w in text])
        return torch.tensor(ids_l)



