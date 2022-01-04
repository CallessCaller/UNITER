import torch
import torch.nn as F
from model.pretrain_vcr import UniterForPretrainingForVCR
from transformers import BertTokenizer

import json
import numpy as np
import pandas as pd

annotPATH = '/home/vcr/vcr1annots/'
imagePATH = '/home/vcr/vcr1images/'

'''
def forward(self, batch, task, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']
        txt_type_ids = batch['txt_type_ids']
'''

train = pd.read_json(path_or_buf=annotPATH + 'train.jsonl', lines=True)
val = pd.read_json(path_or_buf=annotPATH + 'val.jsonl', lines=True)

