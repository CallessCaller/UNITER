# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as F
import torch.cuda.amp as amp
import torch.cuda.amp.autocast_mode

from model.pretrain_vcr import UniterForPretrainingForVCR
from prepro import PretrainDataForVCR, DataLoader, collate
from transformers import AdamW, get_linear_schedule_with_warmup

import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

# random seed
torch.random.manual_seed(42)

'''
 batch = {'input_ids': input_ids.long(),
        'txt_type_ids': txt_type_ids.long(),
        'position_ids': position_ids.long(),
        'img_feat': img_feat.long(),
        'img_pos_feat': img_pos.long(),
        'attn_masks': attn_masks.long(),
        'gather_index': gather_index.long(),
        'masked_input_ids': maksed_tokenzied.long(),
        'txt_labels': txt_label.long(),
        'label_targets': label_targets.long(),
        'masked_img_feat': masked_img_feat.long(),
        'feat_targets': feat_target.long()}
'''

# config 
num_train_steps = 10 #45000
warmup_steps = 4500
accum_steps = 16
valid_steps = 2000
batch_size = 16 // accum_steps #6144
val_batch_size = 8000 // accum_steps,
learning_rate = 3e-05

# dataloader
print('Loading dataset...')
# train_dataset = PretrainDataForVCR(data_type='train')
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate,
#                               prefetch_factor=5, num_workers=5)
# val_dataset = PretrainDataForVCR(data_type='val')
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate,
#                             prefetch_factor=5, num_workers=5)
train_dataset = PretrainDataForVCR(data_type='val')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate,
                              prefetch_factor=5, num_workers=5)
print('Done !!!')

# model
print('Loading model...')
checkpoint = torch.load('pretrained/uniter-base.pt')
model = UniterForPretrainingForVCR.from_pretrained('config/uniter-base.json', checkpoint, img_dim=2048, img_label_dim=1601)
model.config.vocab_size = 30522
model.config.type_vocab_size = 4
model.init_type_embedding()
model.init_word_embedding(81)
model.cuda()
model.train()
print('Done !!!')

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]


# optimizer
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, 4500, num_train_steps)
scaler = amp.GradScaler()
loss_sum = 0
accum = 0


current_step = 0

breakValue = False

for epoch in range(100):
        for i, batch in enumerate(tqdm(train_dataloader)):
                task_prob = torch.rand(1)
                if task_prob > 0.66:
                        task = 'mlm'
                        batch['input_ids'] = batch['masked_input_ids']
                elif task_prob > 0.33:
                        task = 'mrc'
                        batch['img_feat'] = batch['masked_img_feat']
                else:
                        task = 'mrfr'
                        batch['img_feat'] = batch['masked_img_feat']

                with amp.autocast():
                        loss = model(batch, task=task, compute_loss=True)
                        loss = loss.mean()

                scaler.scale(loss).backward()

                loss_sum += loss.item()
                accum += 1

                if accum != accum_steps:
                        continue
                
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
                optimizer.zero_grad()

                accum = 0
                current_step += 1

                if current_step == num_train_steps: 
                        breakValue = True
                        break
        if breakValue:
                break


