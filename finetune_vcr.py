import torch
import torch.nn.functional as F
import torch.cuda.amp as amp
import torch.cuda.amp.autocast_mode
from torch.utils.data import DataLoader, ConcatDataset

from model.vcr import UniterForVisualCommonsenseReasoning
from prepro import FinetuneDataForVCR, ValidationDataForVCR, vcr_collate, vcr_val_collate
from transformers import AdamW, get_linear_schedule_with_warmup

import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import argparse

from torch.utils.tensorboard import SummaryWriter

# random seed
torch.random.manual_seed(42)

# config 
parser = argparse.ArgumentParser(description='Config')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--accum_steps', type=int, default=5)
args = parser.parse_args()

warmup_steps = 800
accum_steps = args.accum_steps
num_train_steps = 20000
valid_steps = num_train_steps // 10
batch_size = args.batch_size #4000
val_batch_size = 8
learning_rate = 6e-05

import time
import os
current_time = time.time()
os.mkdir(f'ckpt/{current_time}')

writer = SummaryWriter(f"./log_finetune/{batch_size}_{accum_steps}_{learning_rate}_{current_time}")

print('Loading dataset...')
qa_dataset = FinetuneDataForVCR(data_type='train', task='qa')
#qa_dataloader = DataLoader(qa_dataset, batch_size=batch_size, shuffle=True, collate_fn=vcr_collate, num_workers=10)
qar_dataset = FinetuneDataForVCR(data_type='train', task='qar')
#qar_dataloader = DataLoader(qar_dataset, batch_size=batch_size, shuffle=True, collate_fn=vcr_collate, num_workers=10)

train_dataset = ConcatDataset([qa_dataset, qar_dataset])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=vcr_collate, num_workers=10)

val_dataset = ValidationDataForVCR(data_type='val')
val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=vcr_val_collate, num_workers=10)
print('Done !!!')

# model
print('Loading model...')
#checkpoint = torch.load('ckpt/UNITER_2nd_45000_32_4')
checkpoint = torch.load('pretrained/uniter-base.pt')
model = UniterForVisualCommonsenseReasoning.from_pretrained('config/uniter-base.json', checkpoint, img_dim=2048)
# model.config.type_vocab_size = 4
model.init_type_embedding()
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
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_train_steps)
scaler = amp.GradScaler()
loss_sum = 0
accum = 0

current_steps = 0
breakValue = False

def compute_accuracies(out_qa, labels_qa, out_qar, labels_qar):
    outputs_qa = out_qa.max(dim=-1)[1]
    outputs_qar = out_qar.max(dim=-1)[1]
    matched_qa = outputs_qa.squeeze() == labels_qa.squeeze()
    matched_qar = outputs_qar.squeeze() == labels_qar.squeeze()
    matched_joined = matched_qa & matched_qar
    n_correct_qa = matched_qa.sum().item()
    n_correct_qar = matched_qar.sum().item()
    n_correct_joined = matched_joined.sum().item()
    return n_correct_qa, n_correct_qar, n_correct_joined

@torch.no_grad()
def validate(model, val_loader):
    print('Start running validation...')
    model.eval()
    val_qa_loss, val_qar_loss = 0, 0
    tot_qa_score, tot_qar_score, tot_score = 0, 0, 0
    n_ex = 0
    for i, batch in enumerate(tqdm(val_loader)):
        scores = model(batch, compute_loss=False)
        qa_targets = batch['qa_targets'].cuda()
        qar_targets = batch['qar_targets'].cuda()
        qids = batch['qids']
        scores = scores.view(len(qids), -1)
        vcr_qa_loss = F.cross_entropy(
                scores[:, :4], qa_targets.squeeze(-1), reduction="sum")
        if scores.shape[1] > 8:
            qar_scores = []
            for batch_id in range(scores.shape[0]):
                answer_ind = qa_targets[batch_id].item()
                qar_index = [4+answer_ind*4+i
                             for i in range(4)]
                qar_scores.append(scores[batch_id, qar_index])
            qar_scores = torch.stack(qar_scores, dim=0)
        else:
            qar_scores = scores[:, 4:]
        # print(qar_scores, qar_targets)
        # tensor([], device='cuda:0', size=(10, 0))
        vcr_qar_loss = F.cross_entropy(
            qar_scores, qar_targets.squeeze(-1), reduction="sum")
        val_qa_loss += vcr_qa_loss.item()
        val_qar_loss += vcr_qar_loss.item()
        curr_qa_score, curr_qar_score, curr_score = compute_accuracies(
            scores[:, :4], qa_targets, qar_scores, qar_targets)
        tot_qar_score += curr_qar_score
        tot_qa_score += curr_qa_score
        tot_score += curr_score
        n_ex += len(qids)

    val_qa_loss /= n_ex
    val_qar_loss /= n_ex
    val_qa_acc = tot_qa_score / n_ex
    val_qar_acc = tot_qar_score / n_ex
    val_acc = tot_score / n_ex

    writer.add_scalar("valid/vcr_qa_loss", val_qa_loss, current_steps)
    writer.add_scalar("valid/vcr_qar_loss", val_qar_loss, current_steps)
    writer.add_scalar("valid/acc_qa", val_qa_acc, current_steps)
    writer.add_scalar("valid/acc_qar", val_qar_acc, current_steps)
    writer.add_scalar("valid/acc", val_acc, current_steps)
    print(f"Score_qa: {val_qa_acc*100:.2f} | Score_qar: {val_qar_acc*100:.2f} | Score_total: {val_acc*100:.2f}")
    writer.flush()
    model.train()

with tqdm(total=num_train_steps) as pbar:
    for epoch in range(100):
        for i, batch in enumerate(train_dataloader):
            with amp.autocast():
                loss = model(batch, compute_loss=True)
                loss = loss.mean()

            scaler.scale(loss).backward()
            loss_sum += loss.item()
            accum += 1

            if accum != accum_steps:
                writer.flush()
                continue

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            accum = 0
            current_steps += 1
            pbar.update(1)

            writer.add_scalar("lr", optimizer.param_groups[0]['lr'], current_steps)
            writer.add_scalar("total_loss", loss_sum/accum_steps, current_steps)

            loss_sum = 0

            writer.flush()

            # validation & model save
            if current_steps % valid_steps == 0:
                validate(model, val_dataloader)
                torch.save(model.state_dict(), f'./ckpt/{current_time}/UNITER_VCR_{current_steps}_{batch_size}_{accum_steps}_{learning_rate}')

            if current_steps == num_train_steps:
                breakValue = True

        if breakValue:
            break