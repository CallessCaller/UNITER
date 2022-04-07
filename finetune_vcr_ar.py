import torch
import torch.nn.functional as F
import torch.cuda.amp as amp
import torch.cuda.amp.autocast_mode
from torch.utils.data import DataLoader, ConcatDataset

from model.vcr import UniterForVisualCommonsenseReasoning
from prepro import ValidationDataForVCR_AR, vcr_val_collate_ar
from transformers import AdamW, get_linear_schedule_with_warmup

import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import argparse

from torch.utils.tensorboard import SummaryWriter

# random seed
torch.random.manual_seed(42) # 42, 29837, 854769803

# config 
parser = argparse.ArgumentParser(description='Config')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--accum_step', type=int, default=128) # 기존 모델과 같은 문제 수  4*128
parser.add_argument('--train_step', type=int, default=5000)
parser.add_argument('--val_step', type=int, default=1000)
parser.add_argument('--lr', type=float, default=6e-5)
parser.add_argument('--ckpt', type=str, default='pretrained/uniter-base.pt')
args = parser.parse_args()


batch_size = args.batch_size #4000
accum_steps = args.accum_step
num_train_steps = args.train_step
ckpt = args.ckpt
ckpt_short = ckpt.split('/')[1].replace('.pt', '')
warmup_steps = num_train_steps // 10
valid_steps = args.val_step #1000 if num_train_steps / 10 < 1000 else num_train_steps /10
val_batch_size = 16
learning_rate = args.lr

import time
import os
current_time = time.localtime()
current_time = time.strftime('%c', current_time)
os.mkdir(f'ckpt/{current_time}')

writer = SummaryWriter(f"./log_finetune/{ckpt_short}_{batch_size}_{accum_steps}_{learning_rate}_{current_time}")

print('Loading dataset...')
train_dataset = ValidationDataForVCR_AR(data_type='train')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=vcr_val_collate_ar, num_workers=10)

val_dataset = ValidationDataForVCR_AR(data_type='val')
val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=vcr_val_collate_ar, num_workers=10)
print('Done !!!')

# model
print('Loading model...')
checkpoint = torch.load(ckpt)
if '2nd' not in ckpt.lower() and 'counterfactual' not in ckpt.lower():
    model = UniterForVisualCommonsenseReasoning.from_pretrained('config/uniter-base.json', checkpoint, img_dim=2048)
    model.init_type_embedding()
else:
    ## 2nd stage pretrained
    model = UniterForVisualCommonsenseReasoning.from_pretrained('config/uniter-base_vcr.json', checkpoint, img_dim=2048)
    # if 'pretrained' in ckpt:
    #     model.init_word_embedding(81)
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
cls_sum = 0
kl_sum = 0
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
            # qar_scores = []
            # for batch_id in range(scores.shape[0]):
            #     answer_ind = qa_targets[batch_id].item()
            #     qar_index = [4+answer_ind*4+i
            #                  for i in range(4)]
            #     qar_scores.append(scores[batch_id, qar_index])
            # qar_scores = torch.stack(qar_scores, dim=0)
            qar_scores = scores[:, 4:8] - scores[:, 8:]
        else:
            qar_scores = scores[:, 4:]

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

kl = torch.nn.KLDivLoss(reduction='batchmean')

with tqdm(total=num_train_steps) as pbar:
    for epoch in range(100):
        for i, batch in enumerate(train_dataloader):
            with amp.autocast():
                scores = model(batch, compute_loss=False, return_full_score=True) # b,2
                qa_targets = batch['qa_targets'].cuda()
                qar_targets = batch['qar_targets'].cuda()
                targets = batch['targets'].cuda() # b, 1
                qids = batch['qids']

                scores_for_loss = scores.view(len(qids), 12, 2) # q, 12, 2
                targets = targets.view(len(qids), -1) #  q, 12

                vcr_loss = F.cross_entropy(scores_for_loss[:, :8, :].reshape(-1, 2), targets[:, :8].reshape(-1), reduction="none")
                vcr_qar_bias_loss = kl(scores_for_loss[:, 8:, :].reshape(-1, 2).log_softmax(dim=-1), 
                                       scores_for_loss[:, 4:8, :].reshape(-1, 2).softmax(dim=-1))

                scores = scores[:, 1:]
                scores = scores.view(len(qids), -1) # q, 12
                outputs_qa = scores[:, :4].max(dim=-1)[1]
                outputs_qar = scores[:, 4:8].max(dim=-1)[1]
                matched_qa = outputs_qa.squeeze() == qa_targets.squeeze()
                matched_qar = outputs_qar.squeeze() == qar_targets.squeeze()
                matched_joined = matched_qa & matched_qar

                vcr_loss = vcr_loss.view(len(qids), -1)
                vcr_loss[matched_joined == 0] *= 2
                vcr_loss = vcr_loss.mean()

                loss = vcr_loss + vcr_qar_bias_loss
                loss /= accum_steps

            scaler.scale(loss).backward()
            loss_sum += loss.item()
            cls_sum += vcr_loss.item()
            kl_sum += vcr_qar_bias_loss.item()
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

            writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], current_steps)
            writer.add_scalar("train/total_loss", loss_sum, current_steps)
            writer.add_scalar("train/cls_loss", cls_sum/accum_steps, current_steps)
            writer.add_scalar("train/kl_loss", kl_sum/accum_steps, current_steps)

            loss_sum = 0
            cls_sum = 0
            kl_sum = 0

            writer.flush()

            # validation & model save
            if current_steps % valid_steps == 0:
                validate(model, val_dataloader)
                torch.save(model.state_dict(), f'./ckpt/{current_time}/{ckpt_short}_{current_steps}_{batch_size}_{accum_steps}_{learning_rate}')

            if current_steps == num_train_steps:
                breakValue = True
                break

        if breakValue:
            break