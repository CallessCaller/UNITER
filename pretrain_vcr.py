import torch
import torch.nn.functional as F
import torch.cuda.amp as amp
import torch.cuda.amp.autocast_mode

from model.pretrain_vcr import UniterForPretrainingForVCR
from prepro import PretrainDataForVCR, DataLoader, collate
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
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--accum_steps', type=int, default=4)
args = parser.parse_args()

warmup_steps = 4500
accum_steps = args.accum_steps
valid_steps = 3000
num_train_steps = 45000
batch_size = args.batch_size #6144
val_batch_size = batch_size
learning_rate = 3e-05

import time
writer = SummaryWriter(f"./log/{batch_size}_{accum_steps}_{time.time()}")

# dataloader
print('Loading dataset...')
train_dataset = PretrainDataForVCR(data_type='train')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=10)
val_dataset = PretrainDataForVCR(data_type='val')
val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True, collate_fn=collate, num_workers=10)
# train_dataset = PretrainDataForVCR(data_type='val')
# train_dataloader = DataLoader(train_dataset, batch_size=val_batch_size, shuffle=True, collate_fn=collate,
#                               num_workers=5)
print('Done !!!')

# model
print('Loading model...')
checkpoint = torch.load('pretrained/uniter-base.pt')
model = UniterForPretrainingForVCR.from_pretrained('config/uniter-base.json', checkpoint, img_dim=2048, img_label_dim=1601)
# only for when we use 1st pretrained ckpt
# model.config.vocab_size = 30522
# model.config.type_vocab_size = 4
# model.init_type_embedding()
# model.init_word_embedding(81)
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


current_steps = 0
breakValue = False
mlm_steps = -1
mrc_steps = -1
mrfr_steps = -1

def compute_accuracy_for_soft_targets(out, labels):
    outputs = out.max(dim=-1)[1]
    labels = labels.max(dim=-1)[1]  # argmax
    n_correct = (outputs == labels).sum().item()
    return n_correct

@torch.no_grad()
def validate(model, val_dataloader):
        print('Start running validation...')
        model.eval()
        mlm_loss = 0
        n_correct = 0
        n_word = 0

        mrc_loss = 0
        n_feat_mrc = 0
        tot_score = 0

        mrfr_loss = 0
        n_feat_mrfr = 0
        n_feat = 0
        for i, batch in enumerate(tqdm(val_dataloader)):
                batch['txt_labels'] = batch['txt_labels'].cuda()
                batch['label_targets'] = batch['label_targets'].cuda()
                batch['img_mask_tgt'] = batch['img_mask_tgt'].cuda()
                #mlm
                scores = model(batch, task='mlm', compute_loss=False)
                labels = batch['txt_labels']
                labels = labels[labels != -1]
                loss =  F.cross_entropy(scores, labels, reduction='sum')
                mlm_loss += loss.item()
                n_correct += (scores.max(dim=-1)[1] == labels).sum().item()
                n_word += labels.numel()

                #mrc
                prediction_soft_label = model(batch, task='mrc', compute_loss=False)
                prediction_soft_label = F.log_softmax(prediction_soft_label, dim=-1)
                label_targets = batch['label_targets']
                #prediction_soft_label = prediction_soft_label[:, :-1]
                loss = F.kl_div(prediction_soft_label, label_targets, reduction='sum')
                tot_score += compute_accuracy_for_soft_targets(prediction_soft_label, label_targets)
               
                mrc_loss += loss.item()
                n_feat += batch['img_mask_tgt'].sum().item()
                
                #mrfr
                loss = model(batch, task='mrfr', compute_loss=True)
                mrfr_loss += loss.sum().item() / 2048

        #mlm
        mlm_loss /= n_word
        acc = n_correct / n_word
        print(f"MLM Val Loss: {mlm_loss} | MLM Val ACC: {acc*100}%")
        writer.add_scalar("val_mlm_loss", mlm_loss, current_steps)
        writer.add_scalar("val_mlm_acc", acc*100, current_steps)

        #mrc
        mrc_loss /= n_feat
        val_acc = tot_score / n_feat
        print(f"MRC Val Loss: {mrc_loss} | MLM Val ACC: {val_acc*100}%")
        writer.add_scalar("val_mrc_loss", mrc_loss, current_steps)
        writer.add_scalar("val_mrc_acc", val_acc*100, current_steps)

        #mrfr
        mrfr_loss /= n_feat
        print(f"MRFR Val Loss: {mrfr_loss}")
        writer.add_scalar("val_mrfr_loss", mrfr_loss, current_steps)
        model.train()

model.train()
with tqdm(total=num_train_steps) as pbar:
        for epoch in range(100):
                #validate(model, val_dataloader)
                #print(f"Epoch: {epoch} Current_step: {current_steps}|")
                for i, batch in enumerate(train_dataloader):
                        task_prob = torch.rand(1)
                        if task_prob > 0.66:
                                task = 'mlm'
                                batch['input_ids'] = batch['masked_input_ids']
                                mlm_steps += 1
                        elif task_prob > 0.33:
                                task = 'mrc'
                                batch['img_feat'] = batch['masked_img_feat']
                                mrc_steps += 1
                        else:
                                task = 'mrfr'
                                batch['img_feat'] = batch['masked_img_feat']
                                mrfr_steps += 1

                        with amp.autocast():
                                loss = model(batch, task=task, compute_loss=True)
                                if task != 'mrc':
                                        loss = loss.mean()

                        scaler.scale(loss).backward()
                        loss_sum += loss.item()
                        accum += 1

                        if task == 'mlm':
                                writer.add_scalar("loss_mlm", loss.item(), current_steps)
                        elif task == 'mrc':
                                writer.add_scalar("loss_mrc", loss.item(), current_steps)
                        else:
                                writer.add_scalar("loss_mrfr", loss.item(), current_steps)

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

                        # validation % model save
                        if (current_steps + 1) % valid_steps == 0:
                                validate(model, val_dataloader)
                                torch.save(model.state_dict(), f'./ckpt/UNITER_2nd_{current_steps + 1}_{batch_size}_{accum_steps}')
                        writer.flush()

                        if (current_steps + 1) == num_train_steps:
                                validate(model, val_dataloader)
                                torch.save(model.state_dict(), f'./ckpt/UNITER_2nd_{current_steps + 1}_{batch_size}_{accum_steps}')
                                breakValue = True
                                writer.flush()
                                break
                if breakValue:
                        print(f"Num steps per task ==> MLM: {mlm_steps} | MRC: {mrc_steps} MRFR: {mrfr_steps}")
                        break


@torch.no_grad()
def validate_mlm(model, val_loader):
        print('Start running MLM validation...')
        val_loss = 0
        n_correct = 0
        n_word = 0
        for i, batch in enumerate(val_loader):
                scores = model(batch, task='mlm', compute_loss=False)
                labels = batch['txt_labels'].cuda()
                labels = labels[labels != -1]
                loss =  F.cross_entropy(scores, labels, reduction='sum')
                val_loss += loss.item()
                n_correct += (scores.max(dim=-1)[1] == labels).sum().item()
                n_word += labels.numel()
        val_loss /= n_word
        acc = n_correct / n_word
        print(f"MLM Val Loss: {val_loss} | MLM Val ACC: {acc*100}%")
        writer.add_scalar("val_mlm_loss", val_loss, current_steps)
        writer.add_scalar("val_mlm_acc", acc*100, current_steps)

@torch.no_grad()
def validate_mrc(model, val_loader):
        print('Start running MRC validation...')
        val_loss = 0
        n_feat = 0
        tot_score = 0
        for i, batch in enumerate(val_loader):
                prediction_soft_label = model(batch, task='mrc', compute_loss=False)

                prediction_soft_label = F.log_softmax(
                        prediction_soft_label, dim=-1)
                label_targets = batch['label_targets']
                loss = F.kl_div(prediction_soft_label, label_targets, reduction='sum')
                tot_score += compute_accuracy_for_soft_targets(prediction_soft_label, label_targets)
               
                val_loss += loss.item()
                n_feat += batch['img_mask_tgt'].sum().item()
        val_loss /= n_feat
        val_acc = tot_score / n_feat
        print(f"MRC Val Loss: {val_loss} | MLM Val ACC: {val_acc*100}%")
        writer.add_scalar("val_mrc_loss", val_loss, current_steps)
        writer.add_scalar("val_mrc_acc", val_acc*100, current_steps)

@torch.no_grad()
def validate_mrfr(model, val_loader):
        print('Start running MRFR validation...')
        val_loss = 0
        n_feat = 0
        for i, batch in enumerate(val_loader):
                loss = model(batch, task='mrfr', compute_loss=True)
                val_loss += loss.sum().item() / 2048
                n_feat += batch['img_mask_tgt'].cuda().sum().item()
        val_loss /= n_feat
        print(f"MRFR Val Loss: {val_loss}")
        writer.add_scalar("val_mrfr_loss", val_loss, current_steps)


