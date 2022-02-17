import torch
import torch.nn.functional as F
import torch.cuda.amp as amp
import torch.cuda.amp.autocast_mode
from torch.utils.data import DataLoader, ConcatDataset

from model.vcr import UniterForVisualCausalReasoning
from prepro_counterfactual import PretrainDataForVCR, vcr_collate
from transformers import AdamW, get_linear_schedule_with_warmup

from tqdm import tqdm
import argparse

from torch.utils.tensorboard import SummaryWriter

# random seed
torch.random.manual_seed(42) # 42, 29837, 854769803

# config 
parser = argparse.ArgumentParser(description='Config')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--accum_step', type=int, default=32)
parser.add_argument('--train_step', type=int, default=20000)
parser.add_argument('--val_step', type=int, default=1000)
parser.add_argument('--lr', type=float, default=6e-5)
parser.add_argument('--ckpt', type=str, default='pretrained/uniter-base.pt')
parser.add_argument('--eval_only', type=str, default=False)
args = parser.parse_args()


batch_size = args.batch_size #4000
accum_steps = args.accum_step
num_train_steps = args.train_step
ckpt = args.ckpt
warmup_steps = num_train_steps // 10
valid_steps = args.val_step #1000 if num_train_steps / 10 < 1000 else num_train_steps /10
val_batch_size = 2
learning_rate = args.lr

if args.eval_only:
    valid_steps = 1

print('Loading dataset...')
train_dataset = PretrainDataForVCR(data_type='train')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=vcr_collate, num_workers=10)

val_dataset = PretrainDataForVCR(data_type='val')
val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=vcr_collate, num_workers=10)
print('Done !!!')

# model
print('Loading model...')
checkpoint = torch.load(ckpt)
model = UniterForVisualCausalReasoning.from_pretrained('config/uniter-base.json', checkpoint, img_dim=2048)
model.init_type_embedding()
model.cuda()
model.train()
print('Done !!!')

import time
import os
current_time = time.localtime()
current_time = time.strftime('%c', current_time)
if args.eval_only == False:
    os.mkdir(f'ckpt/counterfactual_{current_time}')
    writer = SummaryWriter(f"./log_counterfactual/{batch_size}_{accum_steps}_{learning_rate}_{current_time}")
else:
    writer = SummaryWriter(f"./test/{batch_size}_{accum_steps}_{learning_rate}_{current_time}")

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
    print('Start validation...')
    model.eval()

    right_total = 0
    factual_total = 0
    counterfactual_total = 0
    n_sample = 0
    n_negative = 0

    loss_total = 0
    loss_f = 0
    loss_c = 0
    for i, batch in enumerate(tqdm(val_loader)):
        scores = model(batch, compute_loss=False)
        targets = batch['targets'].view(-1).cuda()
        factual_labels = batch['counterfactual_mask'].view(-1).cuda()
        loss = F.cross_entropy(scores, targets.squeeze(-1), reduction='none')

        prediction = torch.argmax(scores, dim=-1) # 64, 1
        right = prediction == targets
        factual = prediction[factual_labels==1] == targets[factual_labels==1]
        counterfactual = prediction[factual_labels==0] == targets[factual_labels==0]

        right_total += right.sum().item()
        factual_total += factual.sum().item()
        counterfactual_total += counterfactual.sum().item()

        n_sample += factual_labels.sum().item()
        n_negative += targets.shape[0] - factual_labels.sum().item()

        loss_total += loss.sum().item()
        loss_f += loss[factual_labels==1].sum().item()
        loss_c += loss[factual_labels==0].sum().item()

    right_total /= (n_sample + n_negative)
    factual_total /= n_sample
    counterfactual_total /= n_negative

    loss_total /= (n_sample + n_negative)
    loss_f /= n_sample
    loss_c /= n_negative

    writer.add_scalar("cf_valid/loss_total", loss_total, current_steps)
    writer.add_scalar("cf_valid/loss_f", loss_f, current_steps)
    writer.add_scalar("cf_valid/loss_c", loss_c, current_steps)
    writer.add_scalar("cf_valid/acc_total", right_total*100, current_steps)
    writer.add_scalar("cf_valid/acc_f", factual_total*100, current_steps)
    writer.add_scalar("cf_valid/acc_c", counterfactual_total*100, current_steps)

    print(f"Loss_total: {loss_total:.4f} | Loss_c: {loss_f:.4f} | Loss_w: {loss_c:.4f}")
    print(f"Score_total: {right_total:.2f} | Score_c: {factual_total:.2f} | Score_w: {counterfactual_total:.2f}")
    writer.flush()
    model.train()
    return


factual_total = 0
counterfactual_total = 0
optimizer.zero_grad()
with tqdm(total=num_train_steps) as pbar:
    for epoch in range(100):
        print(f'Epoch: {epoch}')
        for i, batch in enumerate(train_dataloader):
            factual_labels = batch['counterfactual_mask'].view(-1).cuda()
            with amp.autocast():
                loss = model(batch, compute_loss=True)
                f_loss = loss[factual_labels==1]
                c_loss = loss[factual_labels==0]

                f_loss = f_loss.mean()
                c_loss = c_loss.mean()

                total_loss = f_loss + max(0, 0.5-c_loss) 

            scaler.scale(total_loss).backward()
            loss_sum += total_loss.item()
            factual_total += f_loss.item()
            counterfactual_total += c_loss.item()
            accum += 1

            if accum != accum_steps:
                continue

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            accum = 0
            current_steps += 1
            pbar.update(1)

            # if current_steps % 50 == 0:
            writer.add_scalar("cf_train/lr", optimizer.param_groups[0]['lr'], current_steps)
            writer.add_scalar("cf_train/loss", loss_sum/accum_steps, current_steps)
            writer.add_scalar("cf_train/loss_factual", factual_total/accum_steps, current_steps)
            writer.add_scalar("cf_train/loss_counterfactual", counterfactual_total/accum_steps, current_steps)
            writer.flush()

            loss_sum = 0
            factual_total = 0
            counterfactual_total = 0

            # validation & model save
            if current_steps % valid_steps == 0:
                validate(model, val_dataloader)
                torch.save(model.state_dict(), f'ckpt/counterfactual_{current_time}/{current_steps}_{batch_size}_{accum_steps}_{learning_rate}')

            if current_steps == num_train_steps:
                breakValue = True
                break

        if breakValue:
            break
        