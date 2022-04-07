import torch
import torch.nn.functional as F
import torch.cuda.amp.autocast_mode
from torch.utils.data import DataLoader

from model.vcr import UniterForVisualCommonsenseReasoning
from prepro_ripe import ValidationDataForVCR, vcr_val_collate
from tqdm import tqdm

import argparse

# random seed
torch.random.manual_seed(42)

# config 
parser = argparse.ArgumentParser(description='Config')
parser.add_argument('--ckpt', type=str, default='ckpt/uniter-base_5000_16_64_0.0001')
parser.add_argument('--data_type', type=str, default='val')
args = parser.parse_args()

val_batch_size = 16
ckpt = args.ckpt

print('Loading dataset...')
val_dataset = ValidationDataForVCR(data_type=args.data_type)
val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=vcr_val_collate, num_workers=10)
print('Done !!!')

# model
print('Loading model...')
checkpoint = torch.load(ckpt)
if 'pretrain' in ckpt:
    model = UniterForVisualCommonsenseReasoning.from_pretrained('config/uniter-base.json', checkpoint, img_dim=2048)
    model.init_type_embedding()
else:
    ## 2nd stage pretrained
    model = UniterForVisualCommonsenseReasoning.from_pretrained('config/uniter-base_vcr.json', checkpoint, img_dim=2048)
model.cuda()
model.train()
print('Done !!!')

current_steps = 0


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
    qa_prediction = []
    qa_label = []
    qar_prediction = []
    qar_label = []
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
        
        qa_prediction += torch.argmax(scores[:, :4], dim=-1).tolist()
        qa_label += qa_targets.view(-1).tolist()
        qar_prediction += torch.argmax(scores[:, 4:], dim=-1).tolist()
        qar_label += qar_targets.view(-1).tolist()
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

    print(f"Score_qa: {val_qa_acc*100:.2f} | Score_qar: {val_qar_acc*100:.2f} | Score_total: {val_acc*100:.2f}")
    
    return qa_prediction, qa_label, qar_prediction, qar_label

qa_prediction, qa_label, qar_prediction, qar_label = validate(model, val_dataloader)
