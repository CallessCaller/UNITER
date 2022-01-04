import torch
import torch.nn as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers.models.bert import tokenization_bert
from model.pretrain_vcr import UniterForPretrainingForVCR
from transformers import BertTokenizer

import json
import numpy as np
import pandas as pd
import pickle
from toolz.sandbox import unzip

annotPATH = '/home/vcr/vcr1annots/'
imagePATH = '/home/vcr/vcr1images/'

'''
def forward(self, batch, task, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']          o
        position_ids = batch['position_ids']    x
        img_feat = batch['img_feat']            o 
        img_pos_feat = batch['img_pos_feat']    o
        attention_mask = batch['attn_masks']    o
        gather_index = batch['gather_index']    x
        txt_type_ids = batch['txt_type_ids']    o

         """
        Return:
        :input_ids    (n, max_L) padded with 0
        :position_ids (n, max_L) padded with 0
        :txt_lens     list of [txt_len]
        :img_feat     (n, max_num_bb, feat_dim)
        :img_pos_feat (n, max_num_bb, 7)
        :num_bbs      list of [num_bb]
        :attn_masks   (n, max_{L + num_bb}) padded with 0
        :txt_labels   (n, max_L) padded with -1
        """
'''

'''
['movie', 'objects', 'interesting_scores', 'answer_likelihood', 'img_fn',
'metadata_fn', 'answer_orig', 'question_orig', 'rationale_orig',
'question', 'answer_match_iter', 'answer_sources', 'answer_choices',
'answer_label', 'rationale_choices', 'rationale_sources',
'rationale_match_iter', 'rationale_label', 'img_id', 'question_number',
'annot_id', 'match_fold', 'match_index']

['obj_ids', 'obj_probs', 'attr_ids', 'attr_probs', 'boxes', 'sizes', 'preds_per_image', 'roi_features', 'normalized_boxes']

dict_keys(['boxes', 'segms', 'names', 'width', 'height'])
box is x1y1 x2y2 format
uniter needs xyxywha format
'''

'''
F.kl_div(F.log_softmax(k, 0), F.softmax(k1, 0), reduction="none").mean()
'''


class PretrainDataForVCR(Dataset):
    def __init__(self, data_type='train'):
        super().__init__()
        self.data = pd.read_json(path_or_buf=annotPATH + data_type + '.jsonl', lines=True)
        self.tokenzier = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenzier.max_length = 220

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        question = self.tokenzier(self.data.question_orig[index], return_token_type_ids=False)
        answer = self.tokenzier(self.data.answer_orig[index], return_token_type_ids=False)
        rationale = self.tokenzier(self.data.rationale_orig[index], return_token_type_ids=False)
        # type_id
        # 0 -- question
        # 1 -- region
        # 2 -- answer
        # 3 -- rationale
        tokenzied = question['input_ids'] + answer['input_ids'][1:] + rationale['input_ids'][1:]
        
        token_type_ids = [0 for _ in range(len(question['input_ids']))] + [2 for _ in range(len(answer['input_ids'][1:]))] + [3 for _ in range(len(rationale['input_ids'][1:]))]

        # with open(imagePATH + self.data.metadata_fn[index], 'r') as f:
        #     meta = json.load(f)
        with open(imagePATH + self.data.metadata_fn[index][:-5] + '.pickle', 'rb') as f:
            feature = pickle.load(f)
        
        roi_feature = feature['roi_features']
        nb = roi_feature.shape[1]
        attention_mask = [1 for _ in range(nb)] + question['attention_mask'] + answer['attention_mask'][1:] + rationale['attention_mask'][1:]

        width = feature['normalized_boxes'][:,:,2] - feature['normalized_boxes'][:,:,0]
        height = feature['normalized_boxes'][:,:,3] - feature['normalized_boxes'][:,:,1]
        a = width * height

        pos = torch.cat((feature['normalized_boxes'], width.unsqueeze(-1)), dim=-1)
        pos = torch.cat((pos, height.unsqueeze(-1)), dim=-1)
        pos = torch.cat((pos, a.unsqueeze(-1)), dim=-1)

        return (torch.FloatTensor(tokenzied),
                torch.FloatTensor(token_type_ids),
                torch.FloatTensor(attention_mask),
                torch.FloatTensor(roi_feature).squeeze(0),
                torch.FloatTensor(pos).squeeze(0))

def pad_tensors(tensors, lens=None, pad=0):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    bs = len(tensors)
    hid = tensors[0].size(-1)
    dtype = tensors[0].dtype
    output = torch.zeros(bs, max_len, hid, dtype=dtype)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output

def get_gather_index(txt_lens, num_bbs, batch_size, max_len, out_size):
    assert len(txt_lens) == len(num_bbs) == batch_size
    gather_index = torch.arange(0, out_size, dtype=torch.long,
                                ).unsqueeze(0).repeat(batch_size, 1)

    for i, (tl, nbb) in enumerate(zip(txt_lens, num_bbs)):
        gather_index.data[i, tl:tl+nbb] = torch.arange(max_len, max_len+nbb,
                                                       dtype=torch.long).data
    return gather_index


def collate(batch):
    (input_ids, txt_type_ids, attn_masks, img_feat, img_pos) = map(list, unzip(batch))
    for i in range(3):
        print(img_feat[i].shape)
        print(img_pos[i].shape)
        print(input_ids[i].shape)

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    txt_type_ids = pad_sequence(txt_type_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)
    
    num_bbs = [f.size(0) for f in img_feat]
    img_feat = pad_tensors(img_feat, num_bbs)
    img_pos = pad_tensors(img_pos, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.shape[1]

    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)
    print(gather_index)

    batch = {'input_ids': input_ids,
             'txt_type_ids': txt_type_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos,
             'attn_masks': attn_masks,
             'gather_index': gather_index}

    return batch


dataset = PretrainDataForVCR(data_type='val')
dataloader = DataLoader(dataset, batch_size=3, shuffle=True, collate_fn=collate)

for i, batch in enumerate(dataloader):
    print('New Batch! ', i)
    #print(batch)
    if i == 2: break
