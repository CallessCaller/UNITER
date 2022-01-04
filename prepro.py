import torch
import torch.nn as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from model.pretrain_vcr import UniterForPretrainingForVCR
from transformers import BertTokenizer

import json
import numpy as np
import pandas as pd
import pickle
from toolz.utils import unzip

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

        with open(imagePATH + self.data.metadata_fn[index], 'r') as f:
            meta = json.load(f)
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
                torch.FloatTensor(roi_feature),
                torch.FloatTensor(pos))

def collate(inputs):
    (input_ids, type_ids, att_mask, img_feat, img_pos) = map(list, unzip(input))


dataset = PretrainDataForVCR(data_type='val')
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

for i, batch in enumerate(dataloader):
    print('New Batch! ', i)
    print(batch)
    if i == 2: break
