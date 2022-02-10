import enum
import torch
import torch.nn as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import BertTokenizer

import json
import numpy as np
import pandas as pd
import pickle
import random
from toolz.sandbox import unzip
from cytoolz import concat
import copy
import lmdb

import msgpack
import msgpack_numpy
msgpack_numpy.patch()

annotPATH = '/mnt3/user16/vcr/vcr1annots/'
imagePATH = '/mnt3/user16/vcr/vcr1images/'

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokenizer.max_length = 220

# 이미지, counterfactual 이미지, 의도, counterfactual, event, intent, before, after (input_ids)

### For Pretraining
class PretrainDataForVCR(Dataset):
    def __init__(self, data_type='train'):
        super().__init__()
        self.data = pd.read_json(path_or_buf=annotPATH + data_type + '.jsonl', lines=True)
        self.data_type = data_type
        self.db = lmdb.open(f'/mnt3/user16/vcr/vcr1uniter/img_db/vcr_{data_type}/feat_th0.2_max100_min10/', readonly=True, create=False)
        self.db_begin = self.db.begin(buffers=True)

        self.db_gt = lmdb.open(f'/mnt3/user16/vcr/vcr1uniter/img_db/vcr_gt_{data_type}/feat_numbb100/', readonly=True, create=False)
        self.db_gt_begin = self.db_gt.begin(buffers=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        event = tokenizer(self.data[index]['event'], return_token_type_ids=False)
        intent = ','.join(self.data[index]['intent'])
        befores = self.data[index]['before']
        afters = self.data[index]['after']

        fname = self.data[index]['metadata_fn'].split('/')[-1][:-5]
        img = self.db_begin.get(f'vcr_{self.data_type}_{fname}.npz'.encode('utf-8'))
        img_msg = msgpack.loads(img, raw=False)
        #survivor = np.reshape(img['conf'] > 0.2, (-1))
        features = torch.Tensor(img_msg['features']).float()
        bbs = torch.Tensor(img_msg['norm_bb']).float()
        img_bb = torch.cat([bbs, bbs[:, 4:5]*bbs[:, 5:]], dim=-1)

        img_gt = self.db_gt_begin.get(f'vcr_gt_{self.data_type}_{fname}.npz'.encode('utf-8'))
        img_gt_msg = msgpack.loads(img_gt, raw=False)
        features_gt = torch.Tensor(img_gt_msg['features']).float()
        bbs_gt = torch.Tensor(img_gt_msg['norm_bb']).float()
        img_bb_gt = torch.cat([bbs_gt, bbs_gt[:, 4:5]*bbs_gt[:, 5:]], dim=-1)

        nb = features_gt.shape[0] + features.shape[0]
        roi_feature = torch.cat([features_gt, features], dim=0)
        pos = torch.cat([img_bb_gt, img_bb], dim=0)

        # counterfactual cf
        cf_index = index
        while cf_index == index:
            cf_index = np.random.randint(len(self.data)-1)
        cf_intent = ','.join(self.data[cf_index]['intent'])

        cf_fname = self.data[cf_index]['metadata_fn'].split('/')[-1][:-5]
        img_cf = self.db_begin.get(f'vcr_{self.data_type}_{cf_fname}.npz'.encode('utf-8'))
        img_msg_cf = msgpack.loads(img_cf, raw=False)
        #survivor = np.reshape(img['conf'] > 0.2, (-1))
        features_cf = torch.Tensor(img_msg_cf['features']).float()
        bbs_cf = torch.Tensor(img_msg_cf['norm_bb']).float()
        img_bb_cf = torch.cat([bbs_cf, bbs_cf[:, 4:5]*bbs_cf[:, 5:]], dim=-1)

        img_gt_cf = self.db_gt_begin.get(f'vcr_gt_{self.data_type}_{cf_fname}.npz'.encode('utf-8'))
        img_gt_msg_cf = msgpack.loads(img_gt_cf, raw=False)
        features_gt_cf = torch.Tensor(img_gt_msg_cf['features']).float()
        bbs_gt_cf = torch.Tensor(img_gt_msg_cf['norm_bb']).float()
        img_bb_gt_cf = torch.cat([bbs_gt_cf, bbs_gt_cf[:, 4:5]*bbs_gt_cf[:, 5:]], dim=-1)

        nb_cf = features_gt_cf.shape[0] + features_cf.shape[0]
        roi_feature_cf = torch.cat([features_gt_cf, features_cf], dim=0)
        pos_cf = torch.cat([img_bb_gt_cf, img_bb_cf], dim=0)

        out = []
        intent =  tokenizer(intent, return_token_type_ids=False)
        cf_intent = tokenizer(cf_intent, return_token_type_ids=False)

        # event
        tmp_intent = copy.deepcopy(intent)
        tmp_cf_intent = copy.deepcopy(cf_intent)
        tokenized = tmp_intent['input_ids'] + event['input_ids'][1:] 
        token_type_ids = [0 for _ in range(len(tmp_intent['input_ids']))] + [1 for _ in range(len(event['input_ids'][1:]))] 
        attention_mask = [1 for _ in range(nb)] + tmp_intent['attention_mask'] + event['attention_mask'][1:]
        attention_mask_cf = [1 for _ in range(nb_cf)] + tmp_intent['attention_mask'] + event['attention_mask'][1:] 
        out.append((torch.Tensor(tokenized), torch.Tensor(token_type_ids), 
                    roi_feature, pos,  
                    torch.Tensor(attention_mask), torch.Tensor([1]), torch.Tensor([True])))
        out.append((torch.Tensor(tokenized), torch.Tensor(token_type_ids), 
                    roi_feature_cf, pos_cf,  
                    torch.Tensor(attention_mask_cf), torch.Tensor([1]), torch.Tensor([False])))

        tokenized_cf = tmp_cf_intent['input_ids'] + event['input_ids'][1:]
        token_type_ids_cf = [0 for _ in range(len(tmp_cf_intent['input_ids']))] + [1 for _ in range(len(event['input_ids'][1:]))]
        attention_mask_cf = [1 for _ in range(nb)] + tmp_cf_intent['attention_mask'] + event['attention_mask'][1:]
        out.append((torch.Tensor(tokenized_cf), torch.Tensor(token_type_ids_cf), 
                    roi_feature, pos,  
                    torch.Tensor(attention_mask_cf), torch.Tensor([1]), torch.Tensor([False])))

        # before after
        label = -2
        for inferences in [befores, afters]:
            label += 2
            for i, inference in enumerate(inferences):
                inference = tokenizer(inference, return_token_type_ids=False)
                tmp_intent = copy.deepcopy(intent)
                tmp_cf_intent = copy.deepcopy(cf_intent)
                tokenized = tmp_intent['input_ids'] + inference['input_ids'][1:] 
                token_type_ids = [0 for _ in range(len(tmp_intent['input_ids']))] + [1 for _ in range(len(inference['input_ids'][1:]))] 
                attention_mask = [1 for _ in range(nb)] + tmp_intent['attention_mask'] + inference['attention_mask'][1:]
                attention_mask_cf = [1 for _ in range(nb_cf)] + tmp_intent['attention_mask'] + inference['attention_mask'][1:] 
                out.append((torch.Tensor(tokenized), torch.Tensor(token_type_ids), 
                            roi_feature, pos,  
                            torch.Tensor(attention_mask), torch.Tensor([label]), torch.Tensor([True])))
                out.append((torch.Tensor(tokenized), torch.Tensor(token_type_ids), 
                            roi_feature_cf, pos_cf,  
                            torch.Tensor(attention_mask_cf), torch.Tensor([label]), torch.Tensor([False])))

                tokenized_cf = tmp_cf_intent['input_ids'] + inference['input_ids'][1:]
                token_type_ids_cf = [0 for _ in range(len(tmp_cf_intent['input_ids']))] + [1 for _ in range(len(inference['input_ids'][1:]))]
                attention_mask_cf = [1 for _ in range(nb)] + tmp_cf_intent['attention_mask'] + inference['attention_mask'][1:]
                out.append((torch.Tensor(tokenized_cf), torch.Tensor(token_type_ids_cf), 
                            roi_feature, pos,  
                            torch.Tensor(attention_mask_cf), torch.Tensor([label]), torch.Tensor([False])))

        
        return tuple(out)

def vcr_collate(inputs):
    (input_ids, txt_type_ids, img_feat,
     img_pos, attn_masks, targets, counterfactual_mask) = map(list, unzip(concat(inputs)))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    txt_type_ids = pad_sequence(txt_type_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)
    
    num_bbs = [f.size(0) for f in img_feat]
    img_feat = pad_tensors(img_feat, num_bbs)
    img_pos = pad_tensors(img_pos, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    targets = torch.stack(targets, dim=0)
    counterfactual_mask = torch.stack(counterfactual_mask, dim=0)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.shape[1]
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids.long(),
             'txt_type_ids': txt_type_ids.long(),
             'position_ids': position_ids.long(),
             'img_feat': img_feat.float(),
             'img_pos_feat': img_pos.float(),
             'attn_masks': attn_masks.long(),
             'gather_index': gather_index.long(),
             'targets': targets.long(),
             'counterfactual_mask': counterfactual_mask}

    return batch

'''
F.kl_div(F.log_softmax(k, 0), F.softmax(k1, 0), reduction="none").mean()
'''

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