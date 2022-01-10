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

annotPATH = '/mnt3/user16/vcr/vcr1annots/'
imagePATH = '/mnt3/user16/vcr/vcr1images/'
# annotPATH = '/home/vcr/vcr1annots/'
# imagePATH = '/home/vcr/vcr1images/'

"""
순서를 이렇게 바꿔서도 한번 해보자
        [[txt, img1],
         [txt, img2]]
        """

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

# class ConcatDatasetWithLens(ConcatDataset):
#     """ A thin wrapper on pytorch concat dataset for lens batching """
#     def __init__(self, datasets):
#         super().__init__(datasets)
#         self.lens = [l for dset in datasets for l in len(dset)]

#     def __getattr__(self, name):
#         return self._run_method_on_all_dsets(name)

#     def _run_method_on_all_dsets(self, name):
#         def run_all(*args, **kwargs):
#             return [dset.__getattribute__(name)(*args, **kwargs)
#                     for dset in self.datasets]
#         return run_all

def random_word(tokens, vocab_range=(999, 30522), mask=103):
    """
    Masking some random tokens for Language Model task with probabilities as in
        the original BERT paper.
    :param tokens: list of int, tokenized sentence.
    :param vocab_range: for choosing a random word
    :return: (list of int, list of int), masked tokens and related labels for
        LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = mask

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(range(*vocab_range)))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            output_label.append(token)
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
    if all(o == -1 for o in output_label):
        # at least mask 1
        output_label[0] = tokens[0]
        tokens[0] = mask

    return tokens, output_label


def _get_img_mask(mask_prob, num_bb):
    img_mask = [random.random() < mask_prob for _ in range(num_bb)]
    if not any(img_mask):
        # at least mask 1
        img_mask[random.choice(range(num_bb))] = True
    img_mask = torch.tensor(img_mask)
    return img_mask

def _get_img_tgt_mask(img_mask, txt_len):
    z = torch.zeros(txt_len, dtype=torch.float32)
    img_mask_tgt = torch.cat([z, img_mask], dim=0)
    return img_mask_tgt

def _get_feat_target(img_feat, img_masks):
    img_masks_ext = img_masks.unsqueeze(-1).expand_as(img_feat)  # (n, m, d)
    feat_dim = img_feat.size(-1)
    feat_targets = img_feat[img_masks_ext].contiguous().view(
        -1, feat_dim)  # (s, d)
    return feat_targets

def _mask_img_feat(img_feat, img_masks):
    img_masks_ext = img_masks.unsqueeze(-1).expand_as(img_feat)
    img_feat_masked = img_feat.data.masked_fill(img_masks_ext, 0)
    return img_feat_masked

def _get_targets(img_masks, img_soft_label):
    soft_label_dim = img_soft_label.size(-1)
    img_masks_ext_for_label = img_masks.unsqueeze(-1).expand_as(img_soft_label)
    label_targets = img_soft_label[img_masks_ext_for_label].contiguous().view(
        -1, soft_label_dim)
    return label_targets

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


## Here to start
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
        
        # masking for mlm
        masked_token_q, txt_label_q = random_word(question['input_ids'][1:-1])
        masked_token_a, txt_label_a = random_word(answer['input_ids'][1:-1])
        masked_token_r, txt_label_r = random_word(rationale['input_ids'][1:-1])
        
        maksed_tokenzied = [101] + masked_token_q + [102] + masked_token_a + [102] + masked_token_r + [102]
        txt_label = [-1] + txt_label_q + [-1] + txt_label_a + [-1] + txt_label_r + [-1]

        #preprocessing
        tokenzied = question['input_ids'] + answer['input_ids'][1:] + rationale['input_ids'][1:]
        
        token_type_ids = [0 for _ in range(len(question['input_ids']))] + [2 for _ in range(len(answer['input_ids'][1:]))] + [3 for _ in range(len(rationale['input_ids'][1:]))]

        # with open(imagePATH + self.data.metadata_fn[index], 'r') as f:
        #     meta = json.load(f)
        with open(imagePATH + self.data.metadata_fn[index][:-5] + '.pickle', 'rb') as f:
            feature = pickle.load(f)
        
        roi_feature = feature['roi_features']
        softlabel = feature['softlabels']
        nb = roi_feature.shape[1]
        attention_mask = [1 for _ in range(nb)] + question['attention_mask'] + answer['attention_mask'][1:] + rationale['attention_mask'][1:]

        width = feature['normalized_boxes'][:,:,2] - feature['normalized_boxes'][:,:,0]
        height = feature['normalized_boxes'][:,:,3] - feature['normalized_boxes'][:,:,1]
        a = width * height

        pos = torch.cat((feature['normalized_boxes'], width.unsqueeze(-1)), dim=-1)
        pos = torch.cat((pos, height.unsqueeze(-1)), dim=-1)
        pos = torch.cat((pos, a.unsqueeze(-1)), dim=-1)

        
        # for mrfr, mrc
        img_mask = _get_img_mask(0.15, nb)
        img_mask_tgt = _get_img_tgt_mask(img_mask, len(tokenzied))
        #print(roi_feature.squeeze(0).type(),pos.squeeze(0).type(),softlabel.squeeze(0).type(),img_mask.type(),img_mask_tgt.type())
        
        return (torch.Tensor(tokenzied),
                torch.Tensor(token_type_ids),
                torch.Tensor(attention_mask),
                roi_feature.squeeze(0),
                pos.squeeze(0),
                softlabel.squeeze(0),
                torch.Tensor(maksed_tokenzied),
                torch.Tensor(txt_label),
                img_mask.float(),
                img_mask_tgt.float())


def collate(batch):
    (input_ids, txt_type_ids, attn_masks, img_feat, img_pos, softlabel, maksed_tokenzied, txt_label, img_mask, img_mask_tgt) = map(list, unzip(batch))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    txt_type_ids = pad_sequence(txt_type_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)

    maksed_tokenzied = pad_sequence(maksed_tokenzied, batch_first=True, padding_value=0)
    txt_label = pad_sequence(txt_label, batch_first=True, padding_value=-1)
    
    num_bbs = [f.size(0) for f in img_feat]
    softlabel = pad_tensors(softlabel, num_bbs)
    img_feat = pad_tensors(img_feat, num_bbs)
    img_pos = pad_tensors(img_pos, num_bbs)
    img_mask = pad_sequence(img_mask, batch_first=True, padding_value=0)
    img_mask = img_mask.bool()
    label_targets = _get_targets(img_mask, softlabel)
    feat_target = _get_feat_target(img_feat, img_mask)
    img_mask_tgt = pad_sequence(img_mask_tgt, batch_first=True, padding_value=0)

    masked_img_feat = _mask_img_feat(img_feat, img_mask)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

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
             'masked_input_ids': maksed_tokenzied.long(),
             'txt_labels': txt_label.long(),
             'label_targets': label_targets,
             'masked_img_feat': masked_img_feat.float(),
             'feat_targets': feat_target.float(),
             'img_mask_tgt': img_mask_tgt.long(),
             'img_masks': img_mask}

    return batch


# dataset = PretrainDataForVCR(data_type='val')
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate)

# for i, batch in enumerate(dataloader):
#     print('New Batch! ', i)
#     print(batch)
#     if i == 2: break

### for finetuning
'''
batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        txt_type_ids = batch['txt_type_ids']
        targets = batch['targets']

input_ids_q, token_type_q
input_ids_a, token_type_a
input_ids_r, token_type_r

q + a1 + a2 + a3 + a4, 0 + 2
q + a + r1 + r2 + r3 + r4, 0 + 2 + 3

train: qa 하고 qar
validation qa, qar 동시에, but 형식은 같음
'''

class FinetuneDataForVCR(Dataset):
    def __init__(self, data_type='train', task='qa'):
        super().__init__()
        self.data = pd.read_json(path_or_buf=annotPATH + data_type + '.jsonl', lines=True)
        self.tokenzier = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenzier.max_length = 220
        self.task = task

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        #preprocessing for image
        with open(imagePATH + self.data.metadata_fn[index][:-5] + '.pickle', 'rb') as f:
            feature = pickle.load(f)
        
        roi_feature = feature['roi_features']
        nb = roi_feature.shape[1]

        width = feature['normalized_boxes'][:,:,2] - feature['normalized_boxes'][:,:,0]
        height = feature['normalized_boxes'][:,:,3] - feature['normalized_boxes'][:,:,1]
        a = width * height

        pos = torch.cat((feature['normalized_boxes'], width.unsqueeze(-1)), dim=-1)
        pos = torch.cat((pos, height.unsqueeze(-1)), dim=-1)
        pos = torch.cat((pos, a.unsqueeze(-1)), dim=-1)

        question = self.tokenzier(self.data.question_orig[index], return_token_type_ids=False)
        if self.task == 'qa':
            # qa
            answer_index = self.data.answer_sources[index]
            out = []
            for i in answer_index:
                answer = self.tokenzier(self.data.answer_orig[i], return_token_type_ids=False)
                tmp = copy.deepcopy(question)
                tokenized = tmp['input_ids'] + answer['input_ids'][1:]
                token_type_ids = [0 for _ in range(len(tmp['input_ids']))] + [2 for _ in range(len(answer['input_ids'][1:]))]
                attention_mask = [1 for _ in range(nb)] + tmp['attention_mask'] + answer['attention_mask'][1:]

                if index == i:
                    target = torch.Tensor([1]).long()
                else:
                    target = torch.Tensor([0]).long()
                out.append((torch.Tensor(tokenized), torch.Tensor(token_type_ids),
                            roi_feature.squeeze(0), pos.squeeze(0),
                            torch.Tensor(attention_mask), target))
        else:
            # qar
            rationale_index = self.data.rationale_sources[index]
            answer =  self.tokenzier(self.data.answer_orig[index], return_token_type_ids=False)
            out = []
            for i in rationale_index:
                rationale = self.tokenzier(self.data.rationale_orig[i], return_token_type_ids=False)
                tmp = copy.deepcopy(question)
                tmp_a = copy.deepcopy(answer)
                tokenized = tmp['input_ids'] + tmp_a['input_ids'][1:] + rationale['input_ids'][1:]
                token_type_ids = [0 for _ in range(len(tmp['input_ids']))] + [2 for _ in range(len(tmp_a['input_ids'][1:]))] + [3 for _ in range(len(rationale['input_ids'][1:]))]
                attention_mask = [1 for _ in range(nb)] + tmp['attention_mask'] + tmp_a['attention_mask'][1:] + rationale['attention_mask'][1:]

                if index == i:
                    target = torch.Tensor([1])
                else:
                    target = torch.Tensor([0])
                out.append((torch.Tensor(tokenized), torch.Tensor(token_type_ids),
                            roi_feature.squeeze(0), pos.squeeze(0),
                            torch.Tensor(attention_mask), target))
        
        return tuple(out)

def vcr_collate(inputs):
    (input_ids, txt_type_ids, img_feat,
     img_pos, attn_masks, targets) = map(list, unzip(concat(inputs)))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    txt_type_ids = pad_sequence(txt_type_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)
    
    num_bbs = [f.size(0) for f in img_feat]
    img_feat = pad_tensors(img_feat, num_bbs)
    img_pos = pad_tensors(img_pos, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    targets = torch.stack(targets, dim=0)

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
             'targets': targets.long()}

    return batch
    
def list_to_str(text_list):
    for i, ele in enumerate(text_list):
        if type(ele) == type([]):
            tmp = ''
            for j, e in enumerate(ele):
                tmp += str(e)
                if  j == 0 and len(ele) > 2:
                    tmp += ' , '
                else:
                    if (j+1) != len(ele):
                        tmp += ' and '
            text_list[i] =tmp 
    # print(" ".join(text_list))
    return " ".join(text_list)

class ValidationDataForVCR(Dataset):
    def __init__(self, data_type='val'):
        super().__init__()
        self.data = pd.read_json(path_or_buf=annotPATH + data_type + '.jsonl', lines=True)
        self.tokenzier = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenzier.max_length = 220
        self.data_type = data_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # preprocessing for Image
        qid = index
        qa_target = torch.Tensor([self.data.answer_label[index]])
        qar_target = torch.Tensor([self.data.rationale_label[index]])
        with open(imagePATH + self.data.metadata_fn[index][:-5] + '.pickle', 'rb') as f:
            feature = pickle.load(f)
        
        roi_feature = feature['roi_features']
        nb = roi_feature.shape[1]

        width = feature['normalized_boxes'][:,:,2] - feature['normalized_boxes'][:,:,0]
        height = feature['normalized_boxes'][:,:,3] - feature['normalized_boxes'][:,:,1]
        a = width * height

        pos = torch.cat((feature['normalized_boxes'], width.unsqueeze(-1)), dim=-1)
        pos = torch.cat((pos, height.unsqueeze(-1)), dim=-1)
        pos = torch.cat((pos, a.unsqueeze(-1)), dim=-1)
        
        out = []
        # qa
        if 'gd' not in self.data_type:
            question = self.tokenzier(self.data.question_orig[index], return_token_type_ids=False)
            answer_index = self.data.answer_sources[index]
            for i in answer_index:
                answer = self.tokenzier(self.data.answer_orig[i], return_token_type_ids=False)
                tmp = copy.deepcopy(question)
                tokenized = tmp['input_ids'] + answer['input_ids'][1:]
                token_type_ids = [0 for _ in range(len(tmp['input_ids']))] + [2 for _ in range(len(answer['input_ids'][1:]))]
                attention_mask = [1 for _ in range(nb)] + tmp['attention_mask'] + answer['attention_mask'][1:]

                out.append((torch.Tensor(tokenized), torch.Tensor(token_type_ids),
                            roi_feature.squeeze(0), pos.squeeze(0),
                            torch.Tensor(attention_mask)))

            rationale_index = self.data.rationale_sources[index]
            answer =  self.tokenzier(self.data.answer_orig[index], return_token_type_ids=False)
            for i in rationale_index:
                rationale = self.tokenzier(self.data.rationale_orig[i], return_token_type_ids=False)
                tmp = copy.deepcopy(question)
                tmp_a = copy.deepcopy(answer)
                tokenized = tmp['input_ids'] + tmp_a['input_ids'][1:] + rationale['input_ids'][1:]
                token_type_ids = [0 for _ in range(len(tmp['input_ids']))] + [2 for _ in range(len(tmp_a['input_ids'][1:]))] + [3 for _ in range(len(rationale['input_ids'][1:]))]
                attention_mask = [1 for _ in range(nb)] + tmp['attention_mask'] + tmp_a['attention_mask'][1:] + rationale['attention_mask'][1:]

                out.append((torch.Tensor(tokenized), torch.Tensor(token_type_ids),
                            roi_feature.squeeze(0), pos.squeeze(0),
                            torch.Tensor(attention_mask)))
        else:
            question = self.tokenzier(list_to_str(self.data.question[index]), return_token_type_ids=False)
            answer_choices = self.data.answer_choices[index]
            for i, answer_choice in enumerate(answer_choices):
                answer = self.tokenzier(list_to_str(answer_choice), return_token_type_ids=False)
                tmp = copy.deepcopy(question)
                tokenized = tmp['input_ids'] + answer['input_ids'][1:]
                token_type_ids = [0 for _ in range(len(tmp['input_ids']))] + [2 for _ in range(len(answer['input_ids'][1:]))]
                attention_mask = [1 for _ in range(nb)] + tmp['attention_mask'] + answer['attention_mask'][1:]

                out.append((torch.Tensor(tokenized), torch.Tensor(token_type_ids),
                            roi_feature.squeeze(0), pos.squeeze(0),
                            torch.Tensor(attention_mask)))
            
            rationale_choices = self.data.rationale_choices[index]
            answer =  self.tokenzier(list_to_str(self.data.answer_choices[index][self.data.answer_label[index]]), return_token_type_ids=False)
            for i, rationale_choice in enumerate(rationale_choices):
                rationale = self.tokenzier(list_to_str(rationale_choice), return_token_type_ids=False)
                tmp = copy.deepcopy(question)
                tmp_a = copy.deepcopy(answer)
                tokenized = tmp['input_ids'] + tmp_a['input_ids'][1:] + rationale['input_ids'][1:]
                token_type_ids = [0 for _ in range(len(tmp['input_ids']))] + [2 for _ in range(len(tmp_a['input_ids'][1:]))] + [3 for _ in range(len(rationale['input_ids'][1:]))]
                attention_mask = [1 for _ in range(nb)] + tmp['attention_mask'] + tmp_a['attention_mask'][1:] + rationale['attention_mask'][1:]

                out.append((torch.Tensor(tokenized), torch.Tensor(token_type_ids),
                            roi_feature.squeeze(0), pos.squeeze(0),
                            torch.Tensor(attention_mask)))

        return tuple(out), qid, qa_target, qar_target


def vcr_val_collate(inputs):
    (input_ids, txt_type_ids, img_feat, img_pos, attn_masks) = map(
        list, unzip(concat(outs for outs, _, _, _ in inputs)))

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

    # print([t for _, _, t, _ in inputs])
    qa_targets = torch.stack(
        [t for _, _, t, _ in inputs], dim=0)
    qar_targets = torch.stack(
        [t for _, _, _, t in inputs], dim=0)
    qids = [id_ for _, id_, _, _ in inputs]

    batch = {'qids': qids,
             'input_ids': input_ids.long(),
             'txt_type_ids': txt_type_ids.long(),
             'position_ids': position_ids.long(),
             'img_feat': img_feat.float(),
             'img_pos_feat': img_pos.float(),
             'attn_masks': attn_masks.long(),
             'gather_index': gather_index.long(),
             'qa_targets': qa_targets.long(),
             'qar_targets': qar_targets.long()}

    return batch