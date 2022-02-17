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
# annotPATH = '/home/vcr/vcr1annots/'
# imagePATH = '/home/vcr/vcr1images/'

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokenizer.max_length = 220


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


-- VCR annotation format --
['movie', 'objects', 'interesting_scores', 'answer_likelihood', 'img_fn',
'metadata_fn', 'answer_orig', 'question_orig', 'rationale_orig',
'question', 'answer_match_iter', 'answer_sources', 'answer_choices',
'answer_label', 'rationale_choices', 'rationale_sources',
'rationale_match_iter', 'rationale_label', 'img_id', 'question_number',
'annot_id', 'match_fold', 'match_index']

-- VCR image format --
['obj_ids', 'obj_probs', 'attr_ids', 'attr_probs', 
'boxes', 'sizes', 'preds_per_image', 
'roi_features', 'normalized_boxes']

-- VCR metadata format --
dict_keys(['boxes', 'segms', 'names', 'width', 'height'])
box is x1y1 x2y2 format
uniter needs xyxywha format
'''

### For Pretraining
class PretrainDataForVCR(Dataset):
    def __init__(self, data_type='train'):
        super().__init__()
        self.data = pd.read_json(path_or_buf=annotPATH + data_type + '.jsonl', lines=True)
        self.data_type = data_type
        # tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        # tokenizer.max_length = 220
        self.db = lmdb.open(f'/mnt3/user16/vcr/vcr1uniter/img_db/vcr_{data_type}/feat_th0.2_max100_min10/', readonly=True, create=False)
        self.db_begin = self.db.begin(buffers=True)

        self.db_gt = lmdb.open(f'/mnt3/user16/vcr/vcr1uniter/img_db/vcr_gt_{data_type}/feat_numbb100/', readonly=True, create=False)
        self.db_gt_begin = self.db_gt.begin(buffers=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        with open(imagePATH + self.data.metadata_fn[index], 'r') as f:
            metadata = json.load(f)
        names = metadata['names']
        q_str, _ = list_to_str_only(self.data.question[index], names)
        a_str, _ = list_to_str_only(self.data.answer_choices[index][self.data.answer_label[index]], names)
        r_str, _ = list_to_str_only(self.data.rationale_choices[index][self.data.answer_label[index]], names)
        question = tokenizer(q_str, return_token_type_ids=False)
        answer = tokenizer(a_str, return_token_type_ids=False)
        rationale = tokenizer(r_str, return_token_type_ids=False)
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

        fname = self.data.metadata_fn[index].split('/')[-1][:-5]
        img = self.db_begin.get(f'vcr_{self.data_type}_{fname}.npz'.encode('utf-8'))
        img_msg = msgpack.loads(img, raw=False)
        #survivor = np.reshape(img['conf'] > 0.2, (-1))
        features = torch.Tensor(img_msg['features']).float()
        bbs = torch.Tensor(img_msg['norm_bb']).float()
        img_bb = torch.cat([bbs, bbs[:, 4:5]*bbs[:, 5:]], dim=-1)
        soft_labels = torch.Tensor(img_msg['soft_labels']).float()

        img_gt = self.db_gt_begin.get(f'vcr_gt_{self.data_type}_{fname}.npz'.encode('utf-8'))
        img_gt_msg = msgpack.loads(img_gt, raw=False)
        features_gt = torch.Tensor(img_gt_msg['features']).float()
        bbs_gt = torch.Tensor(img_gt_msg['norm_bb']).float()
        img_bb_gt = torch.cat([bbs_gt, bbs_gt[:, 4:5]*bbs_gt[:, 5:]], dim=-1)
        soft_labels_gt = torch.Tensor(img_gt_msg['soft_labels']).float()

        nb = features_gt.shape[0] + features.shape[0]
        roi_feature = torch.cat([features_gt, features], dim=0)
        softlabel = torch.cat([soft_labels_gt, soft_labels], dim=0)
        pos = torch.cat([img_bb_gt, img_bb], dim=0)

        attention_mask = [1 for _ in range(nb)] + question['attention_mask'] + answer['attention_mask'][1:] + rationale['attention_mask'][1:]

        # for mrfr, mrc
        img_mask = _get_img_mask(0.15, nb)
        img_mask_tgt = _get_img_tgt_mask(img_mask, len(tokenzied))
        
        return (torch.Tensor(tokenzied),
                torch.Tensor(token_type_ids),
                torch.Tensor(attention_mask),
                roi_feature,
                pos,
                softlabel,
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


### For Fintuning & Validation
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
        # tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        # tokenizer.max_length = 220
        self.task = task

        self.db = lmdb.open('/mnt3/user16/vcr/vcr1uniter/img_db/vcr_train/feat_th0.2_max100_min10/', readonly=True, create=False)
        self.db_begin = self.db.begin(buffers=True)

        self.db_gt = lmdb.open('/mnt3/user16/vcr/vcr1uniter/img_db/vcr_gt_train/feat_numbb100/', readonly=True, create=False)
        self.db_gt_begin = self.db_gt.begin(buffers=True)

    def __len__(self):
        return len(self.data)

    def __del__(self):
        self.db.close()
        self.db_gt.close()

    def __getitem__(self, index):
        #preprocessing for image
        fname = self.data.metadata_fn[index].split('/')[-1][:-5]
        img = self.db_begin.get(f'vcr_train_{fname}.npz'.encode('utf-8'))
        img_msg = msgpack.loads(img, raw=False)
        #survivor = np.reshape(img['conf'] > 0.2, (-1))
        features = torch.Tensor(img_msg['features']).float()
        bbs = torch.Tensor(img_msg['norm_bb']).float()
        img_bb = torch.cat([bbs, bbs[:, 4:5]*bbs[:, 5:]], dim=-1)
        #soft_labels = torch.Tensor(img['soft_labels']).float()

        img_gt = self.db_gt_begin.get(f'vcr_gt_train_{fname}.npz'.encode('utf-8'))
        img_gt_msg = msgpack.loads(img_gt, raw=False)
        features_gt = torch.Tensor(img_gt_msg['features']).float()
        bbs_gt = torch.Tensor(img_gt_msg['norm_bb']).float()
        img_bb_gt = torch.cat([bbs_gt, bbs_gt[:, 4:5]*bbs_gt[:, 5:]], dim=-1)
        #soft_labels_gt = torch.Tensor(img_gt['soft_labels']).float()

        nb = features_gt.shape[0] + features.shape[0]
        roi_feature = torch.cat([features_gt, features], dim=0)
        pos = torch.cat([img_bb_gt, img_bb], dim=0)

        with open(imagePATH + self.data.metadata_fn[index], 'r') as f:
            metadata = json.load(f)
        names = metadata['names']

        q_str, new_tokens = list_to_str_only(self.data.question[index], names)
        # tokenizer.add_tokens(new_tokens)
        question = tokenizer(q_str, return_token_type_ids=False, return_attention_mask=False)
        answer_choices = self.data.answer_choices[index]
        
        answer_label = self.data.answer_label[index]
        rationale_label = self.data.rationale_label[index]
        if self.task == 'qa':
            # qa
            out = []
            for i, answer_choice in enumerate(answer_choices):
                a_str, new_tokens = list_to_str_only(answer_choice, names)
                # tokenizer.add_tokens(new_tokens)
                answer = tokenizer(a_str, return_token_type_ids=False, return_attention_mask=False)
                tmp = copy.deepcopy(question)
                tokenized = tmp['input_ids'] + answer['input_ids'][1:]
                token_type_ids = [0 for _ in range(len(tmp['input_ids']))] + [2 for _ in range(len(answer['input_ids'][1:]))]
                if len(tokenized) > 220:
                    tokenized = tokenized[:219] + [tokenized[-1]]
                    token_type_ids = token_type_ids[:220]
                attention_mask = [1 for _ in range(nb + len(tokenized))]

                if i == answer_label:
                    target = torch.Tensor([1]).long()
                else:
                    target = torch.Tensor([0]).long()
                out.append((torch.Tensor(tokenized), torch.Tensor(token_type_ids),
                            roi_feature, pos,
                            torch.Tensor(attention_mask), target))
        else:
            # qar
            rationale_choices = self.data.rationale_choices[index]
            a_str, new_tokens = list_to_str_only(answer_choices[answer_label], names)
            # tokenizer.add_tokens(new_tokens)
            answer =  tokenizer(a_str, return_token_type_ids=False, return_attention_mask=False)
            out = []
            for i, rationale_choice in enumerate(rationale_choices):
                r_str, new_tokens = list_to_str_only(rationale_choice, names)
                # tokenizer.add_tokens(new_tokens)
                rationale = tokenizer(r_str, return_token_type_ids=False, return_attention_mask=False)
                tmp = copy.deepcopy(question)
                tmp_a = copy.deepcopy(answer)
                tokenized = tmp['input_ids'] + tmp_a['input_ids'][1:] + rationale['input_ids'][1:]
                token_type_ids = [0 for _ in range(len(tmp['input_ids']))] + [2 for _ in range(len(tmp_a['input_ids'][1:]))] + [3 for _ in range(len(rationale['input_ids'][1:]))]
                if len(tokenized) > 220:
                    tokenized = tokenized[:219] + [tokenized[-1]]
                    token_type_ids = token_type_ids[:220]
                attention_mask = [1 for _ in range(nb + len(tokenized))]

                if i == rationale_label:
                    target = torch.Tensor([1])
                else:
                    target = torch.Tensor([0])
                out.append((torch.Tensor(tokenized), torch.Tensor(token_type_ids),
                            roi_feature, pos,
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


class ValidationDataForVCR(Dataset):
    def __init__(self, data_type='val'):
        super().__init__()
        if 'custom' in data_type:
            with open(data_type, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.data = pd.read_json(path_or_buf=annotPATH + data_type + '.jsonl', lines=True)
        # tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        # tokenizer.max_length = 220
        self.data_type = data_type

        self.db = lmdb.open('/mnt3/user16/vcr/vcr1uniter/img_db/vcr_val/feat_th0.2_max100_min10/', readonly=True, create=False)
        self.db_begin = self.db.begin(buffers=True)

        self.db_gt = lmdb.open('/mnt3/user16/vcr/vcr1uniter/img_db/vcr_gt_val/feat_numbb100/', readonly=True, create=False)
        self.db_gt_begin = self.db_gt.begin(buffers=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # preprocessing for Image
        qid = index
        qa_target = torch.Tensor([self.data.answer_label[index]])
        qar_target = torch.Tensor([self.data.rationale_label[index]])

        fname = self.data.metadata_fn[index].split('/')[-1][:-5]
        img = self.db_begin.get(f'vcr_val_{fname}.npz'.encode('utf-8'))
        img_msg = msgpack.loads(img, raw=False)
        #survivor = np.reshape(img['conf'] > 0.2, (-1))
        features = torch.Tensor(img_msg['features']).float()
        bbs = torch.Tensor(img_msg['norm_bb']).float()
        img_bb = torch.cat([bbs, bbs[:, 4:5]*bbs[:, 5:]], dim=-1)
        #soft_labels = torch.Tensor(img['soft_labels']).float()

        img_gt = self.db_gt_begin.get(f'vcr_gt_val_{fname}.npz'.encode('utf-8'))
        img_gt_msg = msgpack.loads(img_gt, raw=False)
        features_gt = torch.Tensor(img_gt_msg['features']).float()
        bbs_gt = torch.Tensor(img_gt_msg['norm_bb']).float()
        img_bb_gt = torch.cat([bbs_gt, bbs_gt[:, 4:5]*bbs_gt[:, 5:]], dim=-1)
        #soft_labels_gt = torch.Tensor(img_gt['soft_labels']).float()

        nb = features_gt.shape[0] + features.shape[0]
        roi_feature = torch.cat([features_gt, features], dim=0)
        pos = torch.cat([img_bb_gt, img_bb], dim=0)

        with open(imagePATH + self.data.metadata_fn[index], 'r') as f:
            metadata = json.load(f)
        names = metadata['names']
        
        out = []

        q_str, new_tokens = list_to_str_only(self.data.question[index], names)
        # tokenizer.add_tokens(new_tokens)
        question = tokenizer(q_str, return_token_type_ids=False)
        answer_choices = self.data.answer_choices[index]

        answer_label = self.data.answer_label[index]

        for i, answer_choice in enumerate(answer_choices):
            a_str, new_tokens = list_to_str_only(answer_choice, names)
            # tokenizer.add_tokens(new_tokens)
            answer = tokenizer(a_str, return_token_type_ids=False)
            tmp = copy.deepcopy(question)
            tokenized = tmp['input_ids'] + answer['input_ids'][1:]
            token_type_ids = [0 for _ in range(len(tmp['input_ids']))] + [2 for _ in range(len(answer['input_ids'][1:]))]
            attention_mask = [1 for _ in range(nb)] + tmp['attention_mask'] + answer['attention_mask'][1:]

            out.append((torch.Tensor(tokenized), torch.Tensor(token_type_ids),
                        roi_feature, pos,
                        torch.Tensor(attention_mask)))

        rationale_choices = self.data.rationale_choices[index]
        a_str, new_tokens = list_to_str_only(answer_choices[answer_label], names)
        # tokenizer.add_tokens(new_tokens)
        answer =  tokenizer(a_str, return_token_type_ids=False)

        for i, rationale_choice in enumerate(rationale_choices):
            r_str, new_tokens = list_to_str_only(rationale_choice, names)
            # tokenizer.add_tokens(new_tokens)
            rationale = tokenizer(r_str, return_token_type_ids=False)
            tmp = copy.deepcopy(question)
            tmp_a = copy.deepcopy(answer)
            tokenized = tmp['input_ids'] + tmp_a['input_ids'][1:] + rationale['input_ids'][1:]
            token_type_ids = [0 for _ in range(len(tmp['input_ids']))] + [2 for _ in range(len(tmp_a['input_ids'][1:]))] + [3 for _ in range(len(rationale['input_ids'][1:]))]
            attention_mask = [1 for _ in range(nb)] + tmp['attention_mask'] + tmp_a['attention_mask'][1:] + rationale['attention_mask'][1:]

            out.append((torch.Tensor(tokenized), torch.Tensor(token_type_ids),
                        roi_feature, pos,
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


'''
F.kl_div(F.log_softmax(k, 0), F.softmax(k1, 0), reduction="none").mean()
'''

def list_to_str_only(text_list, name):
    new_tokens = []
    new_text = ''
    for i, ele in enumerate(text_list):
        if type(ele) == type([]):
            for e in ele:
                # if len(name)-1 < int(e):
                #     tmp = f'{e} '
                # else:
                #     tmp = name[int(e)] + f'_{e} '
                tmp = f'{e} '
                new_text += tmp
                new_tokens.append(tmp)
        else:
            new_text += ele
        new_text += ' '
    return new_text, new_tokens


# From UNITER repo
def random_word(tokens, vocab_range=(999, 28996), mask=103):
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