import torch
import torch.nn as F
from model.model import UniterModel
from transformers import BertTokenizer

checkpoint = torch.load('ckpt/pretrained/uniter-base.pt')
model = UniterModel.from_pretrained('config/uniter-base.json', checkpoint, img_dim=2048)
model.cuda()
model.train()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sentence = tokenizer('Hello my name is Park. what is your [MASK] ?', return_tensors='pt')

out = model(input_ids=sentence['input_ids'].cuda(), attention_mask=sentence['attention_mask'].cuda(), txt_type_ids=sentence['token_type_ids'].cuda(),  img_feat=None, img_pos_feat=None, position_ids=sentence['token_type_ids'].cuda())