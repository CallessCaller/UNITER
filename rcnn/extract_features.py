from processing_image import Preprocess
from modeling_frcnn import GeneralizedRCNN
from utils_rcnn import Config
import json
import pandas as pd
import pickle
from tqdm import tqdm

annotPATH = '/mnt3/user16/vcr/vcr1annots/'
imagePATH = '/mnt3/user16/vcr/vcr1images/'
#featurePATH = 'home/vcr/vcr1features/'

frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
frcnn_cfg.min_detections = 10
frcnn_cfg.max_detections = 100
frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
image_preprocess = Preprocess(frcnn_cfg)

file_list = ['gd_val']

for i in range(len(file_list)):
    data = pd.read_json(path_or_buf=annotPATH + file_list[i] + '.jsonl', lines=True)
    for j in tqdm(range(len(data))):
        metadata_fn = data.metadata_fn[j]
        img_fn = data.img_fn[j]
        
        with open(imagePATH + metadata_fn, 'r') as f:
            meta = json.load(f)

        images, sizes, scales_yx = image_preprocess(imagePATH + img_fn)

        gt_boxes = []
        for i, boxes in enumerate(meta['boxes']):
            tmp = [boxes[0]/scales_yx[0][0], boxes[1]/scales_yx[0][1], boxes[2]/scales_yx[0][0], boxes[3]/scales_yx[0][1]]
            gt_boxes.append(tmp)

        output_dict = frcnn(
            images.cuda(),
            sizes,
            scales_yx=scales_yx,
            #gt_boxes=gt_boxes,
            max_detections=frcnn_cfg.max_detections,
            location='cuda:0',
            return_tensors="pt",
        )

        with open(imagePATH + metadata_fn[:-5] + '_no_gt.pickle', 'wb') as f:
            pickle.dump(output_dict, f)
