from processing_image import Preprocess
from modeling_frcnn import GeneralizedRCNN
from utils_rcnn import Config

import json
with open('/home/vcr/vcr1images/movieclips_Yours_Mine_and_Ours/vFSAQ1Nj7fg@0.json', 'r') as f:
    a = json.load(f)

# img_path = ['/home/vcr/vcr1images/movieclips_Yours_Mine_and_Ours/vFSAQ1Nj7fg@11.jpg', '/home/vcr/vcr1images/movieclips_Zodiac/DSuUJ-Scbeg@12.jpg', 
#             '/home/vcr/vcr1images/movieclips_Zodiac/RiTXscx2pJY@15.jpg', '/home/vcr/vcr1images/movieclips_Zodiac/RiTXscx2pJY@18.jpg']
img_path = ['/home/vcr/vcr1images/movieclips_Yours_Mine_and_Ours/vFSAQ1Nj7fg@11.jpg']

frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
frcnn_cfg.min_detections = 10
frcnn_cfg.max_detections = 100
frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
#frcnn.cuda()
image_preprocess = Preprocess(frcnn_cfg)
images, sizes, scales_yx = image_preprocess(img_path)

gt_boxes = []
for i, boxes in enumerate(a['boxes']):
    tmp = [boxes[0]/scales_yx[0][0], boxes[1]/scales_yx[0][1], boxes[2]/scales_yx[0][0], boxes[3]/scales_yx[0][1]]
    gt_boxes.append(tmp)

output_dict = frcnn(
    images.cuda(),
    sizes,
    scales_yx=scales_yx,
    gt_boxes=gt_boxes,
    max_detections=frcnn_cfg.max_detections,
    location='cuda:0',
    return_tensors="pt",
)

print(output_dict['softlabels'][0].shape)
print('hello')

# output_dict['roi_features'].shape = [2, 36, 2048]

# vcr: x1,y1,x2,y2
# this repo: x1, y1, x2, y2

# """
# kwargs:
#     max_detections (int), return_tensors {"np", "pt", None}, padding {None,
#     "max_detections"}, pad_value (int), location = {"cuda", "cpu"}
# """