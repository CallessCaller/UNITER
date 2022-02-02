from collections import defaultdict
import io
import json

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm
import lmdb

import msgpack
import msgpack_numpy
msgpack_numpy.patch()

a = lmdb.open('/mnt3/user16/vcr/vcr1uniter/img_db/vcr_train/feat_th0.2_max100_min10/', readonly=True, create=False)
b = a.begin(buffers=True)
c = b.get('vcr_train_MqMD1DK0CTQ@0.npz'.encode('utf-8'))
d = msgpack.loads(c, raw=False)

print(d)
print(d.keys())