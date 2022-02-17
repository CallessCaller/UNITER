import pickle
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

with open('results/uniter.pickle', 'rb') as f:
    qa_prediction = np.array(pickle.load(f))
    qa_label = np.array(pickle.load(f))
    qar_prediction = np.array(pickle.load(f))
    qar_label = np.array(pickle.load(f))

with open('results/counterfactual.pickle', 'rb') as f:
    cqa_prediction = np.array(pickle.load(f))
    cqa_label = np.array(pickle.load(f))
    cqar_prediction = np.array(pickle.load(f))
    cqar_label = np.array(pickle.load(f))

def find_both_correct(c1, c2):
    tmp = []
    for i, ele in enumerate(c1):
        if ele and c2[i]:
            tmp.append(True)
        else:
            tmp.append(False)
    return np.array(tmp)

def find_different(c1, c2):
    tmp = []
    for i, ele in enumerate(c1):
        if (ele or c2[i]) and ele != c2[i]:
            tmp.append(True)
        else:
            tmp.append(False)
    return np.array(tmp)

def save_array_index(array, path):
    tmp = []
    for i, ele in enumerate(array):
        if ele:
            tmp.append(i)
    with open(path, 'wb') as f:
        pickle.dump(tmp, f)
    
    return tmp

qa_correct = qa_prediction == qa_label
qar_correct = qar_prediction == qar_label
all_correct = find_both_correct(qa_correct, qar_correct)

cqa_correct = cqa_prediction == cqa_label
cqar_correct = cqar_prediction == cqar_label
call_correct = find_both_correct(cqa_correct, cqar_correct)

both_qa = find_both_correct(qa_correct, cqa_correct)
both_qar = find_both_correct(qar_correct, cqar_correct)
both_correct = find_both_correct(all_correct, call_correct)
both_wrong = find_both_correct(all_correct == False, call_correct == False)

print(both_qa.sum(), len(qa_label), qa_correct.sum(), cqa_correct.sum())
print(both_qar.sum(), len(qar_label), qar_correct.sum(), cqar_correct.sum())
print(both_correct.sum(), len(qa_label), all_correct.sum(), call_correct.sum())

only_o_correct = find_different(both_correct, all_correct)
only_c_correct = find_different(both_correct, call_correct)

only_o_correct_qa = find_different(both_qa, qa_correct)
only_c_correct_qa = find_different(both_qa, cqa_correct)

only_o_correct_qar = find_different(both_qar, qar_correct)
only_c_correct_qar = find_different(both_qar, cqar_correct)


print(only_o_correct.sum(), only_c_correct.sum())
print(only_o_correct_qa.sum(), only_c_correct_qa.sum())
print(only_o_correct_qar.sum(), only_c_correct_qar.sum())

save_array_index(both_correct, 'results/both_correct.pickle')
save_array_index(both_qa, 'results/both_qa.pickle')
save_array_index(both_qar, 'results/both_qar.pickle')
save_array_index(both_wrong, 'results/both_wrong.pickle')

save_array_index(only_o_correct, 'results/only_o_both.pickle')
save_array_index(only_c_correct, 'results/only_c_both.pickle')
save_array_index(only_o_correct_qa, 'results/only_o_qa.pickle')
save_array_index(only_c_correct_qa, 'results/only_c_qa.pickle')
save_array_index(only_o_correct_qar, 'results/only_o_qar.pickle')
save_array_index(only_c_correct_qar, 'results/only_c_qar.pickle')