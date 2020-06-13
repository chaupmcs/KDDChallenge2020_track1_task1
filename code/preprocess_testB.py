import pandas as pd 
import numpy as np
import json
import base64
import swifter
from tqdm import tqdm
import csv
import pickle
from sklearn.externals import joblib
import gc
from time import sleep


TRAIN_PATH = '../data/train.tsv'
VAL_PATH = '../data/valid.tsv'
VAL_ANS_PATH = '../data/valid_answer.json'
SAMPLE_PATH = '../data/train.sample.tsv'
LABEL_PATH = '../data/multimodal_labels.txt'
TEST_PATH = '../data/testB.tsv'

def get_label(path):
    with open(path) as f:
        lines = f.readlines()
        label2id = {l.split('\n')[0].split('\t')[1]:int(l.split('\n')[0].split('\t')[0]) for l in lines[1:]}
        id2label = {int(l.split('\n')[0].split('\t')[0]):l.split('\n')[0].split('\t')[1] for l in lines[1:]}
    return label2id, id2label

label2id, id2label = get_label(LABEL_PATH)

print(id2label, label2id)

def convertBoxes(num_boxes, boxes):
    res = None
    try:
        res = np.frombuffer(base64.b64decode(boxes), dtype=np.float32).reshape(num_boxes, 4)
    except:
        pass
    return res

def convertFeature(num_boxes, features):
    res = None
    try:
        res = np.frombuffer(base64.b64decode(features), dtype=np.float32).reshape(num_boxes, 2048)
    except:
        pass
    return res

def convertLabel(num_boxes, label):
    res = None
    try:
        res = np.frombuffer(base64.b64decode(label), dtype=np.int64).reshape(num_boxes)
    except:
        pass
    return res

def convertLabelWord(num_boxes, label):
    tmp= None
    try:
        tmp = np.frombuffer(base64.b64decode(label), dtype=np.int64).reshape(num_boxes)
    except:
        pass
    return '###'.join([id2label[t] for t in tmp])


def convertPos(num_boxes, boxes, H, W):
    pos_list = []
    try:
        for i in range(num_boxes):
            temp = boxes[i]
            pos_list.append([temp[0]/W, 
                             temp[2]/W, 
                             temp[1]/H, 
                             temp[3]/H, 
                             ((temp[2] - temp[0]) * (temp[3] - temp[1]))/ (W*H),])
    except:
        pass
    return pos_list



test = pd.read_csv(TEST_PATH,sep='\t')
test['boxes_convert'] = test.swifter.apply(lambda x: convertBoxes(x['num_boxes'], x['boxes']), axis=1)
test['feature_convert'] = test.swifter.apply(lambda x: convertFeature(x['num_boxes'], x['features']), axis=1)
test['labels_convert'] = test.swifter.apply(lambda x: convertLabel(x['num_boxes'], x['class_labels']), axis=1)
test['label_words'] = test.swifter.apply(lambda x: convertLabelWord(x['num_boxes'], x['class_labels']), axis=1)
test['pos'] = test.swifter.apply(lambda x: convertPos(x['num_boxes'], x['boxes_convert'], x['image_h'], x['image_w']), axis=1)
del test['boxes'], test['features'], test['class_labels']
with open('../data/test_data.pkl', 'wb') as outp:
    pickle.dump(test, outp)
print("test data finish")
