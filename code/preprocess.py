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
TEST_PATH = '../data/testA.tsv'

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
    tmp = None
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

train = pd.read_csv(TRAIN_PATH,sep='\t', chunksize=30000,  quoting=csv.QUOTE_NONE)  #nrows=900000


i = 1
for t in tqdm(train):
    
    LEN = 0
    product_set = set()
    num_boxes_list = []
    image_h_list = []
    image_w_list = []
    words_len_list = []
    words_list = []
    label_list = []
    label_words_list = []
    boxes_list = []
    boxes_feature_list = []
    pos_list = []

    print("starting")
    gc.collect()
    # sleep(1)
    LEN += len(t)
    temp = list(t['query'])
    words_len_list.extend([len(q.split()) for q in temp])
    words_list.extend(temp)

    t['labels_convert_words'] = t.swifter.apply(lambda x: convertLabelWord(x['num_boxes'], x['class_labels']), axis=1)
    temp = list(t['labels_convert_words'])
    label_words_list.extend(temp)

    t['feature_convert'] = t.swifter.apply(lambda x: convertFeature(x['num_boxes'], x['features']), axis=1)
    temp = list(t['feature_convert'])
    boxes_feature_list.extend(temp)

    t['boxes_convert'] = t.swifter.apply(lambda x: convertBoxes(x['num_boxes'], x['boxes']), axis=1)
    temp = list(t['boxes_convert'])

    t['pos'] = t.swifter.apply(lambda x: convertPos(x['num_boxes'], x['boxes_convert'], x['image_h'], x['image_w']), axis=1)
    temp = list(t['pos'])
    pos_list.extend(temp)


    data = pd.DataFrame({
                         'words':words_list,
                         'label_words':label_words_list,
                         'features':boxes_feature_list,
                         'pos':pos_list,
                        })
  
    print(f"writing {i}...")
    with open('../data/temp_data_{}.pkl'.format(i), 'wb') as outp:
        joblib.dump(data, outp)

    del temp, data
    gc.collect()
    # sleep(60)

    print("Done i =", i)

    i += 1


    


    

print("temp data finish! Done creating training data!")
