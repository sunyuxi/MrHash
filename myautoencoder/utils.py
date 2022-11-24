# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
import torch
from torch.utils.data import Dataset
import sys
import numpy as np
import pickle
from PIL import Image
import torchvision
import time

################################
# dataset load
################################
#The name of load_alexnet_feature is ugly. It can load different types of features, not just alexnet features.
def load_alexnet_feature(input_file, class_cnt=10, start=0, end=26000):

    with open(input_file, 'rb') as fo:
        dict = pickle.load(fo)
    data = dict['data']
    print(len(data))
    print(len(data[0]))
    labels = np.array(dict['labels']).astype(np.int32)
    labels = np.eye(class_cnt)[labels].astype(np.int8)

    return data[start:end, ], labels[start:end, ]


class DenseFeatureTwoDataset(Dataset):

    def __init__(self, input_file1, input_file2, class_cnt, start, end):
        self.x1_list, self.y1_list = load_alexnet_feature(input_file1, class_cnt=class_cnt, start=start, end=end)
        self.x2_list, self.y2_list = load_alexnet_feature(input_file2, class_cnt=class_cnt, start=start, end=end)
        assert self.x1_list.shape[0] == self.x2_list.shape[0]
        assert self.y1_list.all() == self.y2_list.all()

    def __len__(self):
        return len(self.x1_list)

    def __getitem__(self, idx):
        return self.x1_list[idx], self.x2_list[idx], self.y1_list[idx], torch.from_numpy(np.array(idx))

#######################################################
# Evaluate Critiron
#######################################################
def sanitize(x):
    return np.ascontiguousarray(x, dtype='float32')

#mean_average_precision: https://github.com/yuanli2333/Hadamard-Matrix-for-hashing/blob/master/test.py
def mean_average_precision(sim, database_labels, test_labels, R, T):  # R = 1000
    start = time.time()
    query_num = test_labels.shape[0]  # total number for testing
    ids = np.argsort(-sim, axis=0)  
    end = time.time()
    print('dot time: ' + str(end - start), flush=True)
    start = time.time()
    
    APx = []
    Recall = []

    for i in range(query_num): 
        label = test_labels[i, :]  # test labels
        if np.sum(label) == 0:
            assert(False)

        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_labels[idx[0:R], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R + 1, 1)  #

        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)
        if relevant_num == 0:  # even no relevant image, still need add in APx for calculating the mean
            APx.append(0)

        all_relevant = np.sum(database_labels == label, axis=1) > 0
        all_num = np.sum(all_relevant)
        r = relevant_num / np.float(all_num)
        Recall.append(r)

    end = time.time()
    print('dot time: ' + str(end - start), flush=True)
    
    return np.mean(np.array(APx)), np.mean(np.array(Recall)), APx
