# -*- coding: utf-8 -*-
#

import numpy as np
import torch
from torch.utils.data import Dataset
import sys

import pickle
from PIL import Image
import torchvision

################################
# dataset load
################################

def load_dataset(file_path, is_train = False):
    x_list = []
    y_list = []

    cnt = 0
    for one in open(file_path):
        arr = one.strip().split()
        x = arr[0]
        y = [ int(i) for i in arr[1:] ]

        x_list.append(x)
        y_list.append(y)

        cnt += 1
        if cnt%4000==0:
            print('load_dataset:' + str(cnt), flush=True)
        #    break

    return x_list, np.array(y_list)

def l2normalize_matrix(v):
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    norm[norm<=0.000001] = 1

    return v / norm 
    
def load_singleimg(img_path, is_train=True):

    if is_train:
        data_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            #torchvision.transforms.CenterCrop(224),
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        data_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    imgdata = Image.open(img_path)
    x = data_transforms(imgdata)
    return x

def load_similarity_matrix(input_file):
    #[0,1]
    similarity_matrix_ori = np.load(input_file)
   
    max_val = np.max(similarity_matrix_ori)
    min_val = np.min(similarity_matrix_ori)
   
    return 2*(similarity_matrix_ori-min_val)/(max_val-min_val)-1

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

class RSDataset(Dataset):

    def __init__(self, img_filepath, input_file1, input_file2, class_cnt, start, end, is_train=False):
        self.is_train = is_train
        self.img_filepath=img_filepath
        self.x_list, self.y_list = load_dataset(self.img_filepath, is_train=is_train)
        self.x1_list, self.y1_list = load_alexnet_feature(input_file1, class_cnt=class_cnt, start=start, end=end)
        self.x2_list, self.y2_list = load_alexnet_feature(input_file2, class_cnt=class_cnt, start=start, end=end)
        assert self.x1_list.shape[0] == self.x2_list.shape[0]
        assert self.y1_list.all() == self.y2_list.all()
        assert self.y_list.all() == self.y1_list.all()
        
    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        return load_singleimg(self.x_list[idx], self.is_train), self.x1_list[idx], self.x2_list[idx], self.y_list[idx], torch.from_numpy(np.array(idx))
