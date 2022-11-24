# -*- coding: utf-8 -*-
#

import torchvision
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

import os
import sys
import argparse
import numpy as np
import time
import random

from metrics import mean_average_precision
from dataset import RSDataset

def predict_hash_code(model, data_loader):  # data_loader is database_loader or test_loader
    model.eval()
    is_start = True
    for idx, (inputs, _, _, label, _) in enumerate(data_loader):
        inputs = Variable(inputs).cuda()
        label = Variable(label).cuda()
        output = model(inputs)

        if is_start:
            all_output = output.data.cpu().float()
            all_label = label.float()
            is_start = False
        else:
            all_output = torch.cat((all_output, output.data.cpu().float()), 0)
            all_label = torch.cat((all_label, label.float()), 0)

    return all_output.cpu().numpy(), all_label.cpu().numpy().astype(np.int8)

def gen_similarity_matrix(model, database_dataset, test_dataset, output_filename, output_hash_filename, args):
    database_loader = DataLoader(database_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print('start to model database', flush=True)
    start = time.time()
    database_hash, database_labels = predict_hash_code(model, database_loader)
    end = time.time()
    print('predict database time:'+str(end-start), flush=True)
    print(database_hash[0])
    print(database_labels[0])
    print(database_hash.shape)
    print(database_labels.shape)
    print('start to testset', flush=True)
    start = end
    test_hash, test_labels = predict_hash_code(model, test_loader)
    end = time.time()
    print('predict test time:'+str(end-start), flush=True)
    
    database_hash[database_hash>=0] = 1
    database_hash[database_hash<0] = -1

    test_hash[test_hash>=0] = 1
    test_hash[test_hash<0] = -1
    
    sim = np.dot(database_hash, test_hash.T)

    np.save(output_filename, sim)

    np_hashdata_all = np.concatenate( (np.array(database_hash).astype(np.int8), np.array(test_hash).astype(np.int8)), axis=0)
    np_hashlabel_all = np.concatenate( (np.array(database_labels).astype(np.int8), np.array(test_labels).astype(np.int8)), axis=0)
    y = np.array([np.where(one==1)[0][0] for one in np_hashlabel_all], dtype=np.int16)

    np.save(output_hash_filename, {'hash': np_hashdata_all, 'y': y})


def test_MAP(model, database_loader, test_loader, args):
    print('start to model database', flush=True)
    start = time.time()
    database_hash, database_labels = predict_hash_code(model, database_loader)
    end = time.time()
    print('predict database time:'+str(end-start), flush=True)
    print(database_hash[0])
    print(database_labels[0])
    print(database_hash.shape)
    print(database_labels.shape)
    print('start to testset', flush=True)
    start = end
    test_hash, test_labels = predict_hash_code(model, test_loader)
    end = time.time()
    print('predict test time:'+str(end-start), flush=True)
    print(test_hash[0])
    print(test_labels[0])
    print(test_hash.shape)
    print(test_labels.shape)
    print('Calculate MAP.....', flush=True)
    start = end

    #argsR_list = [args.R, 1000, 500, 100, 50]
    argsR_list = args.Rlist
    MAP_list = []
    R_list = []
    APx_list = []
    str_MAP='eval_MAP:\t'
    str_R='R:\t'
    str_APx=['APx:\t' for i in range(len(test_hash))]
    for _, argsR in enumerate(argsR_list):
        MAP, R, APx = mean_average_precision(database_hash, test_hash, database_labels, test_labels, argsR, args.T)
        MAP_list.append(MAP)
        R_list.append(R)
        APx_list.append(APx)
        str_MAP += '{:.4f}'.format(MAP) + '\t'
        str_R += str(R) + '\t'
        for i, val in enumerate(APx):
            str_APx[i] += str(val) + '\t'
    
    print(str_MAP)
    print(str_R)
    #print('\n'.join(str_APx))

    MAP, R, APx = MAP_list[0], R_list[0], APx_list[0]
    #MAP, R, APx = mean_average_precision(database_hash, test_hash, database_labels, test_labels, args.R, args.T)
    end = time.time()
    print('MAP time:'+str(end-start), flush=True)
    print('R={}, MAP {:.4f}, Recall {:.4f}'.format(args.R, MAP, R), flush=True)

    return MAP

def test(model, database_dataset, test_dataset, args):
    database_loader = DataLoader(database_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    MAP = test_MAP(model, database_loader, test_loader, args)

    print(MAP)

def main():
    ## fix seed
    seed = 13
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed+1)
    torch.manual_seed(seed+2)
    torch.cuda.manual_seed_all(seed+3)

    parser = argparse.ArgumentParser( description='test_mrhash',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # model
    parser.add_argument('--model_type', type=str, default='alexnet', help='base model')
    # Hashing
    parser.add_argument('--hash_bit', type=int, default=32, help = 'hash bit')

    # Testing
    parser.add_argument('--pretrain_path', type=str, default='null')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--data_name', type=str, default='EuroSAT', help='eurosat')
    parser.add_argument('--R', type=int, default=1900, help='MAP@R')
    parser.add_argument('--T', type=float, default=0, help='Threshold for binary')
    parser.add_argument('--gen_similarity_matrix', type=int, default=0, help='0/1')
    parser.add_argument('--pretrained_type', type=str, default='alexnet', help='alexnet/resnet50')
    parser.add_argument('--model_name', type=str, default='mrhash', help='mrhash')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    args.device = torch.device("cuda" if args.cuda else "cpu")

    if args.data_name == 'EuroSAT' or args.data_name == 'eurosat':
        path1 = 'data/eurosat/all_features_resnet50_without_l2_msi.pickle'
        path2 = 'data/eurosat/all_features_resnet50_without_l2_vhr.pickle'
        class_cnt = 10
        database_dataset = RSDataset('data/EuroSAT/database.txt', path1, path2, class_cnt, start=0, end=26000, is_train=False)
        test_dataset = RSDataset('data/EuroSAT/test.txt', path1, path2, class_cnt, start=26000, end=27000, is_train=False)
        enc_dims=[(database_dataset.x1_list.shape[1], database_dataset.x2_list.shape[1]), 512, 512, 256, 256]
        args.Rlist = [1900]
    else:
        print('Not Implemented')
        assert False
    
    for k, v in vars(args).items():
        print('\t{}: {}'.format(k, v))
    
    model = torch.load(args.pretrain_path)

    if args.gen_similarity_matrix != 1:
        print('test')
        test(model, database_dataset, test_dataset, args)
    else:
        print('generate similarity matrix')
        output_filename='data/' + args.data_name + '/result/'+args.model_name+'_sim_'+args.pretrained_type+'_h' + str(args.hash_bit)+'.npy'
        output_hash_filename='data/' + args.data_name + '/result/'+args.model_name+'_hash_'+args.pretrained_type+'_h' + str(args.hash_bit)+'.npy'
        print(output_filename)
        gen_similarity_matrix(model, database_dataset, test_dataset, output_filename, output_hash_filename, args)


if __name__ == '__main__':
    main()
    