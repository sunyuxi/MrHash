# -*- coding: utf8 -*-
#

import torch
import torch.nn as nn
from torch.nn import Linear
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam

import argparse
import sys
import random
import numpy as np
import itertools
import pickle
import os
import time

from utils import load_alexnet_feature, DenseFeatureTwoDataset
#mean_average_precision_torch,
from utils import mean_average_precision, sanitize

def repeat(l, r):
    return list(itertools.chain.from_iterable(itertools.repeat(x, r) for x in l))

class MyView(nn.Module):
    def __init__(self, size):
        super(MyView, self).__init__()
        self.size = size

    def forward(self, input):
        batch_size = input.size(0)
        out = input.view(batch_size, *self.size)
        return out # (batch_size, *size)


class AELinearConv(nn.Module):

    def __init__(self, laten_dim, enc_dims=[(4096, 2048), 512, 512, 256, 256]):
        super(AELinearConv, self).__init__()
        
        self.laten_dim = laten_dim

        self.encoder = nn.Sequential(
            Linear(enc_dims[0][0], enc_dims[1]), nn.ReLU(),
            Linear(enc_dims[1], enc_dims[2]), nn.ReLU(),
            Linear(enc_dims[2], enc_dims[3]), nn.ReLU(),
            Linear(enc_dims[3], enc_dims[4]), nn.ReLU(),
            Linear(enc_dims[4], laten_dim),
        )
        self.decoder1 = nn.Sequential(
            Linear(laten_dim, enc_dims[4]), nn.ReLU(), 
            Linear(enc_dims[4], enc_dims[3]), nn.ReLU(),
            Linear(enc_dims[3], enc_dims[2]), nn.ReLU(),
            Linear(enc_dims[2], enc_dims[1]), nn.ReLU(),
            Linear(enc_dims[1], enc_dims[0][0])
        )
        self.decoder2 = nn.Sequential(
            Linear(laten_dim, enc_dims[4]), nn.ReLU(), 
            Linear(enc_dims[4], enc_dims[3]), nn.ReLU(),
            Linear(enc_dims[3], enc_dims[2]), nn.ReLU(),
            Linear(enc_dims[2], enc_dims[1]), nn.ReLU(),
            Linear(enc_dims[1], enc_dims[0][1])
        )

    def forward(self, x):
        x_compressed = self.encoder(x)
        x_reconstruct1 = self.decoder1(x_compressed)
        x_reconstruct2 = self.decoder2(x_compressed)
        
        return x_reconstruct1, x_reconstruct2, x_compressed

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (0.5 ** (epoch // 40))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def train_ae(model, train_dataset, valid_dataset, test_dataset, args):

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = Adam(model.parameters(), lr=args.lr)
    
    criterion_mse = nn.MSELoss().to(args.device)

    mAP_best = 0.0
    for epoch in range(args.epoch):

        lr = adjust_learning_rate(optimizer, epoch, args)
        #lr = args.lr
        print('epoch: {} lr: {:.8f}'.format(epoch, lr))

        model.train()
        total_loss, total_loss1, total_loss2 = [], [], []
        for batch_idx, (x_features, x_features_ext, y, _) in enumerate(train_loader):
            x_features = x_features.to(args.device)
            x_features_ext = x_features_ext.to(args.device)

            optimizer.zero_grad()
            x_reconstruct1, x_reconstruct2, _ = model(x_features)

            loss1 = criterion_mse(x_reconstruct1, x_features)
            loss2 = criterion_mse(x_reconstruct2, x_features_ext)
            loss = loss2 + args.lambda1 * loss1 #loss1#
            loss.backward()
            optimizer.step()

            total_loss.append( ( loss.item(), ) )
            total_loss1.append( ( loss1.item(), ) )
            total_loss2.append( ( loss2.item(), ) )

            if batch_idx%10 == 0:
                total_loss_mean, total_loss1_mean, total_loss2_mean = np.array(total_loss).mean(axis=0), np.array(total_loss1).mean(axis=0), np.array(total_loss2).mean(axis=0)
                print('epoch {} iter {} : {:.6f} = {:.6f} + {:.2f}*{:.6f}'.format(epoch, batch_idx, total_loss_mean[0], total_loss2_mean[0], args.lambda1, total_loss1_mean[0]))

        total_loss_mean, total_loss1_mean, total_loss2_mean = np.array(total_loss).mean(axis=0), np.array(total_loss1).mean(axis=0), np.array(total_loss2).mean(axis=0)
        print('epoch {} : {:.6f} = {:.6f} + {:.2f}*{:.6f}'.format(epoch, total_loss_mean[0], total_loss2_mean[0], args.lambda1, total_loss1_mean[0]))       
        sys.stdout.flush()

        mAP_cur = eval_MAP(epoch, model, valid_loader, test_loader, args.R, args.data_hash)
        if mAP_cur>mAP_best:
            old_best = mAP_best
            mAP_best = mAP_cur
            cur_path = args.model_path+'_{:.6f}'.format(mAP_best)
            torch.save(model.state_dict(), cur_path)
            print("model saved to {}".format(cur_path))
            old_path = args.model_path+'_{:.6f}'.format(old_best)
            if os.path.exists(old_path):
                os.remove(old_path)
        print('epoch {} : best mAP: {:.6f} '.format(epoch, mAP_best))


def get_compressed_x(model, data_loader):  # data_loader is database_loader or test_loader
    model.eval()
    is_start = True
    for i, (x_features, x_features_ext, y, _) in enumerate(data_loader):
        x_features = Variable(x_features).cuda()
        x_features_ext = Variable(x_features_ext).cuda()
        
        _, _, x_compressed = model(x_features)

        if is_start:
            all_x_compressed = x_compressed.data.cpu().float()
            all_y = y
            is_start = False
        else:
            all_x_compressed = torch.cat((all_x_compressed, x_compressed.data.cpu().float()), 0)
            all_y = np.concatenate((all_y, y), 0)

    return all_x_compressed.cpu().numpy(), all_y

def eval_MAP(epoch, model, valid_loader, test_loader, R, data_hash=0):

    x_valid, y_valid=get_compressed_x(model, valid_loader)
    x_test, y_test=get_compressed_x(model, test_loader)
    

    x_valid = sanitize(x_valid)
    x_test = sanitize(x_test)

    # transform data to hashcode
    if data_hash == 1:
        print('transform data to hashcode')
        x_valid[x_valid<=0] = -1
        x_valid[x_valid>0] = 1
        x_test[x_test<=0] = -1
        x_test[x_test>0] = 1
    else:
        print('original data')

    x_valid = torch.from_numpy(x_valid).cuda()
    x_test = torch.from_numpy(x_test).cuda()
    sim = torch.cdist(x_valid, x_test, p=2.0)
    sim = sim.cpu().detach().numpy()
    #x_valid, x_test = x_valid.cpu().detach().numpy(), x_test.cpu().detach().numpy()
    print(sim.shape)
    #mAP, _, _ = mean_average_precision_torch(-sim, y_valid, y_test, R, 0)
    #print('epoch_torch {}, mAP: {:.4f}'.format(epoch, mAP))

    mAP, _, _ = mean_average_precision(-sim, y_valid, y_test, R, 0)
    print('epoch {}, mAP: {:.4f}'.format(epoch, mAP))

    sys.stdout.flush()
    return mAP


def test_ae(model, valid_dataset, test_dataset, args):
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model.eval()
    eval_MAP(0, model, valid_loader, test_loader, args.R, args.data_hash)

def main():
    
    seed = 13
    random.seed(seed)
    np.random.seed(seed+1)
    torch.manual_seed(seed+2)
    torch.cuda.manual_seed_all(seed+3)

    #print( 'CUDA:' + str(os.environ['CUDA_VISIBLE_DEVICES']) )

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--dataset', type=str, default='eurosat')
    parser.add_argument('--pretrained_type', type=str, default='resnet50')

    parser.add_argument('--laten_dim', default=64, type=int)
    parser.add_argument('--data_hash', default=0, type=int) # 0 false 1 true
    parser.add_argument('--R', default=1900, type=int)

    parser.add_argument('--ae_name', type=str, default='aelinearconv')
    parser.add_argument('--lambda1', type=float, default=0.1)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--epoch', default=200, type=int)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    args.device = torch.device("cuda" if args.cuda else "cpu")
    
    args.model_path = 'data/' + str(args.dataset) + '/model/ae/two_ae_wonorm_'+ str(args.dataset) + '_' + args.ae_name + '_' + str(args.laten_dim)
    
    if args.dataset == 'EuroSAT' or args.dataset == 'eurosat':
        path1 = 'data/eurosat/all_features_resnet50_without_l2_msi.pickle'
        path2 = 'data/eurosat/all_features_resnet50_without_l2_vhr.pickle'
        class_cnt = 10
        train_dataset = DenseFeatureTwoDataset(path1, path2, class_cnt, start=0, end=26000)
        valid_dataset = DenseFeatureTwoDataset(path1, path2, class_cnt, start=0, end=26000)
        test_dataset = DenseFeatureTwoDataset(path1, path2, class_cnt, start=26000, end=27000)
        enc_dims=[(train_dataset.x1_list.shape[1], train_dataset.x2_list.shape[1]), 512, 512, 256, 256]
        print('eurosat')
    elif args.dataset == 'DSRSID':
        path1 = 'data/DSRSID/all_features_resnet50_without_l2_mul.pickle'
        path2 = 'data/DSRSID/all_features_resnet50_without_l2_pan.pickle'
        class_cnt = 8
        train_dataset = DenseFeatureTwoDataset(path1, path2, class_cnt, start=0, end=75000)
        valid_dataset = DenseFeatureTwoDataset(path1, path2, class_cnt, start=0, end=75000)
        test_dataset = DenseFeatureTwoDataset(path1, path2, class_cnt, start=75000, end=80000)
        enc_dims=[(train_dataset.x1_list.shape[1], train_dataset.x2_list.shape[1]), 512, 512, 256, 256]
        print('DSRSID')
    else:
        assert False
    print(enc_dims)
    print((path1, path2))
    
    model = AELinearConv( laten_dim=args.laten_dim, enc_dims = enc_dims ).to(args.device)
    
    print('modelname: AELinearConv')
    print(model)
    print(args)

    train_ae(model, train_dataset, valid_dataset, test_dataset, args)

if __name__ == '__main__':
    main()