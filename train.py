# -*- coding: utf-8 -*-
#

import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans

import os
import sys
import argparse
import numpy as np
import random

from network import MrHash, MultilabelNet, AELinearConv
from meters import AverageMeter
from test import test_MAP
from dataset import RSDataset, load_similarity_matrix, l2normalize_matrix

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (0.7 ** (epoch // 10))
    optimizer.param_groups[0]['lr'] = args.multi_lr*lr
    optimizer.param_groups[1]['lr'] = lr

    return lr

def loss_store_init(loss_store):
    """
    initialize loss store, transform list to dict by (loss name -> loss register)
    :param loss_store: the list with name of loss
    :return: the dict of loss store
    """
    dict_store = {}
    for loss_name in loss_store:
        dict_store[loss_name] = AverageMeter()
    loss_store = dict_store
    
    return loss_store

def print_loss(epoch, max_epoch, batch_info, loss_store=None):
    if batch_info is not None:
        loss_str = "epoch:[%3d/%3d] [%3d/%3d], " % (epoch, max_epoch, batch_info[0], batch_info[1])
    else:
        loss_str = "epoch:[%3d/%3d] , " % (epoch, max_epoch)

    for name, value in loss_store.items():
        loss_str += name + " {:4.3f}".format(value.avg) + "\t"
    print(loss_str)
    sys.stdout.flush()

def reset_loss(loss_store=None):
    for store in loss_store.values():
        store.reset()

def remark_loss(loss_store, *args):
    """
    store loss into loss store by order
    :param args: loss to store
    :return:
    """
    for i, loss_name in enumerate(loss_store.keys()):
        if isinstance(args[i], torch.Tensor):
            loss_store[loss_name].update(args[i].item())
        else:
            loss_store[loss_name].update(args[i])

def get_compressed_x(model, train_dataset, args):
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model.eval()
    
    all_x_compressed = None
    for i, (_, x_features, _, y, index) in enumerate(train_loader):
        x_features = Variable(x_features).cuda()
        index = index.data.cpu()
        _, _, x_compressed = model(x_features)
        x_compressed = x_compressed.data.cpu().float().numpy()
        y = y.numpy()

        if all_x_compressed is None:
            #print((x_compressed.dtype, y.dtype), flush=True)
            all_x_compressed = np.zeros((len(train_dataset), x_compressed.shape[1]), dtype=x_compressed.dtype)
            all_y = np.zeros((len(train_dataset), y.shape[1]), dtype=y.dtype)
        all_x_compressed[index], all_y[index] = x_compressed, y

    return all_x_compressed, all_y

def get_similarity_matrix_train(model, train_dataset, args):
    x_data, _ = get_compressed_x(model, train_dataset, args)
    x_norm_data = l2normalize_matrix(x_data)
    sim = np.dot(x_norm_data, x_norm_data.T)
    
    max_val, min_val = np.max(sim), np.min(sim)
    assert abs(max_val-min_val)>0.00001
    return 2*(sim-min_val)/(max_val-min_val)-1

def target_distribution(q):
        weight = q**2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

def train_hash(mrhash_model, train_dataset, database_dataset, test_dataset, args):

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    database_loader = DataLoader(database_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    
    #
    params_list = [{'params': mrhash_model.slow_lr_paramaters.parameters(), 'lr': args.multi_lr*args.lr},
                   {'params': mrhash_model.fast_lr_paramaters.parameters()},]
    optimizer = torch.optim.Adam(params_list, lr = args.lr, betas=(0.9, 0.999))

    loss_store = ['kl', 'reconstruct', "loss"]
    loss_store = loss_store_init(loss_store)

    criterion_mse = nn.MSELoss().to(args.device)
    
    best_MAP = 0
    for epoch in range(args.max_epoch+1):

        adjust_learning_rate(optimizer, epoch, args)
        lr0 = optimizer.param_groups[0]['lr']
        lr1 = optimizer.param_groups[1]['lr']
        print('epoch: {} lr0: {:.8f} lr1: {:.8f}'.format(epoch, lr0, lr1))
        
        with torch.no_grad():
            mrhash_model.eval()
            x_train_features = np.array(train_dataset.x1_list)
            x_train_features = torch.Tensor(x_train_features).to(args.device)
            _, _, tmp_q = mrhash_model.multilabelnet(x_train_features)
            tmp_q = tmp_q.data
            target_p = target_distribution(tmp_q)

        mrhash_model.train()
        total_batch = len(train_loader)
        for batch_idx, (x_imgs, x_features, x_features_ext, _, index) in enumerate(train_loader):
            x_imgs, x_features, x_features_ext = x_imgs.to(args.device), x_features.to(args.device), x_features_ext.to(args.device)
            
            optimizer.zero_grad()
            x_hash, x_reconstruct1, x_reconstruct2, x_imgs_q, x_features_q = mrhash_model(x_imgs, x_features)

            loss_reconstruct = criterion_mse(x_reconstruct1, x_features) + criterion_mse(x_reconstruct2, x_features_ext)
            loss_kl = F.kl_div(x_imgs_q.log(), target_p[index]) + F.kl_div(x_features_q.log(), target_p[index])
            
            loss = loss_kl + args.lambda0 * loss_reconstruct 

            loss.backward()
            optimizer.step()
    
            remark_loss(loss_store, loss_kl, loss_reconstruct, loss)
            
            print_loss(epoch, args.max_epoch, (batch_idx, total_batch), loss_store)
            sys.stdout.flush()

        print_loss(epoch, args.max_epoch, None, loss_store)
        reset_loss(loss_store)

        if epoch%args.eval_frequency == 0:
            print('eval, epoch: %d' % epoch)
            MAP = test_MAP(mrhash_model, database_loader, test_loader, args)
            if MAP > best_MAP:
                old_best = best_MAP
                best_MAP = MAP
                
            print('MAP:{:.4f} best_MAP:{:.4f}'.format(MAP, best_MAP))

        if epoch == args.max_epoch:
            torch.save(mrhash_model, args.model_path+'_'+'{:.4f}'.format(MAP))
            print('save model in: %s' % args.model_path)


def main():

    ## fix seed
    seed = 13
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed+1)
    torch.manual_seed(seed+2)
    torch.cuda.manual_seed_all(seed+3)

    parser = argparse.ArgumentParser( description='train_mrhash',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # model
    parser.add_argument('--model_type', type=str, default='alexnet', help='base model')
    # Hashing
    parser.add_argument('--hash_bit', type=int, default=32, help = 'hash bit')
    parser.add_argument('--laten_dim', type=int, default=64, help = 'the dim of laten features')
    parser.add_argument('--n_clusters', type=int, default=16, help = 'cluster count')
    parser.add_argument('--pretrained_dp_path', type=str, default='', help = 'the path of a pretrained autoencoder')

    # Training
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--max_epoch', type=int, default=50, help='training epoch')
    parser.add_argument('--workers', type=int, default=4, help='number of data loader workers.')
    parser.add_argument('--batch_size', type=int, default=128, help='training batch size')
    parser.add_argument('--eval_frequency', type=int, default=5, help='the evaluate frequency for testing')
    parser.add_argument('--data_name', type=str, default='EuroSAT', help='eurosat')
    parser.add_argument('--model_path', type=str, default='EuroSAT', help='eurosat')
    parser.add_argument('--multi_lr', type=float, default=0.01, help= 'multiplier for learning rate')
    parser.add_argument('--lambda0', type=float, default=5.0, help='hyper-parameters 0')
    
    # Testing
    parser.add_argument('--R', type=int, default=1900, help='MAP@R')
    parser.add_argument('--T', type=float, default=0, help='Threshold for binary')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    args.device = torch.device("cuda" if args.cuda else "cpu")

    args.model_path = 'data/' + str(args.data_name) + '/model/mrhash_'+args.model_type+'_'+ str(args.data_name) + '_' + str(args.hash_bit)
    
    if args.data_name == 'EuroSAT' or args.data_name == 'eurosat':
        path1 = 'data/eurosat/all_features_resnet50_without_l2_msi.pickle'
        path2 = 'data/eurosat/all_features_resnet50_without_l2_vhr.pickle'
        class_cnt = 10
        train_dataset = RSDataset('data/EuroSAT/train.txt', path1, path2, class_cnt, start=0, end=26000, is_train=True)
        database_dataset = RSDataset('data/EuroSAT/database.txt', path1, path2, class_cnt, start=0, end=26000, is_train=False)
        test_dataset = RSDataset('data/EuroSAT/test.txt', path1, path2, class_cnt, start=26000, end=27000, is_train=False)
        enc_dims=[(train_dataset.x1_list.shape[1], train_dataset.x2_list.shape[1]), 512, 512, 256, 256]
        args.Rlist = [1900]
    elif args.data_name == 'DSRSID':
        path1 = 'data/DSRSID/all_features_resnet50_without_l2_mul.pickle'
        path2 = 'data/DSRSID/all_features_resnet50_without_l2_pan.pickle'
        class_cnt = 8
        train_dataset = RSDataset('data/DSRSID/train.txt', path1, path2, class_cnt, start=0, end=75000, is_train=True)
        database_dataset = RSDataset('data/DSRSID/database.txt', path1, path2, class_cnt, start=0, end=75000, is_train=False)
        test_dataset = RSDataset('data/DSRSID/test.txt', path1, path2, class_cnt, start=75000, end=80000, is_train=False)
        enc_dims=[(train_dataset.x1_list.shape[1], train_dataset.x2_list.shape[1]), 512, 512, 256, 256]
        args.Rlist = [9375]
    else:
        print('Not Implemented')
        assert False

    ae = AELinearConv(laten_dim=args.laten_dim, enc_dims = enc_dims).to(args.device)
    ae.load_state_dict(torch.load(args.pretrained_dp_path))
    multilabelnet_model = MultilabelNet(n_laten_dim=args.laten_dim, n_clusters=args.n_clusters, alpha=1.0, ae = ae).to(args.device)
    x_compressed, _ = get_compressed_x(ae, train_dataset, args)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    _ = kmeans.fit_predict(x_compressed)
    multilabelnet_model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(args.device)
    
    mrhash_model = MrHash(args, multilabelnet_model).to(args.device)

    print(mrhash_model)

    for k, v in vars(args).items():
        print('\t{}: {}'.format(k, v))

    train_hash(mrhash_model, train_dataset, database_dataset, test_dataset, args)

if __name__ == '__main__':
    main()

