# -*- coding: utf-8 -*-
#

import torch.nn as nn
import torchvision
import torch
from torch.nn import Linear
from torch.nn.parameter import Parameter

class MrHash(nn.Module):
    def __init__(self, args, multilabelnet=None):
        super(MrHash, self).__init__()
        self.hash_bit = args.hash_bit

        self.deephashingnet = DeepHashingNet(args)

        laten_feats_dim = args.laten_dim
        self.laten_feats_layer = nn.Linear(self.hash_bit, laten_feats_dim, bias=False)
        
        self.multilabelnet = multilabelnet
        self.slow_lr_paramaters = nn.Sequential(self.deephashingnet.features, self.deephashingnet.avgpool, self.deephashingnet.classifier)
        self.fast_lr_paramaters = nn.Sequential(self.multilabelnet, self.deephashingnet.hash_layer, self.laten_feats_layer)


    def forward(self, x, x_features=None):
        x_hash = self.deephashingnet(x)
        if x_features is not None:
            x_laten = self.laten_feats_layer(x_hash)
            x_laten_q = 1.0 / (1.0 + torch.sum(torch.pow(x_laten.unsqueeze(1) - self.multilabelnet.cluster_layer, 2), 2) / self.multilabelnet.alpha)
            x_laten_q = x_laten_q.pow((self.multilabelnet.alpha + 1.0) / 2.0)
            x_laten_q = (x_laten_q.t() / torch.sum(x_laten_q, 1)).t()
            x_reconstruct1, x_reconstruct2, x_target_q = self.multilabelnet(x_features)
            
            return x_hash, x_reconstruct1, x_reconstruct2, x_laten_q, x_target_q
        else:
            return x_hash

class DeepHashingNet(nn.Module):
    def __init__(self, args):
        super(DeepHashingNet, self).__init__()
        self.hash_bit = args.hash_bit

        self.base_model = torchvision.models.alexnet(pretrained=True)
        self.features = self.base_model.features
        self.avgpool = self.base_model.avgpool
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i), self.base_model.classifier[i])

        feature_dim = self.base_model.classifier[6].in_features
        self.last_hash_layer = nn.Linear(feature_dim, self.hash_bit)
        self.last_layer = nn.Tanh()

        self.hash_layer = nn.Sequential(self.last_hash_layer, self.last_layer)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        x_hash = self.hash_layer(x)
        
        return x_hash

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

class MultilabelNet(nn.Module):

    def __init__(self,
                 n_laten_dim,
                 n_clusters,
                 alpha=1,
                 ae = None):
        super(MultilabelNet, self).__init__()

        self.alpha = 1.0
        self.ae = ae
        
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_laten_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, x):

        x_reconstruct1, x_reconstruct2, x_compressed = self.ae(x)
        
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(x_compressed.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return x_reconstruct1, x_reconstruct2, q