import torch.nn as nn
from torchvision import models
#from pretrainedmodels import se_resnet50, se_resnext50_32x4d
import numpy as np
from copy import deepcopy
import torch
from torch.nn.utils import weight_norm
import torch.nn.functional as F
#import faiss
import time
from BCNN import BCNN
import os
import math
import timm

class View(nn.Module):
    """Changes view using a nn.Module."""

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


def weight_init(param):
    for m in param.modules():
        if isinstance(m, nn.Conv2d):
             nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
class SCNN(nn.Module):

    def __init__(self):
        """Declare all needed layers."""
        super(SCNN, self).__init__()

        # Linear classifier.

        self.num_class = 39
        self.features = nn.Sequential(nn.Conv2d(3,48,3,1,1),nn.BatchNorm2d(48),nn.ReLU(inplace=True),
                                      nn.Conv2d(48,48,3,2,1),nn.BatchNorm2d(48),nn.ReLU(inplace=True),
                                      nn.Conv2d(48,64,3,1,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                      nn.Conv2d(64,64,3,2,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                      nn.Conv2d(64,64,3,1,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                      nn.Conv2d(64,64,3,2,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                      nn.Conv2d(64,128,3,1,1),nn.BatchNorm2d(128),nn.ReLU(inplace=True),
                                      nn.Conv2d(128,128,3,1,1),nn.BatchNorm2d(128),nn.ReLU(inplace=True),
                                      nn.Conv2d(128,128,3,2,1),nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        weight_init(self.features)
        self.pooling = nn.AvgPool2d(14,1)
        self.projection = nn.Sequential(nn.Conv2d(128,256,1,1,0), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                                        nn.Conv2d(256,256,1,1,0), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        weight_init(self.projection)
        self.classifier = nn.Linear(256,self.num_class)
        weight_init(self.classifier)

        self.pooling = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, X):
        feat = self.features(X)
        X = self.pooling(feat)
        X = X.squeeze(3).squeeze(2)
        X = F.normalize(X, p=2)
        return X, feat

    def save_bn(self, name='saved_bn.pt'):
        bns = nn.ModuleList()
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                bns.append(module)

        bn_name = os.path.join(self.config.ckpt_path, name)
        torch.save(bns, bn_name)


    def load_bn(self, bn_path):
        bns = torch.load(bn_path)
        idx = 0
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.load_state_dict(bns[idx].state_dict())
                idx = idx + 1


class BaseCNN_bn(nn.Module):
    def __init__(self, config):
        """Declare all needed layers."""
        nn.Module.__init__(self)

        self.config = config
        self.n_task = config.n_task
        if (self.config.backbone == 'resnet18'):
            self.backbone = models.resnet18(pretrained=True)
            feat_dim = 512
        elif self.config.backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            feat_dim = 2048
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.save_bn(name='imagenet_bn.pt')

        outdim = 1
        self.fc = nn.ModuleList()
        fc = nn.Linear(feat_dim, outdim, bias=False)
        for i in range(0, self.n_task):
            self.fc.append(deepcopy(fc))

        # Freeze all layers.
        for param in self.backbone.parameters():
            param.requires_grad = False

        if not self.config.fc:
            # Free bn parameters to be learnable on a specific task
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    for param in module.parameters():
                        param.requires_grad = True
                        #param.requires_grad = False #ablation

    def forward(self, im):
        """Forward pass of the network.
        """
        features = []

        N = im.size(0)

        x = self.backbone.conv1(im)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        #feature
        feat1 = self.backbone.avgpool(x)
        feat1 = feat1.squeeze(3).squeeze(2)
        feat1 = F.normalize(feat1, p=2)
        features.append(feat1)

        x = self.backbone.layer2(x)
        #feature
        feat2 = self.backbone.avgpool(x)
        feat2 = feat2.squeeze(3).squeeze(2)
        feat2 = F.normalize(feat2, p=2)
        features.append(feat2)

        x = self.backbone.layer3(x)
        if self.config.representation == 'bpv':
            layer3_feat = x

        #feature
        feat3 = self.backbone.avgpool(x)
        feat3 = feat3.squeeze(3).squeeze(2)
        feat3 = F.normalize(feat3, p=2)
        features.append(feat3)

        x = self.backbone.layer4(x)
        #feature
        feat4 = self.backbone.avgpool(x)
        feat4 = feat4.squeeze(3).squeeze(2)
        feat4 = F.normalize(feat4, p=2)
        features.append(feat4)

        x = self.backbone.avgpool(x)
        x = x.squeeze(3).squeeze(2)
        x = F.normalize(x, p=2)

        output = []
        for idx, fc in enumerate(self.fc):
            for W in fc.parameters():
                #W = F.normalize(W, p=2, dim=1)
                if not self.config.train:
                    fc.weight.data = F.normalize(W, p=2, dim=1)
                    #fc.weight.data = fc.weight.data
            output.append(fc(x))

        return output, features

    def save_bn(self, name='saved_bn.pt'):
        bns = nn.ModuleList()
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                bns.append(module)

        bn_name = os.path.join(self.config.ckpt_path, name)
        torch.save(bns, bn_name)

    def load_bn(self, bn_path):
        bns = torch.load(bn_path)
        idx = 0
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.load_state_dict(bns[idx].state_dict())
                idx = idx + 1

    def load_bn_from_cache(self, bns):
        idx = 0
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.load_state_dict(bns[idx].state_dict())
                idx = idx + 1


class BaseCNN_vanilla(nn.Module):
    def __init__(self, config):
        """Declare all needed layers."""
        nn.Module.__init__(self)

        self.config = config
        self.n_task = config.n_task
        if self.config.backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
        elif self.config.backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        outdim = 1
        self.fc = nn.ModuleList()

        fc = nn.Linear(512, outdim, bias=False)
        for i in range(0, self.n_task):
            self.fc.append(deepcopy(fc))

        if self.config.fc:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, im):
        """Forward pass of the network.
        """
        features = []

        x = self.backbone.conv1(im)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = x.squeeze(3).squeeze(2)
        x = F.normalize(x, p=2)

        output = []
        for idx, fc in enumerate(self.fc):
            for W in fc.parameters():
                #W = F.normalize(W, p=2, dim=1)
                if not self.config.train:
                    fc.weight.data = F.normalize(W, p=2, dim=1)
                    #fc.weight.data = fc.weight.data
            output.append(fc(x))

        return output, x