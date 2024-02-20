import torch.nn as nn
from torchvision import models
import torch
import torch.nn.functional as F
from SCNN import SCNN
from copy import deepcopy
import os


class DBCNN(nn.Module):

    def __init__(self, config):
        """Declare all needed layers."""
        nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.n_task = config.n_task
        self.backbone = models.resnet18(pretrained=True)
        self.config = config

        scnn = SCNN()
        scnn = torch.nn.DataParallel(scnn).cuda()
        scnn.load_state_dict(torch.load(config.scnn_root))
        self.sfeatures = scnn.module.features
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.ModuleList()
        # using deepcopy to insure each fc layer initialized from the same parameters
        # (for fair comparision with sequentail/individual training)
        if self.config.JL:
            fc = nn.Linear(512 * 128, 1, bias=True)
        else:
            fc = nn.Linear(512 * 128, 1, bias=False)
        # Initialize the fc layers.
        nn.init.kaiming_normal_(fc.weight.data)
        for i in range(0, self.n_task):
            self.fc.append(deepcopy(fc))

        #always freeze SCNN
        for param in self.sfeatures.parameters():
            param.requires_grad = False

        if config.fc:
            # Freeze all previous layers.
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward pass of the network.
        """
        N = x.size()[0]

        x1 = self.backbone.conv1(x)
        x1 = self.backbone.bn1(x1)
        x1 = self.backbone.relu(x1)
        x1 = self.backbone.maxpool(x1)

        x1 = self.backbone.layer1(x1)
        x1 = self.backbone.layer2(x1)
        x1 = self.backbone.layer3(x1)
        x1 = self.backbone.layer4(x1)

        H = x1.size()[2]
        W = x1.size()[3]
        assert x1.size()[1] == 512

        x2 = self.sfeatures(x)
        H2 = x2.size()[2]
        W2 = x2.size()[3]
        assert x2.size()[1] == 128

        sfeat = self.pooling(x2)
        sfeat = sfeat.squeeze(3).squeeze(2)
        sfeat = F.normalize(sfeat, p=2)

        if (H != H2) | (W != W2):
            x2 = F.upsample_bilinear(x2, (H, W))

        x1 = x1.view(N, 512, H * W)
        x2 = x2.view(N, 128, H * W)
        x = torch.bmm(x1, torch.transpose(x2, 1, 2)) / (H * W)  # Bilinear
        assert x.size() == (N, 512, 128)
        x = x.view(N, 512 * 128)
        #x = torch.sqrt(x + 1e-8)
        x = F.normalize(x)

        output = []

        if not self.config.JL:
            for idx, fc in enumerate(self.fc):
                if not self.config.train:
                    for W in fc.parameters():
                        fc.weight.data = F.normalize(W, p=2, dim=1)
                output.append(fc(x))
        else:
            for idx, fc in enumerate(self.fc):
                output.append(fc(x))


        return output, sfeat


class DBCNN_bn(nn.Module):

    def __init__(self, config):
        """Declare all needed layers."""
        nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.n_task = config.n_task
        self.backbone = models.resnet18(pretrained=True)
        self.config = config

        scnn = SCNN()
        scnn = torch.nn.DataParallel(scnn).cuda()
        scnn.load_state_dict(torch.load(config.scnn_root))
        self.sfeatures = scnn.module.features
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.ModuleList()
        # using deepcopy to insure each fc layer initialized from the same parameters
        # (for fair comparision with sequentail/individual training)
        if self.config.JL:
            fc = nn.Linear(512 * 128, 1, bias=True)
        else:
            fc = nn.Linear(512 * 128, 1, bias=False)
        # Initialize the fc layers.
        nn.init.kaiming_normal_(fc.weight.data)
        for i in range(0, self.n_task):
            self.fc.append(deepcopy(fc))

        #always freeze SCNN
        for param in self.sfeatures.parameters():
            param.requires_grad = False

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

    def forward(self, x):
        """Forward pass of the network.
        """
        N = x.size()[0]

        features = []

        x1 = self.backbone.conv1(x)
        x1 = self.backbone.bn1(x1)
        x1 = self.backbone.relu(x1)
        x1 = self.backbone.maxpool(x1)

        x1 = self.backbone.layer1(x1)
        feat1 = x1.squeeze(3).squeeze(2)
        feat1 = F.normalize(feat1, p=2)
        features.append(feat1)
        x1 = self.backbone.layer2(x1)
        feat2 = x1.squeeze(3).squeeze(2)
        feat2 = F.normalize(feat2, p=2)
        features.append(feat2)
        x1 = self.backbone.layer3(x1)
        feat3 = x1.squeeze(3).squeeze(2)
        feat3 = F.normalize(feat3, p=2)
        features.append(feat3)
        x1 = self.backbone.layer4(x1)
        feat4 = x1.squeeze(3).squeeze(2)
        feat4 = F.normalize(feat4, p=2)
        features.append(feat4)

        H = x1.size()[2]
        W = x1.size()[3]
        assert x1.size()[1] == 512

        x2 = self.sfeatures(x)
        H2 = x2.size()[2]
        W2 = x2.size()[3]
        assert x2.size()[1] == 128

        sfeat = self.pooling(x2)
        sfeat = sfeat.squeeze(3).squeeze(2)
        sfeat = F.normalize(sfeat, p=2)

        if (H != H2) | (W != W2):
            x2 = F.upsample_bilinear(x2, (H, W))

        x1 = x1.view(N, 512, H * W)
        x2 = x2.view(N, 128, H * W)
        x = torch.bmm(x1, torch.transpose(x2, 1, 2)) / (H * W)  # Bilinear
        assert x.size() == (N, 512, 128)
        x = x.view(N, 512 * 128)
        #x = torch.sqrt(x + 1e-8)
        x = F.normalize(x)

        output = []

        if not self.config.JL:
            for idx, fc in enumerate(self.fc):
                if not self.config.train:
                    for W in fc.parameters():
                        fc.weight.data = F.normalize(W, p=2, dim=1)
                output.append(fc(x))
        else:
            for idx, fc in enumerate(self.fc):
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