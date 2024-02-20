import os
import time
import scipy.stats
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from MNL_Loss import Fidelity_Loss
from Transformers import AdaptiveResize
from copy import deepcopy
from BaseCNN_all import BaseCNN_bn, BaseCNN_vanilla
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from utils import set_dataset, set_dataset2
from DBCNN import DBCNN, DBCNN_bn

import warnings
warnings.filterwarnings("ignore")
import random

from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()

def fix_ln(m):
    classname = m.__class__.__name__
    if classname.find('LayerNorm') != -1:
        m.eval()

class Trainer(object):
    def __init__(self, config):
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.config = config
        # for replay
        self.replayers = {}

        if self.config.amp:
            self.scaler = GradScaler()

        if not self.config.train:
            self.config.verbose = False

        self.train_transform = transforms.Compose([
            AdaptiveResize(768),
            transforms.RandomCrop(config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

        self.test_transform = transforms.Compose([
            AdaptiveResize(768),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])


        self.train_batch_size = config.batch_size
        self.test_batch_size = 1
        self.ranking = config.ranking

        if self.config.GDumb:
            csv_file = os.path.join(config.trainset, 'gdumb', str(config.indicator), config.train_txt),
        else:
            csv_file = os.path.join(config.trainset, 'splits2', str(config.split), config.train_txt),
        self.train_data, self.train_loader = set_dataset(config, csv_file[0], config.trainset,
                                                         self.train_transform, num_workers=12, shuffle=True,
                                                         test=(not config.ranking), verbose=self.config.verbose)



        #for k-means
        if self.config.train_kmeans:
            self.train_data_kmeans, _ = set_dataset(config, csv_file[0], config.trainset,
                                                    self.test_transform, num_workers=12, shuffle=False,
                                                    test=(not config.ranking), verbose=self.config.verbose)
            self.kmeans_loader = DataLoader(self.train_data_kmeans,
                                            batch_size=1,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=12)


        # testing and validation sets configuration
        csv_file = os.path.join(config.live_set, 'splits2', str(config.split), 'live_test.txt')
        self.live_data, self.live_loader = set_dataset(config, csv_file, config.live_set,
                                                       self.test_transform, num_workers=1, shuffle=False, test=True,
                                                       verbose=self.config.verbose)

        csv_file = os.path.join(config.live_set, 'splits2', str(config.split), 'live_val.txt')
        self.live_data_val, self.live_loader_val = set_dataset(config, csv_file, config.live_set,
                                                       self.test_transform, num_workers=1, shuffle=False, test=True,
                                                       verbose=self.config.verbose)

        csv_file = os.path.join(config.csiq_set, 'splits2', str(config.split), 'csiq_test.txt')
        self.csiq_data, self.csiq_loader = set_dataset(config, csv_file, config.csiq_set,
                                                       self.test_transform, num_workers=1, shuffle=False, test=True,
                                                       verbose=self.config.verbose)

        csv_file = os.path.join(config.csiq_set, 'splits2', str(config.split), 'csiq_val.txt')
        self.csiq_data_val, self.csiq_loader_val = set_dataset(config, csv_file, config.csiq_set,
                                                       self.test_transform, num_workers=1, shuffle=False, test=True,
                                                       verbose=self.config.verbose)


        csv_file = os.path.join(config.bid_set, 'splits2', str(config.split), 'bid_test.txt')
        self.bid_data, self.bid_loader = set_dataset(config, csv_file, config.bid_set,
                                                       self.test_transform, num_workers=1, shuffle=False, test=True,
                                                     verbose=self.config.verbose)

        csv_file = os.path.join(config.bid_set, 'splits2', str(config.split), 'bid_val.txt')
        self.bid_data_val, self.bid_loader_val = set_dataset(config, csv_file, config.bid_set,
                                                       self.test_transform, num_workers=1, shuffle=False, test=True,
                                                     verbose=self.config.verbose)

        csv_file = os.path.join(config.clive_set, 'splits2', str(config.split), 'clive_test.txt')
        self.clive_data, self.clive_loader = set_dataset(config, csv_file, config.clive_set,
                                                       self.test_transform, num_workers=1, shuffle=False, test=True,
                                                         verbose=self.config.verbose)

        csv_file = os.path.join(config.clive_set, 'splits2', str(config.split), 'clive_val.txt')
        self.clive_data_val, self.clive_loader_val = set_dataset(config, csv_file, config.clive_set,
                                                       self.test_transform, num_workers=1, shuffle=False, test=True,
                                                         verbose=self.config.verbose)

        csv_file = os.path.join(config.koniq10k_set, 'splits2', str(config.split), 'koniq10k_test.txt')
        self.koniq10k_data, self.koniq10k_loader = set_dataset(config, csv_file, config.koniq10k_set,
                                                       self.test_transform, num_workers=1, shuffle=False, test=True,
                                                               verbose=self.config.verbose)

        csv_file = os.path.join(config.koniq10k_set, 'splits2', str(config.split), 'koniq10k_val.txt')
        self.koniq10k_data_val, self.koniq10k_loader_val = set_dataset(config, csv_file, config.koniq10k_set,
                                                       self.test_transform, num_workers=1, shuffle=False, test=True,
                                                               verbose=self.config.verbose)

        csv_file = os.path.join(config.kadid10k_set, 'splits2', str(config.split), 'kadid10k_test.txt')
        self.kadid10k_data, self.kadid10k_loader = set_dataset(config, csv_file, config.kadid10k_set,
                                                               self.test_transform, num_workers=1, shuffle=False,
                                                               test=True, verbose=self.config.verbose)

        csv_file = os.path.join(config.kadid10k_set, 'splits2', str(config.split), 'kadid10k_val.txt')
        self.kadid10k_data_val, self.kadid10k_loader_val = set_dataset(config, csv_file, config.kadid10k_set,
                                                               self.test_transform, num_workers=1, shuffle=False,
                                                               test=True, verbose=self.config.verbose)



        self.task2loader = {'live': self.live_loader,
                            'csiq': self.csiq_loader,
                            'bid': self.bid_loader,
                            'clive': self.clive_loader,
                            'koniq10k': self.koniq10k_loader,
                            'kadid10k': self.kadid10k_loader}


        self.task2loader_val = {'live': self.live_loader_val,
                            'csiq': self.csiq_loader_val,
                            'bid': self.bid_loader_val,
                            'clive': self.clive_loader_val,
                            'koniq10k': self.koniq10k_loader_val,
                            'kadid10k': self.kadid10k_loader_val}



        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda else "cpu")

        if self.config.shared_head:
            self.config.n_task = 1
            self.config.task_id = 0

        # initialize the model
        if not self.config.train:
            if config.network == 'basecnn':
                self.model = BaseCNN_vanilla(config)
            elif config.network == 'dbcnn':
                if self.config.train_bn:
                    self.model = DBCNN_bn(config)
                else:
                    self.model = DBCNN(config)
                #self.model.train()
                self.model.sfeatures.apply(fix_bn)
                if self.config.fc:
                    self.model.backbone.apply(fix_bn)
            self.model.eval()
        else:
            if config.network == 'basecnn':
                if self.config.train_bn:
                    self.model = BaseCNN_vanilla(config)
                self.model.train()
                if self.config.fc:
                    self.model.backbone.apply(fix_bn)
                else:
                    if self.config.train_bn:
                        print('only train bn parameters')
                    elif self.config.piggyback:
                        print('Training with PiggyBack, do noting special for bn')
            elif config.network == 'dbcnn':
                if self.config.train_bn:
                    self.model = DBCNN_bn(config)
                    self.model.train()
                    self.model.backbone.apply(fix_bn)
                else:
                    self.model = DBCNN(config)
                    self.model.train()
                self.model.sfeatures.apply(fix_bn)
                if self.config.fc:
                    self.model.backbone.apply(fix_bn)
            else:
                raise NotImplementedError("Not supported network, need to be added!")

        self.model.to(self.device)
        self.model_name = type(self.model).__name__

        if self.config.verbose:
            print(self.model)

        if (not self.config.train) & (self.config.use_scnn):
            self.scnn = BaseCNN_bn(config)
            self.scnn.to(self.device)
            self.scnn.eval()

        # loss function
        if config.ranking:
            if config.fidelity | config.b_fidelity:
                print('use fidelity loss')
                self.loss_fn = Fidelity_Loss()
            else:
                print('use cross entropy loss')
                self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            print('use mse loss')
            self.loss_fn = nn.MSELoss()

        self.loss_fn.to(self.device)

        self.initial_lr = config.lr
        if self.initial_lr is None:
            lr = 0.0005
        else:
            lr = self.initial_lr

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr,
            weight_decay=5e-4)

        # some states
        self.best_srcc = 0
        self.start_epoch = 0
        self.start_step = 0
        self.train_loss = []
        self.test_results_srcc = {'live': [], 'csiq': [], 'bid': [], 'clive': [], 'koniq10k': [], 'kadid10k': []}
        self.val_results_srcc = {'live': [], 'csiq': [], 'bid': [], 'clive': [], 'koniq10k': [], 'kadid10k': []}
        self.ckpt_path = config.ckpt_path
        self.ckpt_resume_path = config.ckpt_resume_path
        self.resume_new = config.resume_new
        self.max_epochs = config.max_epochs
        self.epochs_per_eval = config.epochs_per_eval
        self.epochs_per_save = config.epochs_per_save
        self.indicator = config.indicator

        ###prepare for regularization-based continual learning methods
        #EWC
        self.params = {n:p for n,p in self.model.backbone.named_parameters() if (p.requires_grad) & (not 'fc' in n)}
        self.regularization_terms = {}
        self.task_count = self.config.indicator
        self.online_reg = self.config.online_reg

        # SI
        if self.config.SI:
            self.damping_factor = 0.1
            self.w = {}
            for n, p in self.params.items():
                self.w[n] = p.clone().detach().zero_()
        # The initial_params will only be used in the first task (when the regularization_terms is empty)
        self.initial_params = {}
        for n, p in self.params.items():
            self.initial_params[n] = p.clone().detach()

        # try load the model
        if config.resume or not config.train:
            if (not self.config.train):
                ckpt = self._get_checkpoint_new(path=config.ckpt_resume_path, resume_best=config.resume_best)
                self._load_checkpoint(ckpt=ckpt)
            else:
                if config.ckpt:
                    ckpt = os.path.join(config.ckpt_resume_path, config.ckpt)
                else:
                    ckpt = self._get_checkpoint_new(path=config.ckpt_resume_path, resume_best=config.resume_best)
                self._load_checkpoint(ckpt=ckpt)

            if (self.config.train_bn) & (self.config.train) & (self.config.fc) & (self.config.task_id > 0):
                task_folder = 'train_on_' + self.config.id2dataset[self.config.task_id]
                bn_path = os.path.join(self.config.base_ckpt_path, task_folder, 'imagenet_bn.pt')
                self.model.load_bn(bn_path)

            if self.resume_new:
                self.start_epoch = 0
                self.start_step = 0
                self.train_loss = []
                self.best_srcc = 0
                self.test_results_srcc = {'live': [], 'csiq': [], 'bid': [], 'clive': [],
                                          'koniq10k': [], 'kadid10k': []}
                self.val_results_srcc = {'live': [], 'csiq': [], 'bid': [], 'clive': [],
                                          'koniq10k': [], 'kadid10k': []}

                self.optimizer = torch.optim.Adam(
                    self.model.parameters(), lr=lr,
                    weight_decay=5e-4)

            if config.lwf:
                self.model_old = deepcopy(self.model)
                self.model_old.eval()

        if self.config.SI:
           for n, p in self.params.items():
             self.w[n] = p.clone().detach().zero_()

        if self.config.train_kmeans:
            task_folder = 'train_on_' + self.config.id2dataset[self.config.task_id]

            if self.config.distortion_bn:
                bn_path = os.path.join(self.config.base_ckpt_path, str(self.config.split), task_folder, 'distortion_bn.pt')
            else:
                bn_path = os.path.join(self.config.base_ckpt_path, str(self.config.split), task_folder,
                                       'imagenet_bn.pt')

            if not self.config.network == 'dbcnn':
                self.model.load_bn(bn_path)

        if self.config.train:
            self.scheduler = lr_scheduler.StepLR(self.optimizer,
                                                last_epoch=self.start_epoch - 1,
                                                step_size=config.decay_interval,
                                                gamma=config.decay_ratio)

    def train_kmeans(self):
        if os.path.exists(os.path.join(self.ckpt_path, 'features.npy')) | os.path.exists(os.path.join(self.ckpt_path, 'features1.npy')) :
            if self.config.train_bn:
                features1 = np.load(os.path.join(self.ckpt_path, 'features1.npy'))
                features2 = np.load(os.path.join(self.ckpt_path, 'features2.npy'))
                features3 = np.load(os.path.join(self.ckpt_path, 'features3.npy'))
                features4 = np.load(os.path.join(self.ckpt_path, 'features4.npy'))
                #sfeatures = np.load(os.path.join(self.ckpt_path, 'sfeatures.npy'))
                print('features loaded!')
            else:
                features = np.load(os.path.join(self.ckpt_path, 'features.npy'))
                print('features loaded!')
        else:
            if self.config.network == 'dbcnn':
                features = self.compute_features_single()
                np.save(os.path.join(self.ckpt_path, 'features.npy'), features)
            else:
                with torch.no_grad():
                    # features = self.compute_features()
                    features = self.compute_features_single()
                    if self.config.train_bn:
                        features1 = features[0]
                        features2 = features[1]
                        features3 = features[2]
                        features4 = features[3]
                        #sfeatures = features[4]
                        np.save(os.path.join(self.ckpt_path, 'features1.npy'), features[0])
                        np.save(os.path.join(self.ckpt_path, 'features2.npy'), features[1])
                        np.save(os.path.join(self.ckpt_path, 'features3.npy'), features[2])
                        np.save(os.path.join(self.ckpt_path, 'features4.npy'), features[3])
                        #np.save(os.path.join(self.ckpt_path, 'sfeatures.npy'), features[4])
                    else:
                        np.save(os.path.join(self.ckpt_path, 'features.npy'), features)

                print('features saved!')

        if self.config.train_bn:
            if not os.path.exists(os.path.join(self.ckpt_path, 'cluster_sk1.pt')):
                estimator = KMeans(n_clusters=self.config.num_cluster, n_init=20, verbose=True, max_iter=1000)
                estimator.fit(features1)
                cluster_name = os.path.join(self.ckpt_path, 'cluster_sk1.pt')
                torch.save(torch.from_numpy(estimator.cluster_centers_), cluster_name)

                estimator = KMeans(n_clusters=self.config.num_cluster, n_init=20, verbose=True, max_iter=1000)
                estimator.fit(features2)
                cluster_name = os.path.join(self.ckpt_path, 'cluster_sk2.pt')
                torch.save(torch.from_numpy(estimator.cluster_centers_), cluster_name)

                estimator = KMeans(n_clusters=self.config.num_cluster, n_init=20, verbose=True, max_iter=1000)
                estimator.fit(features3)
                cluster_name = os.path.join(self.ckpt_path, 'cluster_sk3.pt')
                torch.save(torch.from_numpy(estimator.cluster_centers_), cluster_name)

                estimator = KMeans(n_clusters=self.config.num_cluster, n_init=20, verbose=True, max_iter=1000)
                estimator.fit(features4)
                cluster_name = os.path.join(self.ckpt_path, 'cluster_sk4.pt')
                torch.save(torch.from_numpy(estimator.cluster_centers_), cluster_name)
        else:
            # compare
            if not os.path.exists(os.path.join(self.ckpt_path, 'cluster_sk.pt')):
                estimator = KMeans(n_clusters=self.config.num_cluster, n_init=20, verbose=True, max_iter=1000)
                estimator.fit(features)
                cluster_name = os.path.join(self.ckpt_path, 'cluster_sk.pt')
                torch.save(torch.from_numpy(estimator.cluster_centers_), cluster_name)

        if not self.config.train_bn:
            # mean features baseline
            features = np.mean(features, axis=0)
            features = features / np.linalg.norm(features)
            mean_name = os.path.join(self.ckpt_path, 'mean_feature.pt')
            torch.save(torch.from_numpy(features), mean_name)
            print('mean features saved!')

    def fit(self):
        if self.ranking:
            for epoch in range(self.start_epoch, self.max_epochs):
                _ = self._train_single_epoch(epoch, self.train_loader)
                self.scheduler.step()

                if self.config.piggyback:
                    if self.config.threshold_fn == 'binarizer':
                        print('Num 0ed out parameters:')
                        for idx, module in enumerate(self.model.backbone.modules()):
                            if 'ElementWise' in str(type(module)):
                                num_zero = module.mask_real.data.lt(5e-5).sum()
                                total = module.mask_real.data.numel()
                                print(idx, num_zero, total)

            if self.config.reg_trigger:
                # 2.Backup the weight of current task
                # not works for fc layers warm-up
                if (not self.config.fc):

                    model_name_best = os.path.join(self.ckpt_path, self.model_name + '_best.pt')
                    self._load_checkpoint(model_name_best)

                    self.params = {n: p for n, p in self.model.backbone.named_parameters() if
                                   (p.requires_grad) & (not 'fc' in n)}

                    task_param = {}
                    for n, p in self.params.items():
                        task_param[n] = p.clone().detach()
                    # 3.Calculate the importance of weights for current task
                    importance = self.calculate_importance()

                    if self.online_reg and len(self.regularization_terms) > 0:
                        self.regularization_terms[1] = {'importance': importance, 'task_param': task_param}
                    else:
                        self.regularization_terms[self.task_count] = {'importance': importance,
                                                                      'task_param': task_param}

                    model_name = type(self.model).__name__
                    model_name_best = os.path.join(self.ckpt_path, model_name + '_best.pt')

                    epoch = self.start_epoch - 1
                    if self.config.amp:
                        self._save_checkpoint({
                            'epoch': epoch,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'train_loss': self.train_loss,
                            'test_results_srcc': self.test_results_srcc,
                            'val_results_srcc': self.val_results_srcc,
                            'best_srcc': self.best_srcc,
                            'regularization_terms': self.regularization_terms,
                            'replayers': self.replayers,
                            'scaler': self.scaler.state_dict(),
                            'w': self.w
                        }, model_name_best)
                    else:
                        self._save_checkpoint({
                            'epoch': epoch,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'train_loss': self.train_loss,
                            'test_results_srcc': self.test_results_srcc,
                            'val_results_srcc': self.val_results_srcc,
                            'best_srcc': self.best_srcc,
                            'regularization_terms': self.regularization_terms,
                            'replayers': self.replayers,
                            'w': self.w
                        }, model_name_best)

            #save current bn after finishing the training
            if self.config.train_bn:
                self.model.save_bn()
        else:
            raise NotImplementedError("Only support ranking now!")

    def do_batch(self, x1, x2):
        y1, _ = self.model(x1)
        y2, _ = self.model(x2)

        output = []
        for y11, y22 in zip(y1, y2):
            y_diff = y11 - y22
            p = y_diff
            if self.config.fidelity:
                constant = torch.sqrt(torch.Tensor([2])).to(self.device)
                p = 0.5 * (1 + torch.erf(p / constant))
            output.append(p)

        return output


    def do_batch_old(self, x1, x2):
        if self.config.shared_head:
            n_old_task = 1
        else:
            n_old_task = self.config.task_id
        ps = []
        y1, _ = self.model_old(x1)
        y2, _ = self.model_old(x2)
        for task_idx in range(n_old_task):
            y_diff = y1[task_idx] - y2[task_idx]
            p = y_diff

            if self.config.fidelity:
                constant = torch.sqrt(torch.Tensor([2])).to(self.device)
                p = 0.5 * (1 + torch.erf(p / constant))
                ps.append(p)
                #p = 0.5 * (1 + torch.erf(p / constant))
        return ps

    def _train_single_epoch(self, epoch, train_loader):
        # initialize logging system
        num_steps_per_epoch = len(self.train_loader)
        local_counter = epoch * num_steps_per_epoch + 1
        start_time = time.time()
        beta = 0.9
        running_loss = 0 if epoch == 0 else self.train_loss[-1]
        loss_corrected = 0.0
        running_duration = 0.0
        #replay_iter = {}
        if self.config.replay:
            replay_loaders = {}
            for idx, task_name in enumerate(self.replayers):
                replay_loaders[task_name] = iter(self.replayer_list[task_name])
                #replay_iter[task_name] = 1
            if self.config.sample_strategy == 'random':
                picker = np.random.randint(low=0, high=len(self.replayers), size=len(self.train_loader))

        # start training
        print('Adam learning rate: {:.8f}'.format(self.optimizer.param_groups[0]['lr']))
        for step, sample_batched in enumerate(train_loader, 0):
            if step < self.start_step:
                continue
            # for SI
            if self.config.SI:
                # 1.Save current parameters
                old_params = {}
                for n, p in self.params.items():
                    old_params[n] = p.clone().detach()

            x1, x2, g, _, _, yb = sample_batched['I1'], sample_batched['I2'], sample_batched['y'], \
                                               sample_batched['std1'], sample_batched['std2'], sample_batched['yb']

            x1 = Variable(x1)
            x2 = Variable(x2)
            g = Variable(g).view(-1, 1)
            yb = Variable(yb).view(-1, 1)
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            g = g.to(self.device)
            yb = yb.to(self.device)


            if self.config.amp:
                with autocast():
                    p = self.do_batch(x1, x2)
                    if (not self.config.b_fidelity) & (self.config.fidelity):
                        self.loss = self.loss_fn(p[self.config.task_id], g.detach())
                    else:
                        self.loss = self.loss_fn(p[self.config.task_id], yb.detach())

            else:
                p = self.do_batch(x1, x2)
                if (not self.config.b_fidelity) & (self.config.fidelity):
                    self.loss = self.loss_fn(p[self.config.task_id], g.detach())
                else:
                    self.loss = self.loss_fn(p[self.config.task_id], yb.detach())

            if torch.isnan(self.loss):
                print('skip nan loss')
                continue

            reg_loss = 0
            if (self.config.lwf) | ((self.config.lwf)):
                if self.config.amp:
                    with autocast():
                        with torch.no_grad():
                            p_old = self.do_batch_old(x1, x2)
                        if ((self.config.lwf) & (not self.config.icarl)) | (
                                (self.config.lwf)):
                            for i, old_pred in enumerate(p_old):
                                reg_loss += self.config.reg_weight * self.loss_fn(p[i], old_pred.detach())
                else:
                    with torch.no_grad():
                        p_old = self.do_batch_old(x1, x2)
                    if (self.config.lwf) | (self.config.lwf):
                        for i, old_pred in enumerate(p_old):
                            reg_loss += self.config.reg_weight * self.loss_fn(p[i], old_pred.detach())
            else:  ##weight importance regularization
                if self.config.amp:
                    with autocast():
                        reg_loss = self.reg_criterion()
                else:
                    reg_loss = self.reg_criterion()

            self.loss += reg_loss
            if torch.isnan(self.loss):
                print('skip nan loss')
                continue

            self.optimizer.zero_grad()
            if self.config.amp:
                self.scaler.scale(self.loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.loss.backward()
                self.optimizer.step()

            if self.config.SI:
                # 3.Accumulate the w
                for n, p in self.params.items():
                    delta = p.detach() - old_params[n]
                    delta = delta
                    if p.grad is not None:  # In multi-head network, some head could have no grad (lazy) since no loss go through it.
                        self.w[n] -= p.grad * delta  # w[n] is >=0
            else:
                self.w = {}


            # statistics
            running_loss = beta * running_loss + (1 - beta) * self.loss.data.item()
            loss_corrected = running_loss / (1 - beta ** local_counter)

            current_time = time.time()
            duration = current_time - start_time
            running_duration = beta * running_duration + (1 - beta) * duration
            duration_corrected = running_duration / (1 - beta ** local_counter)
            examples_per_sec = self.train_batch_size / duration_corrected
            format_str = ('(E:%d, S:%d / %d) [Loss = %.4f] [Reg Loss = %.4f] (%.1f samples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (epoch, step + 1, num_steps_per_epoch, loss_corrected, reg_loss,
                                examples_per_sec, duration_corrected))

            local_counter += 1
            self.start_step = 0
            start_time = time.time()

        self.train_loss.append(loss_corrected)

        save_best = False
        if (epoch+1) % self.epochs_per_eval == 0:
        #if (not self.config.fc) & ((epoch + 1) % self.epochs_per_eval == 0):
            # evaluate after every other epoch
            test_results_srcc, val_results_srcc, all_hat_test, all_hat_val = self.eval()
            self.test_results_srcc['live'].append(test_results_srcc['live'])
            self.test_results_srcc['csiq'].append(test_results_srcc['csiq'])
            self.test_results_srcc['kadid10k'].append(test_results_srcc['kadid10k'])
            self.test_results_srcc['bid'].append(test_results_srcc['bid'])
            self.test_results_srcc['clive'].append(test_results_srcc['clive'])
            self.test_results_srcc['koniq10k'].append(test_results_srcc['koniq10k'])

            self.val_results_srcc['live'].append(val_results_srcc['live'])
            self.val_results_srcc['csiq'].append(val_results_srcc['csiq'])
            self.val_results_srcc['kadid10k'].append(val_results_srcc['kadid10k'])
            self.val_results_srcc['bid'].append(val_results_srcc['bid'])
            self.val_results_srcc['clive'].append(val_results_srcc['clive'])
            self.val_results_srcc['koniq10k'].append(val_results_srcc['koniq10k'])

            out_str = 'Testing: LIVE SRCC: {:.4f}  CSIQ SRCC: {:.4f}  BID SRCC: {:.4f}\n' \
                      'CLIVE SRCC: {:.4f}  KONIQ10K SRCC: {:.4f} KADID10K SRCC: {:.4f}'.format(
                test_results_srcc['live'],
                test_results_srcc['csiq'],
                test_results_srcc['bid'],
                test_results_srcc['clive'],
                test_results_srcc['koniq10k'],
                test_results_srcc['kadid10k'],
            )

            out_str2 = 'Validation: LIVE SRCC: {:.4f}  CSIQ SRCC: {:.4f} BID SRCC: {:.4f}\n' \
                       'CLIVE SRCC: {:.4f}  KONIQ10K SRCC: {:.4f} KADID10K SRCC: {:.4f}'.format(
                val_results_srcc['live'],
                val_results_srcc['csiq'],
                val_results_srcc['bid'],
                val_results_srcc['clive'],
                val_results_srcc['koniq10k'],
                val_results_srcc['kadid10k'],
            )

            print(out_str)
            print(out_str2)

            task = self.config.id2dataset[self.indicator-1]

            if self.config.JL:
                indicator = 0
                for idx, item in enumerate(val_results_srcc):
                    indicator += val_results_srcc[item]
                indicator = indicator / len(val_results_srcc)
            else:
                indicator = val_results_srcc[task]

            if indicator > self.best_srcc:
                self.best_srcc = indicator
                print('new best ! SRCC = {}'.format(self.best_srcc))
                save_best = True

        if (epoch == (self.config.max_epochs2-1)) & self.config.save_replay:
            # sampling for replay
            task_name = task
            replay_txt = task_name + '_train_score.txt'
            csv_file = os.path.join(self.config.trainset, 'splits2', str(self.config.split), replay_txt)
            replay_data = pd.read_csv(csv_file, sep='\t', header=None)
            num_data = len(replay_data)
            if self.config.task_id == 0:  # first, no previously stored sample
                #self.replayers = {}
                slice_idx = np.arange(0, num_data)
                np.random.shuffle(slice_idx)
                if num_data > self.config.replay_memory:
                    slice_idx = slice_idx[0:self.config.replay_memory]
                    replay_data = replay_data.iloc[slice_idx, :]
                self.replayers[task_name] = replay_data
            else:
                previous_length = 0
                for i in range(self.config.task_id):  # traverse previous task(s)
                    previous_name = self.config.id2dataset[i]
                    previous_length += len(self.replayers[previous_name])

                remaining_budget = self.config.replay_memory - previous_length

                if remaining_budget >= num_data: #have enough room, just save all data
                    self.replayers[task_name] = replay_data
                else:  # insufficient memory, randomly remove some samples from previous memory
                    current_task_num = self.config.task_id + 1  # number of tasks up till now
                    each_num = self.config.replay_memory // current_task_num
                    slice_idx = np.arange(0, each_num)
                    np.random.shuffle(slice_idx)

                    # handling previously stored samples
                    for i in range(self.config.task_id):
                        previous_name = self.config.id2dataset[i]
                        previous_data = self.replayers[previous_name]
                        previous_len = len(previous_data)
                        if previous_len > each_num:  # the i-th previous task has more samples train the average num
                            previous_slice = np.arange(0, previous_len)
                            np.random.shuffle(previous_slice)
                            previous_slice = previous_slice[0:each_num]
                            previous_data = previous_data.iloc[previous_slice, :]
                            self.replayers[previous_name] = previous_data
                        else:
                            continue  # do nothing if the i-th old task has less samples than the average num

                    # sampling on the current task
                    if num_data > each_num:
                        slice_idx = np.arange(0, num_data)
                        np.random.shuffle(slice_idx)
                        slice_idx = slice_idx[0:each_num]
                    replay_data = replay_data.iloc[slice_idx, :]
                    self.replayers[task_name] = replay_data

        if (epoch + 1) % self.epochs_per_save == 0:
            model_name = self.model_name
            model_name_latest = os.path.join(self.ckpt_path, model_name + '_latest.pt')
            #'scaler': self.scaler.state_dict()
            if self.config.amp:
                self._save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'train_loss': self.train_loss,
                    'test_results_srcc': self.test_results_srcc,
                    'val_results_srcc': self.val_results_srcc,
                    'best_srcc': self.best_srcc,
                    'regularization_terms': self.regularization_terms,
                    'replayers': self.replayers,
                    'scaler': self.scaler.state_dict(),
                    'w': self.w
                }, model_name_latest)

                if save_best:
                    model_name_best = os.path.join(self.ckpt_path, model_name + '_best.pt')
                    self._save_checkpoint({
                        'epoch': epoch,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'train_loss': self.train_loss,
                        'test_results_srcc': self.test_results_srcc,
                        'val_results_srcc': self.val_results_srcc,
                        'best_srcc': self.best_srcc,
                        'regularization_terms': self.regularization_terms,
                        'replayers': self.replayers,
                        'scaler': self.scaler.state_dict(),
                        'w': self.w
                    }, model_name_best)
                    if self.config.train_bn:
                        self.model.save_bn(name='best_bn.pt')
                elif (epoch == (self.config.max_epochs2 - 1)) & self.config.save_replay:
                    replayers_cached = self.replayers
                    model_name_best = os.path.join(self.ckpt_path, model_name + '_best.pt')
                    self._load_checkpoint(ckpt=model_name_best)
                    self._save_checkpoint({
                        'epoch': epoch,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'train_loss': self.train_loss,
                        'test_results_srcc': self.test_results_srcc,
                        'val_results_srcc': self.val_results_srcc,
                        'best_srcc': self.best_srcc,
                        'regularization_terms': self.regularization_terms,
                        'replayers': replayers_cached,
                        'scaler': self.scaler.state_dict(),
                        'w': self.w
                    }, model_name_best)
            else:
                self._save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'train_loss': self.train_loss,
                    'test_results_srcc': self.test_results_srcc,
                    'val_results_srcc': self.val_results_srcc,
                    'best_srcc': self.best_srcc,
                    'regularization_terms': self.regularization_terms,
                    'replayers': self.replayers,
                    'w': self.w
                }, model_name_latest)

                if save_best:
                    model_name_best = os.path.join(self.ckpt_path, model_name + '_best.pt')
                    self._save_checkpoint({
                        'epoch': epoch,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'train_loss': self.train_loss,
                        'test_results_srcc': self.test_results_srcc,
                        'val_results_srcc': self.val_results_srcc,
                        'best_srcc': self.best_srcc,
                        'regularization_terms': self.regularization_terms,
                        'replayers': self.replayers,
                        'w': self.w
                    }, model_name_best)
                    if self.config.train_bn:
                        self.model.save_bn(name='best_bn.pt')
                elif (epoch == (self.config.max_epochs2 - 1)) & self.config.save_replay:
                    replayers_cached = self.replayers
                    model_name_best = os.path.join(self.ckpt_path, model_name + '_best.pt')
                    self._load_checkpoint(ckpt=model_name_best)
                    self._save_checkpoint({
                        'epoch': epoch,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'train_loss': self.train_loss,
                        'test_results_srcc': self.test_results_srcc,
                        'val_results_srcc': self.val_results_srcc,
                        'best_srcc': self.best_srcc,
                        'regularization_terms': self.regularization_terms,
                        'replayers': replayers_cached,
                        'w': self.w
                    }, model_name_best)

        return self.loss.data.item()

    def process_one_batch(self, sample_batched):
        x, gmos, gstd = sample_batched['I'], sample_batched['mos'], sample_batched['std']
        gmos = Variable(gmos).view(-1, 1)
        gstd = Variable(gstd).view(-1, 1)
        x = x.to(self.device)
        gmos = gmos.to(self.device)
        gstd = gstd.to(self.device)
        pmos = self.model(x)
        return pmos, gmos, gstd

    def compute_features_single(self):
        print('Compute features')
        self.model.eval()

        N = len(self.kmeans_loader) # for processing image pairs
        print('extracting fc features for kmeans')
        for i, sample_batched in enumerate(tqdm(self.kmeans_loader)):
            x = sample_batched['I']

            x = Variable(x)
            x = x.to(self.device)

            bs = x.size(0) #batch size
            assert bs == 1

            if self.config.network == 'dbcnn':
                _, sfeat = self.model(x)
            else:
                _, feat = self.model(x)
                #sfeat, _ = self.scnn(x)

            if self.config.train_bn:
                feat1 = feat[0].cpu().numpy()
                if i == 0:
                    features1 = np.zeros((N, feat1.shape[1]), dtype='float32')
                feat1 = feat1.astype('float32')
                features1[i * bs: (i + 1) * bs, :] = feat1

                feat2 = feat[1].cpu().numpy()
                if i == 0:
                    features2 = np.zeros((N, feat2.shape[1]), dtype='float32')
                feat2 = feat2.astype('float32')
                features2[i * bs: (i + 1) * bs, :] = feat2

                feat3 = feat[2].cpu().numpy()
                if i == 0:
                    features3 = np.zeros((N, feat3.shape[1]), dtype='float32')
                feat3 = feat3.astype('float32')
                features3[i * bs: (i + 1) * bs, :] = feat3

                feat4 = feat[3].cpu().numpy()
                if i == 0:
                    features4 = np.zeros((N, feat4.shape[1]), dtype='float32')
                feat4 = feat4.astype('float32')
                features4[i * bs: (i + 1) * bs, :] = feat4

            elif self.config.network == 'dbcnn':
                sfeat = sfeat.cpu().numpy()
                if i == 0:
                    features = np.zeros((N, sfeat.shape[1]), dtype='float32')
                sfeat = sfeat.astype('float32')
                features[i * bs: (i + 1) * bs, :] = sfeat
            else:
                feat = feat.cpu().numpy()
                if i == 0:
                    features = np.zeros((N, feat.shape[1]), dtype='float32')
                feat = feat.astype('float32')
                features[i * bs: (i + 1) * bs, :] = feat

        if self.config.train_bn:
            features = []
            features.append(features1)
            features.append(features2)
            features.append(features3)
            features.append(features4)

        return features

    def select_head(self, task_id):
        return {
            '0': 'train_on_live',
            '1': 'train_on_csiq',
            '2': 'train_on_bid',
            '3': 'train_on_clive',
            '4': 'train_on_koniq10k',
            '5': 'train_on_kadid10k',
        }[task_id]

    def eval_single(self, dataloader, bn_caches=None):
        q_mos = []
        q_hat = []
        self.model.eval()

        assignment_weights = []
        for step, sample_batched in enumerate(dataloader, 0):
            x, y = sample_batched['I'], sample_batched['mos']

            x = Variable(x)
            x = x.to(self.device)

            #y_bar, _ = self.model(x)
            if (self.config.use_scnn) & (not self.config.train):
                if self.config.distortion_bn:
                    bn_path = os.path.join(self.config.ckpt_path, 'distortion_bn.pt')
                else:
                    bn_path = os.path.join(self.config.ckpt_path, 'imagenet_bn.pt')
                self.scnn.load_bn(bn_path)
                self.scnn.eval()
                _, feat = self.scnn(x)
                feat1 = feat[0]
                feat2 = feat[1]
                feat3 = feat[2]
                feat4 = feat[3]

            weights = []
            weights1 = []
            weights2 = []
            weights3 = []
            weights4 = []
            y_pred = torch.zeros(self.config.current_task_id+1)
            if self.config.weighted_output:
                assert self.config.current_task_id >= 1

                for i in range(self.config.current_task_id+1):
                    task_folder = 'train_on_' + self.config.id2dataset[i]

                    self.config.task_id = i
                    if self.config.train_bn:
                        bn_path = os.path.join(self.config.base_ckpt_path, task_folder, 'best_bn.pt')

                        self.model.load_bn_from_cache(bn_caches[self.config.id2dataset[i]])
                        self.model.eval()
                        y_bar, _ = self.model(x)

                        cluster_path = os.path.join(self.config.base_ckpt_path, task_folder, 'cluster_sk1.pt')
                        centroids = torch.load(cluster_path)
                        D = euclidean_distances(feat1.cpu().numpy(), centroids.numpy())
                        D = np.min(D)
                        D = torch.from_numpy(np.array(D)).cuda()
                        weights1.append(D)

                        cluster_path = os.path.join(self.config.base_ckpt_path, task_folder, 'cluster_sk2.pt')
                        centroids = torch.load(cluster_path)
                        D = euclidean_distances(feat2.cpu().numpy(), centroids.numpy())
                        D = np.min(D)
                        D = torch.from_numpy(np.array(D)).cuda()
                        weights2.append(D)

                        cluster_path = os.path.join(self.config.base_ckpt_path, task_folder, 'cluster_sk3.pt')
                        centroids = torch.load(cluster_path)
                        D = euclidean_distances(feat3.cpu().numpy(), centroids.numpy())
                        D = np.min(D)
                        D = torch.from_numpy(np.array(D)).cuda()
                        weights3.append(D)

                        cluster_path = os.path.join(self.config.base_ckpt_path, task_folder, 'cluster_sk4.pt')
                        centroids = torch.load(cluster_path)
                        D = euclidean_distances(feat4.cpu().numpy(), centroids.numpy())
                        D = np.min(D)
                        D = torch.from_numpy(np.array(D)).cuda()
                        weights4.append(D)

                        if self.config.backbone == 'resnet18_v':
                            y_pred[i] = y_bar[0]
                        else:
                            hidx = self.config.current2default[i]
                            y_pred[i] = y_bar[hidx][0][0]
                    else:
                        y_bar, feat = self.model(x)
                        cluster_path = os.path.join(self.config.base_ckpt_path, task_folder, 'cluster_sk.pt')

                        if self.config.use_scnn:
                            cluster_path = os.path.join(self.config.base_ckpt_path, task_folder, 'cluster_sk1.pt')
                            centroids = torch.load(cluster_path)
                            D = euclidean_distances(feat1.cpu().numpy(), centroids.numpy())
                            D = np.min(D)
                            D = torch.from_numpy(np.array(D)).cuda()
                            weights1.append(D)

                            cluster_path = os.path.join(self.config.base_ckpt_path, task_folder, 'cluster_sk2.pt')
                            centroids = torch.load(cluster_path)
                            D = euclidean_distances(feat2.cpu().numpy(), centroids.numpy())
                            D = np.min(D)
                            D = torch.from_numpy(np.array(D)).cuda()
                            weights2.append(D)

                            cluster_path = os.path.join(self.config.base_ckpt_path, task_folder, 'cluster_sk3.pt')
                            centroids = torch.load(cluster_path)
                            D = euclidean_distances(feat3.cpu().numpy(), centroids.numpy())
                            D = np.min(D)
                            D = torch.from_numpy(np.array(D)).cuda()
                            weights3.append(D)

                            cluster_path = os.path.join(self.config.base_ckpt_path, task_folder, 'cluster_sk4.pt')
                            centroids = torch.load(cluster_path)
                            D = euclidean_distances(feat4.cpu().numpy(), centroids.numpy())
                            D = np.min(D)
                            D = torch.from_numpy(np.array(D)).cuda()
                            weights4.append(D)
                        else:
                            centroids = torch.load(cluster_path)
                            D = euclidean_distances(feat.cpu().numpy(), centroids.numpy())
                            D = np.min(D)
                            D = torch.from_numpy(np.array(D)).cuda()
                            weights.append(D)

                        y_pred[i] = y_bar[i][0][0]

                if self.config.train_bn | self.config.use_scnn:
                    weights1 = torch.stack(weights1)
                    weights1 = F.softmin(weights1*32, dim=0)

                    weights2 = torch.stack(weights2)
                    weights2 = F.softmin(weights2*32, dim=0)

                    weights3 = torch.stack(weights3)
                    weights3 = F.softmin(weights3*32, dim=0)

                    weights4 = torch.stack(weights4)
                    weights4 = F.softmin(weights4*32, dim=0)

                    weights = (weights3 + weights4) / 2

                    if self.config.highest_output:
                        idx = torch.argmax(weights)
                        y_bar = y_pred[idx].cuda()
                    else:
                        y_bar = torch.dot(weights, y_pred.cuda())
                else:
                    weights = torch.stack(weights)
                    weights = F.softmin(weights * 32, dim=0)
                    y_bar = torch.dot(weights, y_pred.cuda())

                assignment_weights.append(weights.data)
            else:
                y_bar, _ = self.model(x)
                if self.config.train_bn:
                    head_index = self.config.current2default[self.config.task_id]
                    y_bar = y_bar[head_index]
                else:
                    y_bar = y_bar[self.config.task_id]
            y_bar.cpu()
            q_mos.append(y.data.numpy())
            q_hat.append(y_bar.cpu().data.numpy())

        if self.config.weighted_output:
            assignment_weights = torch.stack(assignment_weights, dim=0)
            assignment_weights = torch.mean(assignment_weights, dim=0)
            return q_mos, q_hat, assignment_weights
        else:
            return q_mos, q_hat

    def eval(self):
        srcc_val = {}
        srcc_test = {}
        self.model.eval()
        for task in self.config.dataset:
            srcc_val[task] = 0
            srcc_test[task] = 0
        bn_caches = {}
        ln_caches = {}
        if self.config.weighted_output & self.config.train_bn:
            for i in range(self.config.current_task_id+1):
                task_folder = 'train_on_' + self.config.id2dataset[i]
                self.config.task_id = i
                bn_path = os.path.join(self.config.base_ckpt_path, task_folder, 'best_bn.pt')
                bns = torch.load(bn_path)
                bn_caches[self.config.id2dataset[i]] = bns


        all_hat_test = {}
        all_hat_val = {}

        for idx, task in enumerate(self.config.dataset):
            # if not self.config.weighted_output:
            #     continue
            if self.config.eval_dict[task]:
                if self.config.weighted_output:
                    if self.config.amp:
                        with autocast():
                            q_mos, q_hat, weights = self.eval_single(self.task2loader_val[task], bn_caches)
                    else:
                        q_mos, q_hat, weights = self.eval_single(self.task2loader_val[task], bn_caches)


                    print('-------------Validation-----------')
                    print('assignment weights on {} is {}'.format(task, weights))
                    #print('least distance on {} is {}'.format(task, distances))
                else:
                    q_mos, q_hat = self.eval_single(self.task2loader_val[task])

                srcc_val[task] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
                all_hat_val[task] = q_hat

                #print('Validation: max/min predictions on {} are {} and {}'.format(task, np.max(q_hat), np.min(q_hat)))

                if self.config.weighted_output:
                    if self.config.amp:
                        with autocast():
                            q_mos, q_hat, weights = self.eval_single(self.task2loader[task], bn_caches)
                    else:
                        q_mos, q_hat, weights = self.eval_single(self.task2loader[task], bn_caches)

                    print('-------------Testing-----------')
                    print('assignment weights on {} is {}'.format(task, weights))
                    #print('least distance on {} is {}'.format(task, distances))
                else:
                    if self.config.amp:
                        with autocast():
                            q_mos, q_hat = self.eval_single(self.task2loader[task])
                    else:
                        q_mos, q_hat = self.eval_single(self.task2loader[task])

                srcc_test[task] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
                all_hat_test[task] = q_hat

                #print('Testing: max/min predictions on {} are {} and {}'.format(task, np.max(q_hat), np.min(q_hat)))

        return srcc_test, srcc_val, all_hat_test, all_hat_val


    def eval_each(self):
        srcc_test = {}
        srcc_val = {}
        all_hat_test = {}
        all_hat_val = {}
        self.model.eval()

        live_bn = os.path.join(self.config.base_ckpt_path, 'train_on_live', 'best_bn.pt')
        csiq_bn = os.path.join(self.config.base_ckpt_path, 'train_on_csiq', 'best_bn.pt')
        bid_bn = os.path.join(self.config.base_ckpt_path, 'train_on_bid', 'best_bn.pt')
        clive_bn = os.path.join(self.config.base_ckpt_path, 'train_on_clive', 'best_bn.pt')
        koniq10k_bn = os.path.join(self.config.base_ckpt_path, 'train_on_koniq10k', 'best_bn.pt')
        kadid10k_bn = os.path.join(self.config.base_ckpt_path, 'train_on_kadid10k', 'best_bn.pt')

        if self.config.task_id == 0:
            enforce_bn = live_bn
        elif self.config.task_id == 1:
            enforce_bn = csiq_bn
        elif self.config.task_id == 2:
            enforce_bn = bid_bn
        elif self.config.task_id == 3:
            enforce_bn = clive_bn
        elif self.config.task_id == 4:
            enforce_bn = koniq10k_bn
        elif self.config.task_id == 5:
            enforce_bn = kadid10k_bn


        if self.config.eval_dict['live']:
            self.config.task_id = self.config.dataset2id['live']
            if self.config.train_bn:
               if self.config.enforce_bn:
                   self.model.load_bn(enforce_bn)
               else:
                   self.model.load_bn(live_bn)

            q_mos, q_hat = self.eval_single(self.live_loader_val)
            srcc_val['live'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
            #print('Validation: max/min predictions on live are {} and {}'.format(np.max(q_hat), np.min(q_hat)))
            all_hat_val['live'] = q_hat
            q_mos, q_hat = self.eval_single(self.live_loader)
            srcc_test['live'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
            all_hat_test['live'] = q_hat
            #print('Testing: max/min predictions on live are {} and {}'.format(np.max(q_hat), np.min(q_hat)))
        else:
            srcc_test['live'] = 0
            srcc_val['live'] = 0

        if self.config.eval_dict['csiq']:
            self.config.task_id = self.config.dataset2id['csiq']
            if self.config.train_bn:
               if self.config.enforce_bn:
                   self.model.load_bn(enforce_bn)
               else:
                   self.model.load_bn(csiq_bn)

            q_mos, q_hat = self.eval_single(self.csiq_loader_val)
            srcc_val['csiq'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
            all_hat_val['csiq'] = q_hat
            #print('Validation: max/min predictions on csiq are {} and {}'.format(np.max(q_hat), np.min(q_hat)))

            q_mos, q_hat = self.eval_single(self.csiq_loader)
            srcc_test['csiq'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
            all_hat_test['csiq'] = q_hat
            #print('Testing: max/min predictions on csiq are {} and {}'.format(np.max(q_hat), np.min(q_hat)))
        else:
            srcc_test['csiq'] = 0
            srcc_val['csiq'] = 0

        if self.config.eval_dict['bid']:
            self.config.task_id = self.config.dataset2id['bid']
            if self.config.train_bn:
               if self.config.enforce_bn:
                   self.model.load_bn(enforce_bn)
               else:
                   self.model.load_bn(bid_bn)


            q_mos, q_hat = self.eval_single(self.bid_loader_val)
            srcc_val['bid'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
            all_hat_val['bid'] = q_hat
            #print('Validation: max/min predictions on bid are {} and {}'.format(np.max(q_hat), np.min(q_hat)))

            q_mos, q_hat = self.eval_single(self.bid_loader)
            all_hat_test['bid'] = q_hat

            srcc_test['bid'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
            srcc_val['bid'] = scipy.stats.mstats.pearsonr(x=q_mos, y=q_hat)[0]
            #print('Tesing: max/min predictions on bid are {} and {}'.format(np.max(q_hat), np.min(q_hat)))
        else:
            srcc_test['bid'] = 0
            srcc_val['bid'] = 0

        if self.config.eval_dict['clive']:
            self.config.task_id = self.config.dataset2id['clive']
            if self.config.train_bn:
               if self.config.enforce_bn:
                   self.model.load_bn(enforce_bn)
               else:
                   self.model.load_bn(clive_bn)

            q_mos, q_hat = self.eval_single(self.clive_loader_val)
            srcc_val['clive'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
            all_hat_val['clive'] = q_hat
            #print('Validation: max/min predictions on clive are {} and {}'.format(np.max(q_hat), np.min(q_hat)))


            q_mos, q_hat = self.eval_single(self.clive_loader)
            srcc_test['clive'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
            all_hat_test['clive'] = q_hat
            #print('Testing: max/min predictions on clive are {} and {}'.format(np.max(q_hat), np.min(q_hat)))
        else:
            srcc_test['clive'] = 0
            srcc_val['clive'] = 0
       # ####prepare for replay
        if self.config.eval_dict['koniq10k']:
            self.config.task_id = self.config.dataset2id['koniq10k']
            if self.config.train_bn:
               if self.config.enforce_bn:
                   self.model.load_bn(enforce_bn)
               else:
                   self.model.load_bn(koniq10k_bn)

            q_mos, q_hat = self.eval_single(self.koniq10k_loader_val)
            srcc_val['koniq10k'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
            all_hat_val['koniq10k'] = q_hat
            #print('Validation: max/min predictions on koniq10k are {} and {}'.format(np.max(q_hat), np.min(q_hat)))

            q_mos, q_hat = self.eval_single(self.koniq10k_loader)
            srcc_test['koniq10k'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
            all_hat_test['koniq10k'] = q_hat
            #print('Testing: max/min predictions on koniq10k are {} and {}'.format(np.max(q_hat), np.min(q_hat)))
        else:
            srcc_test['koniq10k'] = 0
            srcc_val['koniq10k'] = 0


        if self.config.eval_dict['kadid10k']:
            self.config.task_id = self.config.dataset2id['kadid10k']
            if self.config.train_bn:
               if self.config.enforce_bn:
                   self.model.load_bn(enforce_bn)
               else:
                   self.model.load_bn(kadid10k_bn)


            q_mos, q_hat = self.eval_single(self.kadid10k_loader_val)
            srcc_val['kadid10k'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
            all_hat_val['kadid10k'] = q_hat
            #print('Validation: max/min predictions on kadid10k are {} and {}'.format(np.max(q_hat), np.min(q_hat)))

            q_mos, q_hat = self.eval_single(self.kadid10k_loader)
            srcc_test['kadid10k'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
            all_hat_test['kadid10k'] = q_hat
            #print('Testing: max/min predictions on kadid10k are {} and {}'.format(np.max(q_hat), np.min(q_hat)))
        else:
            srcc_test['kadid10k'] = 0
            srcc_val['kadid10k'] = 0

        return srcc_test, srcc_val, all_hat_test, all_hat_val

    def kmeans_each(self):
        self.train_kmeans()


    def get_scores_single(self, dataloader, task_id):
        q_mos = []
        q_hat = []
        for step, sample_batched in enumerate(dataloader, 0):
            x, y, std = sample_batched['I'], sample_batched['mos'], sample_batched['std']
            x = Variable(x)
            x = x.to(self.device)

            y_bar, _ = self.model(x)
            y_bar[task_id].cpu()
            q_mos.append(y.data.numpy())
            q_hat.append(y_bar[task_id].cpu().data.numpy())

        return q_mos, q_hat

    def get_scores(self):
        all_mos = {}
        all_hat = {}
        self.model.eval()

        if self.config.task_id >= 0:
            q_mos, q_hat = self.get_scores_single(self.live_loader, 0)
            all_mos['live'] = q_mos
            all_hat['live'] = q_hat
        else:
            all_mos['live'] = 0
            all_hat['live'] = 0

        if self.config.task_id >= 1:
            q_mos, q_hat = self.get_scores_single(self.csiq_loader, 1)
            all_mos['csiq'] = q_mos
            all_hat['csiq'] = q_hat
        else:
            all_mos['csiq'] = 0
            all_hat['csiq'] = 0

        if self.config.task_id >= 2:
            q_mos, q_hat = self.get_scores_single(self.bid_loader, 2)
            all_mos['bid'] = q_mos
            all_hat['bid'] = q_hat
        else:
            all_mos['bid'] = 0
            all_hat['bid'] = 0

        if self.config.task_id >= 3:
            q_mos, q_hat = self.get_scores_single(self.clive_loader, 3)
            all_mos['clive'] = q_mos
            all_hat['clive'] = q_hat
        else:
            all_mos['clive'] = 0
            all_hat['clive'] = 0

        if self.config.task_id >= 4:
            q_mos, q_hat = self.get_scores_single(self.koniq10k_loader, 4)
            all_mos['koniq10k'] = q_mos
            all_hat['koniq10k'] = q_hat
        else:
            all_mos['koniq10k'] = 0
            all_hat['koniq10k'] = 0

        if self.config.task_id >= 5:
            q_mos, q_hat = self.get_scores_single(self.kadid10k_loader, 5)
            all_mos['kadid10k'] = q_mos
            all_hat['kadid10k'] = q_hat
        else:
            all_mos['kadid10k'] = 0
            all_hat['kadid10k'] = 0


        return all_mos, all_hat


    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            print("[*] loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt)

            if self.config.amp:
                self.scaler.load_state_dict(checkpoint['scaler'])

            self.start_epoch = checkpoint['epoch'] + 1
            self.train_loss = checkpoint['train_loss']
            self.test_results_srcc = checkpoint['test_results_srcc']
            self.val_results_srcc = checkpoint['val_results_srcc']
            if not self.config.train:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if self.config.train:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_srcc = checkpoint['best_srcc']
            if self.config.reg_trigger:
                self.regularization_terms = checkpoint['regularization_terms']
            if self.config.SI:
                self.w = checkpoint['w']
            self.replayers = checkpoint['replayers']
            print("[*] loaded checkpoint '{}' (epoch {})"
                  .format(ckpt, checkpoint['epoch']))
        else:
            print("[!] no checkpoint found at '{}'".format(ckpt))

    @staticmethod
    def _get_latest_checkpoint(path):
        ckpts = os.listdir(path)
        ckpts = [ckpt for ckpt in ckpts if not os.path.isdir(os.path.join(path, ckpt))]
        all_times = sorted(ckpts, reverse=True)
        return os.path.join(path, all_times[0])

    # @staticmethod
    def _get_checkpoint_new(self, path, resume_best):
        #ckpts = os.listdir(path)
        if self.config.train_bn:
            if self.config.network == 'dbcnn':
                if resume_best:
                    ckpt = 'DBCNN_bn_best.pt'
                else:
                    ckpt = 'DBCNN_bn_latest.pt'
            else:
                if resume_best:
                    ckpt = 'BaseCNN_bn_best.pt'
                else:
                    ckpt = 'BaseCNN_bn_latest.pt'
        else:
            if resume_best:
                if self.config.network == 'dbcnn':
                    ckpt = 'DBCNN_best.pt'
                else:
                    ckpt = 'BaseCNN_vanilla_best.pt'
            else:
                if self.config.network == 'dbcnn':
                    ckpt = 'DBCNN_latest.pt'
                else:
                    ckpt = 'BaseCNN_vanilla_latest.pt'
        return os.path.join(path, ckpt)

    # save checkpoint
    @staticmethod
    def _save_checkpoint(state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)

    ##############implmentaions of other continual learning methods##################
    def reg_criterion(self):
        reg_loss = 0
        if (self.config.reg_trigger) & (len(self.regularization_terms)>0):
            # Calculate the reg_loss only when the regularization_terms exists
            for i,reg_term in self.regularization_terms.items():
                task_reg_loss = 0
                importance = reg_term['importance']
                task_param = reg_term['task_param']
                for n, p in self.params.items():
                    reg_loss_all = (importance[n] * (p - task_param[n]) ** 2)
                    if torch.isnan(reg_loss_all).any():
                        reg_loss_all = reg_loss_all.to('cpu').detach().numpy()
                        reg_loss_all = np.array(np.nansum(reg_loss_all))
                        #task_reg_loss.data = torch.from_numpy(reg_loss_all).to(self.device)
                        task_reg_loss = torch.from_numpy(reg_loss_all).to(self.device)
                    else:
                        task_reg_loss += reg_loss_all.sum()
                reg_loss += task_reg_loss
            reg_loss = self.config.reg_weight * reg_loss
        return reg_loss

    def calculate_importance(self):
        # Initialize the importance matrix
        if self.config.SI:
            assert self.online_reg
            # Initialize the importance matrix
            if len(self.regularization_terms) > 0:  # The case of after the first task
                importance = self.regularization_terms[1]['importance']
                prev_params = self.regularization_terms[1]['task_param']
            else:  # It is in the first task
                importance = {}
                for n, p in self.params.items():
                    importance[n] = p.clone().detach().fill_(0)  # zero initialized
                prev_params = self.initial_params

            # Calculate or accumulate the Omega (the importance matrix)
            for n, p in importance.items():
                delta_theta = self.params[n].detach() - prev_params[n]
                p += self.w[n] / (delta_theta ** 2 + self.damping_factor)
                self.w[n].zero_()
        elif self.config.MAS:
            assert self.online_reg
            if (self.online_reg) & (len(self.regularization_terms) > 0):
                importance = self.regularization_terms[1]['importance']
            else:
                importance = {}
                for n, p in self.params.items():
                    importance[n] = p.clone().detach().fill_(0)  # zero initialized
            # Set model to evaluation mode
            mode = self.model.training
            self.model.eval()
            self.model.backbone.apply(fix_bn)

            # Create data-loader to give batches of size 1
            data_loader = DataLoader(self.train_data,
                                     batch_size=1,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=12)

            for step, sample_batched in enumerate(tqdm(data_loader)):

                if step < self.start_step:
                    continue

                x1, x2, g, _, _, yb = sample_batched['I1'], sample_batched['I2'], sample_batched['y'], \
                                           sample_batched['std1'], sample_batched['std2'], sample_batched['yb']
                x1 = Variable(x1)
                x2 = Variable(x2)
                g = Variable(g).view(-1, 1)
                yb = Variable(yb).view(-1, 1)
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                g = g.to(self.device)
                yb = yb.to(self.device)

                pred = self.do_batch(x1, x2)

                pred = pred[self.config.task_id]
                pred.pow_(2)
                loss = pred.mean()
                # Calculate gradient
                self.model.zero_grad()
                loss.backward()

                for n, p in importance.items():
                    if self.params[n].grad is not None:  # Some heads can have no grad if no loss applied on them.
                        p += (self.params[n].grad.abs() / len(data_loader))

                self.model.train(mode=mode)
        else:
            if self.online_reg and len(self.regularization_terms) > 0:
                importance = self.regularization_terms[1]['importance']
            else:
                importance = {}
                for n, p in self.params.items():
                    importance[n] = p.clone().detach().fill_(0)  # zero initialized

            # Set model to evaluation mode
            mode = self.model.training
            self.model.eval()
            self.model.backbone.apply(fix_bn)

            # Create data-loader to give batches of size 1
            data_loader = DataLoader(self.train_data,
                                     batch_size=1,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=12)

            for step, sample_batched in enumerate(tqdm(data_loader)):

                if step < self.start_step:
                    continue

                x1, x2, g, _, _, yb = sample_batched['I1'], sample_batched['I2'], sample_batched['y'], \
                                           sample_batched['std1'], sample_batched['std2'], sample_batched['yb']
                x1 = Variable(x1)
                x2 = Variable(x2)
                g = Variable(g).view(-1, 1)
                yb = Variable(yb).view(-1, 1)
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                g = g.to(self.device)
                yb = yb.to(self.device)

                pred = self.do_batch(x1, x2)

                if self.config.fidelity:
                    loss = self.loss_fn(pred[self.config.task_id], g.detach())
                else:
                    loss = self.loss_fn(pred[self.config.task_id], yb.detach())

                # Calculate gradient
                self.model.zero_grad()
                loss.backward()

                for n, p in importance.items():
                    if self.params[n].grad is not None:  # Some heads can have no grad if no loss applied on them.
                        p += ((self.params[n].grad ** 2) / len(data_loader))

            self.model.train(mode=mode)

        return importance

