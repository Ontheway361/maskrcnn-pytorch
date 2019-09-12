#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import shutil
from modules import RPN_CFG
from dataset import get_coco
import dataset.transforms as T
from config  import train_setting
from tensorboardX import SummaryWriter
from detect_lib import FasterRCNN_Resnet

from IPython import embed

class DetectTrain(object):

    def __init__(self, args):

        self.args      = args
        self.model     = None
        self.optim     = None
        self.lr_adj    = None
        self.data      = dict()

        self.report_keyinfo(args)
        self.environ_set(args)


    @staticmethod
    def report_keyinfo(args):
        print(('''
            Parameters for environ and model.
            model Configurations:
                dataset     :   {}
                model_info  :   {}
                aspr        :   {}
                epochs      :   {}
                batch_size  :   {}
                workers     :   {}
                device      :   {}
                    '''.format(args.dataset, args.basenet, RPN_CFG['aspect_ratio'], \
                               args.epochs, args.batch_size, args.workers, args.device)))


    @staticmethod
    def environ_set(args):

        if torch.cuda.is_available() and args.device == 'cuda':
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')


    @staticmethod
    def trans(is_training = True):

        transforms = []
        transforms.append(T.ToTensor())
        if is_training:
            transforms.append(T.RandomHorizontalFlip(0.5))

        return T.Compose(transforms)


    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


    def _dataloader(self):

        self.data['train_loader'] = torch.utils.data.DataLoader(
                                        get_coco(self.args.data_path, 'val', self.trans()),
                                        batch_size=self.args.batch_size,
                                        num_workers=self.args.workers,
                                        shuffle=True, drop_last=True,
                                        collate_fn=self.collate_fn)

        # self.data['val_loader']   = torch.utils.data.DataLoader(
        #                                 get_coco(self.args.data_path, 'val', self.trans(False)),
        #                                 batch_size=self.args.batch_size,
        #                                 num_workers=self.args.workers,
        #                                 shuffle=False, drop_last=False,
        #                                 collate_fn=self.collate_fn)
        print('dataset loading was finished ...')


    def _modelloader(self):

        self.model = FasterRCNN_Resnet(
                         num_classes=self.args.num_classes,
                         basenet=self.args.basenet,
                         with_fpn=self.args.with_fpn)

        assert self.args.checkpoint is not None
        cp_path = os.path.join(self.args.output_dir, self.args.checkpoint)
        self.model.load_state_dict(torch.load(cp_path, map_location=lambda storage, loc: storage))

        if torch.cuda.is_available() and self.args.device == 'cuda':
            self.model = self.model.to(torch.device(self.args.device))
            torch.backends.cudnn.benchmark = True
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpus_id)

        print('model loading was finished ...')


    def _optimizer_setting(self):

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optim  = torch.optim.SGD(params, lr=self.args.lr, \
                          momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        self.lr_adj = torch.optim.lr_scheduler.MultiStepLR(self.optim, \
                          milestones=self.args.lr_steps, gamma=self.args.lr_gamma)

        print('optimizer was ready ...')


    @staticmethod
    def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

        def f(x):
            if x >= warmup_iters:
                return 1
            alpha = float(x) / warmup_iters
            return warmup_factor * (1 - alpha) + alpha

        return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


    def _train_step(self, epoch):
        ''' Train unit for a single epoch '''

        self.model.train()

        lr_scheduler = None

        if epoch == 0:
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(self.data['train_loader']) - 1)
            lr_scheduler = self.warmup_lr_scheduler(self.optim, warmup_iters, warmup_factor)

        for iter, (imgs, targets) in enumerate(self.data['train_loader']):

            imgs = list(img.to(self.args.device) for img in imgs)
            targets = [{k: v.to(self.args.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(imgs, targets)

            losses = sum(loss for loss in loss_dict.values())

            embed()

            self.optim.zero_grad()
            losses.backward()
            self.optim.step()

            if lr_scheduler is not None:
                lr_scheduler.step()


    def _eval_step(self, epoch):

        self.model.eval()
        cpu_device = torch.device("cpu")

        for iter, (imgs, targets) in enumerate(self.data['val_loader']):

            imgs = list(img.to(self.args.device) for img in imgs)
            targets = [{k: v.to(self.args.device) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                outputs = self.model(imgs)
                outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
                res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}


    def _main_loop(self):

        for epoch in range(self.args.epochs):

            self._train_step(epoch)

            # self.lr_adj.step()
            #
            # self._eval_step(epoch)


    def runner(self):

        self._dataloader()
        self._modelloader()
        embed()
        self._optimizer_setting()
        self._main_loop()


if __name__ == '__main__':

    detector = DetectorTrain(train_setting())
    detector.runner()
