#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: lujie
Created on 2019/07/05
"""

import os
import time
import datetime
import torch
from torch import nn
import torch.utils.data
from detectlib import fasterrcnn_resnet50_fpn

import reference.utils as utils
import reference.transforms as T
from config import parameters_setting
from reference.coco_utils import get_coco, get_coco_kp
from reference.engine import train_one_epoch, evaluate
from reference import GroupedBatchSampler, create_aspect_ratio_groups

from IPython import embed

class CoCoRunner(object):

    def __init__(self, args):

        self.args          = args
        self.model_details = dict()
        self.datas_details = dict()


    def _get_dataset(self, image_set='train', transform = None):

        paths = {
            "coco"    : (self.args.data_path, get_coco, 91),
            "coco_kp" : (self.args.data_path, get_coco_kp, 2)
        }
        p, ds_fn, _ = paths[self.args.dataset]

        ds = ds_fn(p, image_set=image_set, transforms=transform)
        return ds


    @staticmethod
    def get_transform(train = True):
        ''' Set the transform for coco '''

        transforms = []
        transforms.append(T.ToTensor())
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)


    def _modelsetting(self):
        '''
        Module loading and optimizer setting

        step - 1. loading the model and set the device status
        step - 2. define the optimizer and lr_scheduler
        setp - 3. if use resume_checkpoint, laoding the checkpoint
        '''

        # step - 1
        self.model_details['device'] = torch.device(self.args.device)
        self.model_details['model']  = fasterrcnn_resnet50_fpn(
                                           num_classes=self.args.num_classes,
                                           pretrained=self.args.pretrained)
        self.model_details['model'].to(self.model_details['device'])

        if self.args.distributed:
            '''
            # multi-machines && multi-gpus
            self.model_details['model'] = torch.nn.parallel.DistributedDataParallel(\
                                              self.model_details['model'], device_ids=self.gpus_id)
            '''
            # single-machine && multi-gpus
            self.model_details['model'] = torch.nn.DataParallel(self.model_details['model'], device_ids=self.args.gpus).cuda()

            self.model_details['model_without_ddp'] = self.model_details['model'].module
        else:
            self.model_details['model_without_ddp'] = self.model_details['model']

        # step - 2
        params = [p for p in self.model_details['model'].parameters() if p.requires_grad]
        self.model_details['optimizer'] = torch.optim.SGD(params, lr=self.args.lr, \
                                              momentum=self.args.momentum, weight_decay=self.args.weight_decay)

        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_steps, gamma=args.lr_gamma)
        self.model_details['lr_scheduler'] = torch.optim.lr_scheduler.MultiStepLR(self.model_details['optimizer'],\
                                                 milestones=self.args.lr_steps, gamma=self.args.lr_gamma)

        # step - 3
        if self.args.resume:
            checkpoint = torch.load(self.args.resume, map_location='cpu')
            self.model_details['model_without_ddp'].load_state_dict(checkpoint['state_dict'])
            self.model_details['optimizer'].load_state_dict(checkpoint['optimizer'])
            self.model_details['lr_scheduler'].load_state_dict(checkpoint['lr_scheduler'])

        print('model setting is finished ...')


    def _dataloader(self):

        dataset      = self._get_dataset(image_set='train', transform=self.get_transform(train=True))
        dataset_test = self._get_dataset(image_set='val',   transform=self.get_transform(train=False))

        if self.args.distributed:
            self.datas_details['train_sampler'] = torch.utils.data.distributed.DistributedSampler(dataset)
            self.datas_details['test_sampler']  = torch.utils.data.distributed.DistributedSampler(dataset_test)
        else:
            self.datas_details['train_sampler'] = torch.utils.data.RandomSampler(dataset)
            self.datas_details['test_sampler']  = torch.utils.data.SequentialSampler(dataset_test)

        if self.args.aspect_ratio_group_factor >= 0:
            group_ids = create_aspect_ratio_groups(dataset, k=self.args.aspect_ratio_group_factor)
            train_batch_sampler = GroupedBatchSampler(self.datas_details['train_sampler'], group_ids, self.args.batch_size)
        else:
            train_batch_sampler = torch.utils.data.BatchSampler(self.datas_details['train_sampler'], \
                                      self.args.batch_size, drop_last=True)

        self.datas_details['train'] = torch.utils.data.DataLoader(dataset, batch_sampler=train_batch_sampler, \
                                          num_workers=self.args.workers, collate_fn=utils.collate_fn)

        self.datas_details['eval']  = torch.utils.data.DataLoader(dataset_test, batch_size=1, sampler=self.datas_details['test_sampler'], \
                                          num_workers=self.args.workers, collate_fn=utils.collate_fn)
        print(" data loading finished ... ")

        if self.args.test_only:
            evaluate(self.model_details['model'], self.datas_details['eval'], device=self.model_details['device'])
            return


    def _main_loop(self):

        print("Start training")
        start_time = time.time()

        for epoch in range(self.args.epochs):

            if self.args.distributed:
                self.datas_details['train_sampler'].set_epoch(epoch)

            train_one_epoch(self.model_details['model'], self.model_details['optimizer'], self.datas_details['train'], \
                            self.model_details['device'], epoch, self.args.print_freq)

            self.model_details['lr_scheduler'].step()

            if self.args.output_dir:

                if not os.path.exists(self.args.output_dir):
                    os.mkdir(os.args.output_dir)

                utils.save_on_master({
                    'state_dict': self.model_details['model_without_ddp'].state_dict(),
                    'optimizer': self.model_details['optimizer'].state_dict(),
                    'lr_scheduler': self.model_details['lr_scheduler'].state_dict(),
                    'args': self.args},
                    os.path.join(self.args.output_dir, 'model_{}.pth'.format(epoch)))

            # evaluate after every epoch
            evaluate(self.model_details['model'], self.datas_details['eval'], device=self.model_details['device'])

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def runner(self):
        '''
        Pipeline of detector

        step - 1. set the environ of os
        step - 2. model setting module
        step - 3. data loading module
        step - 4. train/eval main loop
        '''

        # step - 1
        utils.init_distributed_mode(args)

        # step - 2
        self._modelsetting()

        # step - 3
        self._dataloader()

        # step - 4
        if not self.args.test_only:
            self._main_loop()


if __name__ == "__main__":

    args = parameters_setting()
    detector = CoCoRunner(args)
    detector.runner()
