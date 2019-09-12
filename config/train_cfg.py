#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2019/09/08
@author: relu
"""

import os
import argparse
from IPython import embed

root_path = '/Users/g/Documents/relu/'   # local-pc
# root_path = '/home/gpu3/data/relu'   # gpu-server
data_path = os.path.join(root_path, 'benchmark_images/coco')
save_path = os.path.join(root_path, 'saved_model/detection/fasterrcnn')

def train_setting():

    parser = argparse.ArgumentParser('PyTorch FasterRCNN Detection Training')

    # about I/O
    parser.add_argument('--dataset',    type=str, default='coco')
    parser.add_argument('--num_classes',type=int, default=91)
    parser.add_argument('--data_path',  type=str, default=data_path)
    parser.add_argument('--output_dir', type=str, default=save_path)

    # about model
    parser.add_argument('--basenet',   type=str,  default='resnet50')
    parser.add_argument('--with_fpn',  type=str,  default=True)
    parser.add_argument('--checkpoint',type=str,  default='maskrcnn_resnet50_fpn_coco_org.pth',
                        choices=['fasterrcnn_resnet50_fpn_coco_org.pth', 'maskrcnn_resnet50_fpn_coco_org.pth'])
    parser.add_argument("--test_only", type=bool, default=False)

    # about hardware
    parser.add_argument('--device',     type=str,  default='cpu')   # TODO
    parser.add_argument('--gpus_id',    type=list, default=[0,1])    # TODO
    parser.add_argument('--workers',    type=int,  default=4)


    # about optimizer
    parser.add_argument('--lr',          type=float, default=0.02)
    parser.add_argument('--momentum',    type=float, default=0.9)
    parser.add_argument('--weight_decay',type=float, default=1e-4)
    parser.add_argument('--lr_steps',    type=list,  default=[8, 11])
    parser.add_argument('--lr_gamma',    type=float, default=0.1)

    # about train/eval
    parser.add_argument('--epochs',    type=int, default=20)
    parser.add_argument('--batch_size',type=int, default=2)
    parser.add_argument('--print_freq',type=int, default=10000)

    args = parser.parse_args()

    return args
