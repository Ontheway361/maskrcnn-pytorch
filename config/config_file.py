#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: lujie
Created on 2019/07/05
"""

import os
import argparse
from IPython import embed

root_path = '/home/lujie/Documents/deep_learning'   # local-pc
# root_path = '/home/gpu3/data/relu'   # gpu-server
data_path = os.path.join(root_path, 'benchmark_images/coco')
save_path = os.path.join(root_path, 'saved_model/detection')

def parameters_setting():

    parser = argparse.ArgumentParser('Config for coco_runner ')

    # about I/O
    parser.add_argument('--dataset',    type=str, default='coco')
    parser.add_argument('--num_classes',type=int, default=80)
    parser.add_argument('--data_path',  type=str, default=data_path)
    parser.add_argument('--output_dir', type=str, default=save_path)

    # about model
    parser.add_argument('--model',     type=str,  default='fasterrcnn_resnet50_fpn')
    parser.add_argument('--resume',    type=str,  default=None)
    parser.add_argument("--test_only", type=bool, default=False)
    parser.add_argument("--pretrained",type=bool, default=False)

    # about hardware
    parser.add_argument('--device',     type=str,  default='cpu')   # TODO
    parser.add_argument('--gpus_id',    type=list, default=[0,1])    # TODO
    parser.add_argument('--workers',    type=int,  default=8)
    parser.add_argument('--world-size', type=int,  default=1)
    parser.add_argument('--dist-url',   type=str,  default='env://')

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
    parser.add_argument('--aspect-ratio-group-factor', type=int, default=0)


    args = parser.parse_args()

    return args
