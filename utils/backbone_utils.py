#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn
from modules import FPN
from basenets import resnet
from collections import OrderedDict
from utils.misc import FrozenBatchNorm2d
from basenets._utils import IntermediateLayerGetter


class BackboneWithFPN(nn.Sequential):
    ''' Return the dict of feature_map of selected layers '''
    # Note : nn.Sequential

    def __init__(self, basebone, return_layers, in_channels_list, out_channels):

        # pick up the feature_map whose name in return_layers
        body = IntermediateLayerGetter(basebone, return_layers=return_layers)
        fpn  = FPN(in_channels_list, out_channels, with_tmp=True)

        super(BackboneWithFPN, self).__init__(OrderedDict([("body", body), ("fpn", fpn)]))

        self.out_channels = out_channels


def build_backbone(basenet, with_fpn = True):

    basebone = resnet.__dict__[basenet](pretrained=False, norm_layer=FrozenBatchNorm2d)

    # freeze the first layer
    for name, parameter in basebone.named_parameters():
        if ('layer2' not in name) and ('layer3' not in name) and ('layer4' not in name):
            parameter.requires_grad_(False)

    return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}
    in_channels_stage2, out_channels = 256, 256
    in_channels_list = [in_channels_stage2 * i for i in [1, 2, 4, 8]]
    backbone = BackboneWithFPN(basebone, return_layers, in_channels_list, out_channels)

    return backbone
