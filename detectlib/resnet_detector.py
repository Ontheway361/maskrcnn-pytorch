#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2019/09/08
@author: relu
"""

from torch import nn
from basenets import resnet
from collections import OrderedDict
from detectlib import FasterRCNN
from modules import FPN, RPN, RoI
from utils.misc import FrozenBatchNorm2d

from utils.transform import GeneralTrans
from basenets._utils import IntermediateLayerGetter
from basenets.utils import load_state_dict_from_url

from IPython import embed


__all__ = ["fasterrcnn_resnet50_fpn"]


class BackboneWithFPN(nn.Sequential):
    ''' Return the dict of feature_map of selected layers '''
    # Note : nn.Sequential

    def __init__(self, basebone, return_layers, in_channels_list, out_channels):

        # pick up the feature_map whose name in return_layers
        body = IntermediateLayerGetter(basebone, return_layers=return_layers)
        fpn  = FPN(in_channels_list, out_channels, with_tmp=True)

        super(BackboneWithFPN, self).__init__(OrderedDict([("body", body), ("fpn", fpn)]))

        self.out_channels = out_channels


class FasterRCNN_Resnet(FasterRCNN):

    def __init__(self, num_classes, basenet = 'resnet50', pretrain = True):

        gener_trans = GeneralTrans()
        backbone    = self._backbone(basenet)
        rpn         = RPN(backbone.out_channels)
        roi_heads   = RoI(num_classes, backbone.out_channels)
        
        super(FasterRCNN_Resnet, self).__init__(backbone, rpn, roi_heads, gener_trans)


    def _backbone(self, basenet, with_fpn = True):
        ''' Generate the backbone according to basenet
        # body : resnet-50 truncated before fc
        # fcn  : feature pyramid
        '''

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


def fasterrcnn_resnet50_fpn(num_classes = 91, basenet = 'resnet50', with_fpn = True, pretrained = True):

    if pretrained:
        basenet = 'resnet50'
        cp_urls = 'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'

    model = FasterRCNN_Resnet(num_classes, basenet)

    if pretrained:
        state_dict = load_state_dict_from_url(cp_urls, progress=True)
        model.load_state_dict(state_dict)

    return model
