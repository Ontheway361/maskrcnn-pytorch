#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn
from modules import RPN, RoI
from collections import OrderedDict
from detect_lib import GeneralizedRCNN
from utils.transform import GeneralTrans
from utils.backbone_utils import build_backbone
from basenets.utils import load_state_dict_from_url

from IPython import embed

class MaskRCNN_Resnet(GeneralizedRCNN):

    def __init__(self, num_classes, basenet = 'resnet50', with_fpn = True):

        gener_trans = GeneralTrans()
        backbone    = build_backbone(basenet, with_fpn)
        rpn         = RPN(backbone.out_channels)
        roi         = RoI(num_classes, backbone.out_channels, 'segment')

        super(MaskRCNN_Resnet, self).__init__(backbone, rpn, roi, gener_trans)



def maskrcnn_resnet50_fpn(num_classes = 91, basenet = 'resnet50', with_fpn = True, pretrained = True):

    if pretrained:
        basenet = 'resnet50'
        cp_urls = 'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth'

    model = MaskRCNN_Resnet(num_classes, basenet)

    if pretrained:
        state_dict = load_state_dict_from_url(cp_urls, progress=True)
        model.load_state_dict(state_dict)

    return model
