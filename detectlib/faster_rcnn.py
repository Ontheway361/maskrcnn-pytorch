#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2019/09/08
@author: relu
"""

import torch
from torch import nn
from collections import OrderedDict


class FasterRCNN(nn.Module):
    ''' Outline of FasterRCNN '''

    def __init__(self, backbone, rpn, roi_heads, transform):

        super(FasterRCNN, self).__init__()

        self.transform = transform
        self.backbone  = backbone
        self.rpn       = rpn
        self.roi_heads = roi_heads


    def forward(self, images, targets=None):

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        original_image_sizes = [img.shape[-2:] for img in images]

        images, targets = self.transform(images, targets)

        features = self.backbone(images.tensors)

        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])

        proposals, proposal_losses = self.rpn(images, features, targets)

        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            return losses

        return detections
