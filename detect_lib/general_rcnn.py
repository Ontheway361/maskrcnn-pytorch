#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
from collections import OrderedDict


class GeneralizedRCNN(nn.Module):
    ''' Outline of GeneralizedRCNN '''

    def __init__(self, backbone, rpn, roi, transform):

        super(GeneralizedRCNN, self).__init__()

        self.transform = transform
        self.backbone  = backbone
        self.rpn       = rpn
        self.roi_heads = roi


    def forward(self, images, targets=None):

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        original_image_sizes = [img.shape[-2:] for img in images]

        images, targets = self.transform(images, targets)

        features = self.backbone(images.tensors)

        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])

        proposals, rpn_losses = self.rpn(images, features, targets)

        detections, roi_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(rpn_losses)
        losses.update(roi_losses)

        if self.training:
            return losses

        return detections
