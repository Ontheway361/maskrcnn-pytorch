#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2019/09/08
@author: relu
"""

import torch
from torch import nn
from torch.nn import functional as F

from modules.cfg import RPN_CFG
import utils.bbox_utils as box_utils
import utils.rpn_utils  as rpn_utils

from IPython import embed

class RPN(nn.Module):

    def __init__(self, out_channels = 256):

        super(RPN, self).__init__()

        self.anchor_generator = rpn_utils.AnchorGenerator(
                                    RPN_CFG['anchor_sizes'],
                                    RPN_CFG['aspect_ratio'] * len(RPN_CFG['anchor_sizes']))

        self.head             = rpn_utils.RPNHead(out_channels, len(RPN_CFG['aspect_ratio'][0]))

        self.box_coder        = rpn_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        self.box_similarity   = box_utils.box_iou  # used during training

        self.proposal_matcher = rpn_utils.Matcher(
                                    RPN_CFG['fg_iou_thresh'],
                                    RPN_CFG['bg_iou_thresh'],
                                    allow_low_quality_matches=True)

        self.fg_bg_sampler    = rpn_utils.BalancedPositiveNegativeSampler(
                                    RPN_CFG['batch_size_per_image'],
                                    RPN_CFG['positive_fraction'])
        # used during testing
        self.min_size        = 1e-3
        self.nms_thresh      = RPN_CFG['nms_thresh']
        self._pre_nms_top_n  = {'training' : RPN_CFG['pre_nms_top_n_train'],
                                'testing'  : RPN_CFG['pre_nms_top_n_test']}
        self._post_nms_top_n = {'training' : RPN_CFG['post_nms_top_n_train'],
                                'testing'  : RPN_CFG['post_nms_top_n_test']}


    @property
    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']


    @property
    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']


    def assign_targets_to_anchors(self, anchors, targets):

        labels, matched_gt_boxes = [], []
        for anchors_per_image, targets_per_image in zip(anchors, targets):

            gt_boxes = targets_per_image["boxes"]
            match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)
            matched_idxs = self.proposal_matcher(match_quality_matrix)
            # get the targets corresponding GT for each proposal
            # NB: need to clamp the indices because we can have a single
            # GT in the image, and matched_idxs can be -2, which goes
            # out of bounds
            matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

            labels_per_image = matched_idxs >= 0
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            # discard indices that are between thresholds
            inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_per_image[inds_to_discard] = -1

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes


    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        r = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, 1):
            num_anchors = ob.shape[1]
            pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)


    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        ''' '''
        # do not backprop throught objectness
        num_images, device = proposals.shape[0], proposals.device
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        levels = [
            torch.full((n,), idx, dtype=torch.int64, device=device)
                for idx, n in enumerate(num_anchors_per_level)
        ]
        levels = torch.cat(levels, 0)
        levels = levels.reshape(1, -1).expand_as(objectness)

        # select top_n boxes independently per level before applying nms
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)
        batch_idx = torch.arange(num_images, device=device)[:, None]
        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]

        final_boxes, final_scores = [], []
        for boxes, scores, lvl, img_shape in zip(proposals, objectness, levels, image_shapes):

            boxes = box_utils.clip_boxes_to_image(boxes, img_shape)
            keep  = box_utils.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            # non-maximum suppression, independently done per level
            keep = box_utils.batched_nms(boxes, scores, lvl, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.post_nms_top_n]
            boxes, scores = boxes[keep], scores[keep]
            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores


    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness = objectness.flatten()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = F.l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            reduction="sum",
        ) / (sampled_inds.numel())

        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss


    @staticmethod
    def concat_box_prediction_layers(box_cls, box_regression):

        def permute_and_flatten(layer, num_insts, n_pred, feat_h, feat_w):
            layer = layer.view(num_insts, -1, n_pred, feat_h, feat_w)
            layer = layer.permute(0, 3, 4, 1, 2)
            layer = layer.reshape(num_insts, -1, n_pred)
            return layer

        cls_flattened, regression_flattened = [], []
        for cls_pl, regression_pl in zip(box_cls, box_regression):

            num_insts, num_aspr, feat_h, feat_w = cls_pl.shape
            cls_pl = permute_and_flatten(cls_pl, num_insts, 1,  feat_h, feat_w)
            cls_flattened.append(cls_pl)

            regression_pl = permute_and_flatten(regression_pl, num_insts, 4, feat_h, feat_w)
            regression_flattened.append(regression_pl)

        box_cls = torch.cat(cls_flattened, dim=1).reshape(-1, 1)
        box_regression = torch.cat(regression_flattened, dim=1).reshape(-1, 4)

        return box_cls, box_regression


    def forward(self, images, features, targets=None):

        # RPN uses all feature maps that are available
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)   # generate the obj, deltas
        anchors = self.anchor_generator(images, features)    # forward-module of AnchorGenerator
        # anchors[inst_1.anchors, ... , inst_n.anchors]
        # inst_k.anchors with size : [num_anchors_k, 4]
        num_insts = len(anchors)
        num_anchors_per_level = [obj[0].numel() for obj in objectness]
        objectness, pred_bbox_deltas = self.concat_box_prediction_layers(objectness, pred_bbox_deltas)

        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_insts, -1, 4)

        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, \
                                                  num_anchors_per_level)
        losses = {}
        if self.training:
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(objectness, pred_bbox_deltas, \
                                                    labels, regression_targets)
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        return boxes, losses
