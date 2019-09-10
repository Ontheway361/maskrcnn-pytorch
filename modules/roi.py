#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2019/09/08
@author: relu
"""

import torch
from torch import nn
import torch.nn.functional as F

import utils.bbox_utils as box_ops
import utils.misc as misc_nn_ops
import utils.rpn_utils as rpn_utils
import utils.roi_align as roi_align
from modules.cfg import RoI_CFG
from utils.roi_utils import MultiScaleRoIAlign

from IPython import embed


class TwoMLPHead(nn.Module):

    ''' Standard heads for FPN-based models '''

    def __init__(self, in_channels, representation_size):

        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):

        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


class Predictor(nn.Module):
    '''
    Standard classification + bounding box regression layers for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    '''

    def __init__(self, in_channels, num_classes):

        super(Predictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):

        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


class RoI(nn.Module):


    def __init__(self, num_classes, out_channels):

        super(RoI, self).__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = rpn_utils.Matcher(
                                    RoI_CFG['fg_iou_thresh'],
                                    RoI_CFG['bg_iou_thresh'],
                                    allow_low_quality_matches=False)

        self.fg_bg_sampler = rpn_utils.BalancedPositiveNegativeSampler(
                                 RoI_CFG['batch_size_per_image'],
                                 RoI_CFG['positive_fraction'])

        self.nms_thresh   = RoI_CFG['nms_thresh']
        self.score_thresh = RoI_CFG['score_thresh']
        self.detections_per_img = RoI_CFG['detections_per_img']

        self.box_coder = rpn_utils.BoxCoder(RoI_CFG['reg_weights'])

        self.box_roi_pool  = MultiScaleRoIAlign(
                                 RoI_CFG['featmap_names'],
                                 RoI_CFG['output_size'],
                                 RoI_CFG['sampling_ratio'])

        self.box_head      = TwoMLPHead(
                                 out_channels * self.box_roi_pool.output_size[0] ** 2,
                                 RoI_CFG['representation_size'])

        self.box_predictor = Predictor(RoI_CFG['representation_size'], num_classes)


    def check_targets(self, targets):
        ''' Check the target '''

        assert targets is not None
        assert all("boxes" in t for t in targets)
        assert all("labels" in t for t in targets)


    def add_gt_proposals(self, proposals, gt_boxes):
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals


    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):

        matched_idxs, labels = [], []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):

            match_quality_matrix = self.box_similarity(gt_boxes_in_image, proposals_in_image)
            matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

            clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

            labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
            labels_in_image = labels_in_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_in_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)

        return matched_idxs, labels


    def subsample(self, labels):

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []

        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):

            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            sampled_inds.append(img_sampled_inds)

        return sampled_inds


    def select_training_samples(self, proposals, targets):
        ''' '''

        self.check_targets(targets)

        gt_boxes  = [t["boxes"] for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)

        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)

        for img_id in range(num_images):

            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]
            matched_gt_boxes.append(gt_boxes[img_id][matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)

        return proposals, matched_idxs, labels, regression_targets


    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        ''' Post-process for prediction '''

        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)

        all_boxes, all_scores, all_labels = [], [], []
        for boxes, scores, image_shape in zip(pred_boxes, pred_scores, image_shapes):

            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)

            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels


    @staticmethod
    def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
        ''' Computes the loss for Faster R-CNN '''

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        classification_loss = F.cross_entropy(class_logits, labels)

        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        N, num_classes = class_logits.shape
        box_regression = box_regression.reshape(N, -1, 4)

        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            reduction="sum",
        )
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss


    def forward(self, features, proposals, image_shapes, targets=None):

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)

        box_features = self.box_roi_pool(features, proposals, image_shapes)

        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result, losses = [], {}

        if self.training:
            loss_classifier, loss_box_reg = self.fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                'loss_classifier' : loss_classifier,
                'loss_box_reg' : loss_box_reg,
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            for i in range(len(boxes)):
                result.append(dict(boxes=boxes[i], labels=labels[i], scores=scores[i]))

        return result, losses
