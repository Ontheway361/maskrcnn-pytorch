#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F

from IPython import embed

class BalancedPositiveNegativeSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentace of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction


    def __call__(self, matched_idxs):
        """
        Arguments:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image in matched_idxs:
            positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
            negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)

            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)

            # randomly select positive and negative examples
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx


@torch.jit.script
def encode_boxes(gt_boxes, anchors, weights):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    Encode a set of proposals with respect to some reference boxes

    Arguments:
        reference_boxes (Tensor): reference boxes
        proposals (Tensor): boxes to be encoded
    """

    # perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    anchors_x1 = anchors[:, 0].unsqueeze(1)
    anchors_y1 = anchors[:, 1].unsqueeze(1)
    anchors_x2 = anchors[:, 2].unsqueeze(1)
    anchors_y2 = anchors[:, 3].unsqueeze(1)

    gt_boxes_x1 = gt_boxes[:, 0].unsqueeze(1)
    gt_boxes_y1 = gt_boxes[:, 1].unsqueeze(1)
    gt_boxes_x2 = gt_boxes[:, 2].unsqueeze(1)
    gt_boxes_y2 = gt_boxes[:, 3].unsqueeze(1)

    # implementation starts here
    ex_widths = anchors_x2 - anchors_x1
    ex_heights = anchors_y2 - anchors_y1
    ex_ctr_x = anchors_x1 + 0.5 * ex_widths
    ex_ctr_y = anchors_y1 + 0.5 * ex_heights

    gt_widths = gt_boxes_x2 - gt_boxes_x1
    gt_heights = gt_boxes_y2 - gt_boxes_y1
    gt_ctr_x = gt_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = gt_boxes_y1 + 0.5 * gt_heights

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)

    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets


class BoxCoder(object):
    """
    This class encodes and decodes a set of bboxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip


    def encode(self, gt_boxes, anchors):
        ''' Encode the gt_boxes according to anchors '''

        boxes_per_image = [len(b) for b in gt_boxes]
        gt_boxes = torch.cat(gt_boxes, dim=0)
        anchors  = torch.cat(anchors, dim=0)
        dtype, device   = gt_boxes.dtype, gt_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(gt_boxes, anchors, weights)

        return targets.split(boxes_per_image, 0)


    def decode_single(self, anchor_deltas, anchors):
        '''
        anchors : [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        '''

        anchors = anchors.to(anchor_deltas.dtype)

        widths  = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        ctr_x = anchors[:, 0] + 0.5 * widths
        ctr_y = anchors[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = anchor_deltas[:, 0::4] / wx
        dy = anchor_deltas[:, 1::4] / wy
        dw = anchor_deltas[:, 2::4] / ww
        dh = anchor_deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(anchor_deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes


    def decode(self, anchor_deltas, anchors):
        ''' Refine anchors with anchor_deltas to generate the proposal '''

        assert isinstance(anchors, (list, tuple))
        if isinstance(anchor_deltas, (list, tuple)):
            anchor_deltas = torch.cat(anchor_deltas, dim=0)
        assert isinstance(anchor_deltas, torch.Tensor)

        anchors_per_image = [len(b) for b in anchors]
        concat_anchors = torch.cat(anchors, dim=0)
        proposals = self.decode_single(anchor_deltas.reshape(sum(anchors_per_image), -1), concat_anchors)
        return proposals.reshape(sum(anchors_per_image), -1, 4)


class Matcher(object):
    '''
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.

    The matcher returns a tensor of size N containing the index of the ground-truth
    element m that matches to prediction n. If there is no match, a negative value
    is returned.
    '''

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def __init__(self, high_threshold=0.7, low_threshold=0.3, allow_low_quality_matches=False):
        '''
        Args:
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        '''

        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches


    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find highest quality match available, even if it is low, including ties
        gt_pred_pairs_of_highest_quality = torch.nonzero(
            match_quality_matrix == highest_quality_foreach_gt[:, None]
        )

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]


    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    " No ground-truth boxes available for one of the images during training ")
            else:
                raise ValueError(
                    " No proposal boxes available for one of the images during training ")

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()

        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds  = (matched_vals >= self.low_threshold) & (matched_vals < self.high_threshold)
        matches[below_low_threshold] = Matcher.BELOW_LOW_THRESHOLD
        matches[between_thresholds]  = Matcher.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches


class AnchorGenerator(nn.Module):
    ''' Anchor generator equiped with multi-sizes, multi-aspect_ratios, even for pyramid-featmaps '''

    def __init__(self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):

        super(AnchorGenerator, self).__init__()

        if not isinstance(sizes[0], (list, tuple)):
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)

        assert len(sizes) == len(aspect_ratios)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}


    @staticmethod
    def generate_anchors(scales, aspect_ratios, device="cpu"):

        scales = torch.as_tensor(scales, dtype=torch.float32, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=torch.float32, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2 #center_point
        return base_anchors.round()


    def set_cell_anchors(self, device):

        if self.cell_anchors is not None:
            return self.cell_anchors

        cell_anchors = [self.generate_anchors(sizes, aspect_ratios, device)
                            for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)]

        self.cell_anchors = cell_anchors


    def num_anchors_per_location(self):
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]


    def grid_anchors(self, grid_sizes, strides):
        '''
        Generate the anchors according to base_anchor and stride
        grid_size : size of feature_map
        stride    : map_stride between img and feature_map
        '''
        anchors = list()
        for size, stride, base_anchors in zip(grid_sizes, strides, self.cell_anchors):

            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device
            shifts_x = torch.arange(0, grid_width, dtype=torch.float32, device=device) * stride_width
            shifts_y = torch.arange(0, grid_height,dtype=torch.float32, device=device) * stride_height
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts  = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))

        return anchors


    def cached_grid_anchors(self, grid_sizes, strides):

        key = tuple(grid_sizes) + tuple(strides)
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides)   #
        self._cache[key] = anchors
        return anchors


    def forward(self, image_list, feature_maps):
        '''
        step - 1. generate the base_anchor according to scales and aspect_ratios
        step - 2. generate the anchors over all feature maps for each type of image_size
        NOTE : there may be multi-scales feature_map
        '''

        # step - 1
        self.set_cell_anchors(feature_maps[0].device)

        # step - 2
        grid_sizes  = tuple([feature_map.shape[-2:] for feature_map in feature_maps])
        image_size  = image_list.tensors.shape[-2:]
        strides     = tuple((image_size[0] / g[0], image_size[1] / g[1]) for g in grid_sizes)
        all_anchors = self.cached_grid_anchors(grid_sizes, strides)

        anchors = []
        # deal with each instance in batch
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            for anchors_per_feature_map in all_anchors:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        return anchors


class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads

    Arguments:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """

    def __init__(self, in_channels, num_anchors):

        super(RPNHead, self).__init__()

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred  = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for l in self.children():
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):

        logits, bbox_reg = [], []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg
