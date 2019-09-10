#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
from utils.bbox_utils import box_area
from utils.roi_align  import roi_align

from IPython import embed

class LevelMapper(object):
    ''' Determine which FPN level each RoI in a set of RoIs should map to '''


    def __init__(self, k_min, k_max, canonical_scale=224, canonical_level=4, eps=1e-6):

        self.k_min = k_min
        self.k_max = k_max
        self.s0    = canonical_scale
        self.lvl0  = canonical_level
        self.eps   = eps


    def __call__(self, boxlists):

        # Compute level ids
        s = torch.sqrt(torch.cat([box_area(boxlist) for boxlist in boxlists]))

        # Eqn.(1) in FPN paper
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0 + self.eps))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return target_lvls.to(torch.int64) - self.k_min


class MultiScaleRoIAlign(nn.Module):
    '''
    Multi-scale RoIAlign pooling, which is useful for detection with or without FPN.

    It infers the scale of the pooling via the heuristics present in the FPN paper.

    Arguments:
        featmap_names (List[str]): the names of the feature maps that will be used
            for the pooling.
        output_size (List[Tuple[int, int]] or List[int]): output size for the pooled region
        sampling_ratio (int): sampling ratio for ROIAlign
    '''

    def __init__(self, featmap_names, output_size, sampling_ratio):

        super(MultiScaleRoIAlign, self).__init__()

        if isinstance(output_size, int):
            output_size = (output_size, output_size)

        self.featmap_names = featmap_names
        self.sampling_ratio = sampling_ratio
        self.output_size = tuple(output_size)
        self.scales = None
        self.map_levels = None


    def convert_to_roi_format(self, proposals):
        ''' Just put a flag for each proposal to note the proposal is responsible for which img '''

        concat_proposals = torch.cat(proposals, dim=0)
        device, dtype = concat_proposals.device, concat_proposals.dtype

        ids = torch.cat(
            [
                torch.full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(proposals)
            ],
            dim=0,
        )
        rois = torch.cat([ids, concat_proposals], dim=1)

        return rois


    def infer_scale(self, feature, original_size):
        # assumption: the scale is of the form 2 ** (-k), with k integer
        size = feature.shape[-2:]
        possible_scales = []
        for s1, s2 in zip(size, original_size):
            approx_scale = float(s1) / s2
            scale = 2 ** torch.tensor(approx_scale).log2().round().item()
            possible_scales.append(scale)
        assert possible_scales[0] == possible_scales[1]
        return possible_scales[0]


    def setup_scales(self, features, image_shapes):

        # extract the max-len for each dimension (max_y_length, max_x_length)
        original_input_shape = tuple(max(s) for s in zip(*image_shapes))
        scales = [self.infer_scale(feat, original_input_shape) for feat in features]

        # get the levels in the feature map by leveraging the fact that the network always
        # downsamples by a factor of 2 at each level.
        lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        self.scales = scales
        self.map_levels = LevelMapper(lvl_min, lvl_max)


    def forward(self, x, proposals, image_shapes):
        """
        Arguments:
            x (OrderedDict[Tensor]): feature maps for each level. They are assumed to have
                all the same number of channels, but they can have different sizes.
            proposals (List[Tensor[N, 4]]): proposals to be used to perform the pooling operation, in
                [x0, y0, x1, y1] format and in the image reference size, not the feature map
                reference.
            image_shapes (List[Tuple[height, width]]): the sizes of each image before they
                have been fed to a CNN to obtain feature maps. This allows us to infer the
                scale factor for each one of the levels to be pooled.
        Returns:
            result (Tensor)
        """

        x = [v for k, v in x.items() if k in self.featmap_names]
        num_levels = len(x)
        rois = self.convert_to_roi_format(proposals)

        if self.scales is None:
            self.setup_scales(x, image_shapes)

        if num_levels == 1:
            return roi_align(
                x[0], rois,
                output_size=self.output_size,
                spatial_scale=self.scales[0],
                sampling_ratio=self.sampling_ratio
            )

        levels = self.map_levels(proposals)

        num_rois, num_channels = len(rois), x[0].shape[1]
        dtype, device = x[0].dtype, x[0].device
        result = torch.zeros((num_rois, num_channels,) + self.output_size, dtype=dtype, device=device)

        for level, (per_level_feature, scale) in enumerate(zip(x, self.scales)):

            idx_in_level = torch.nonzero(levels == level).squeeze(1)
            rois_per_level = rois[idx_in_level]

            result[idx_in_level] = roi_align(
                per_level_feature, rois_per_level,
                output_size=self.output_size,
                spatial_scale=scale, sampling_ratio=self.sampling_ratio)

        return result