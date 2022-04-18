# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
import warnings

import torch

from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class Pix2seq(SingleStageDetector):
    r"""Implementation of `Pix2seq: A Language Modeling Framework for Object Detection
     <https://arxiv.org/abs/2109.10852>`_"""

    def __init__(self,
                 backbone,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(Pix2seq, self).__init__(backbone, None, bbox_head, train_cfg,
                                      test_cfg, pretrained, init_cfg)

    def train_step(self, ori_data, optimizer):

        if isinstance(ori_data, list):
            data = {}
            data['img_metas'] = ori_data[0]['img_metas'] + \
                ori_data[1]['img_metas']
            data['img'] = torch.cat([ori_data[0]['img'], ori_data[1]['img']])
            data['gt_bboxes'] = ori_data[0]['gt_bboxes'] + \
                ori_data[1]['gt_bboxes']
            data['gt_labels'] = ori_data[0]['gt_labels'] + \
                ori_data[1]['gt_labels']
        else:
            data = ori_data
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs
