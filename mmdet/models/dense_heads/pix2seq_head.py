# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, build_activation_layer
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy
from mmdet.models.utils import build_transformer
from ..builder import HEADS
from .anchor_free_head import AnchorFreeHead
import torch.distributed as dist


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


@HEADS.register_module()
class Pix2seqHead(AnchorFreeHead):
    """Implements the Pix2seq transformer head.

    Implementation of `Pix2seq: A Language Modeling Framework for Object Detection
     <https://arxiv.org/abs/2109.10852>`_

    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    _version = 2

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query=100,
                 num_reg_fcs=2,
                 transformer=None,
                 num_bins=2000,
                 num_vocal=2094,
                 eos_coef=0.1,
                 drop_token=False,
                 random_token=False,
                 rand_target=True,
                 split_loss=False,
                 eos_bias=0,
                 sync_cls_avg_factor=False,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 init_cfg=None,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.num_query = num_query
        self.num_bins = num_bins
        self.num_vocal = num_vocal
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.fp16_enabled = False
        self.drop_token = drop_token
        self.random_token = random_token
        self.rand_target = rand_target
        self.split_loss = split_loss
        self.eos_bias = eos_bias
        empty_weight = torch.ones(self.num_vocal)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(self.act_cfg)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)
        self.transformer.num_classes = num_classes
        self.transformer.num_bins = num_bins
        self.embed_dims = self.transformer.embed_dims
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            f' and {num_feats}.'
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        self.input_proj = nn.Sequential(
            Conv2d(self.in_channels, self.embed_dims, kernel_size=(1, 1)),
            nn.GroupNorm(32, self.embed_dims))
        self.vocal_classifier = nn.Linear(self.embed_dims, self.num_vocal)
        self.det_embed = nn.Embedding(1, self.embed_dims)
        self.vocal_embed = nn.Embedding(self.num_vocal - 2, self.embed_dims)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None or version < 2) and self.__class__ is Pix2seqHead:
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    def forward(self, feats, img_metas, targets=None):
        assert len(feats) == 1, "only one feature is supported"
        results = self.forward_single(feats[0], img_metas, targets)
        return results

    def forward_single(self, x, img_metas, targets):
        # construct binary masks which used for the transformer.
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.
        batch_size = x.size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        masks = x.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            masks[img_id, :img_h, :img_w] = 0

        x = self.input_proj(x)
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)
        # position encoding
        pos_embed = self.positional_encoding(masks)  # [bs, embed_dim, h, w]
        # outs_dec: [nb_dec, bs, num_query, embed_dim]
        if self.training:
            input_seq = self.build_input_seq(
                targets=targets, drop_token=self.drop_token, random_token=self.random_token)
            pred_seq_logits = self.transformer(x, input_seq, masks,
                                               pos_embed, self.det_embed, self.vocal_embed,
                                               self.vocal_classifier, self.num_vocal)
        else:
            pred_seq_logits = self.transformer(x, -1, masks,
                                               pos_embed, self.det_embed, self.vocal_embed,
                                               self.vocal_classifier, self.num_vocal)
        return pred_seq_logits

    @force_fp32(apply_to=('pred_seq_logits'))
    def loss(self, pred_seq_logits, targets):
        # pred_seq_logits = pred_seq_logits[0]
        target_seq = self.build_target_seq(targets)
        num_pos = (target_seq > -1).sum()
        num_pos = torch.as_tensor(
            [num_pos], dtype=torch.float, device=target_seq.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_pos)
        num_pos = torch.clamp(num_pos / get_world_size(), min=1).item()
        pred_seq_logits = pred_seq_logits.reshape(-1, self.num_vocal)
        target_seq = target_seq.flatten()
        losses = dict()
        if self.split_loss:
            cls_mask = target_seq > self.num_bins+1
            reg_mask = torch.logical_not(cls_mask)
            loss_seq_cls = F.cross_entropy(
                pred_seq_logits[cls_mask][:, self.num_bins +
                                          1:], target_seq[cls_mask] - self.num_bins - 1,
                weight=self.empty_weight[self.num_bins + 1:], reduction='sum') / num_pos
            loss_seq_reg = F.cross_entropy(
                pred_seq_logits[reg_mask][:,
                                          :self.num_bins + 1], target_seq[reg_mask],
                weight=self.empty_weight[:self.num_bins + 1], reduction='sum') / num_pos
            losses["loss_seq_cls"] = loss_seq_cls
            losses["loss_seq_reg"] = loss_seq_reg
        else:
            loss_seq = F.cross_entropy(
                pred_seq_logits, target_seq, weight=self.empty_weight, reduction='sum') / num_pos
            losses["loss_seq"] = loss_seq
        return losses

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_bboxes,
                           gt_labels,
                           img_meta,
                           gt_bboxes_ignore=None):
        raise NotImplementedError
    # over-write because img_metas are needed as inputs for bbox_head.

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):

        assert proposal_cfg is None, '"proposal_cfg" must be None'
        targets = []
        for i in range(len(img_metas)):
            if self.rand_target:
                num_object = gt_bboxes[i].shape[0]
                rand_idx = torch.randperm(num_object).to(gt_bboxes[i].device)
                targets.append({
                    "boxes": gt_bboxes[i][rand_idx],
                    "labels": gt_labels[i][rand_idx],
                    "size": img_metas[i]["img_shape"],
                })
            else:
                targets.append({
                    "boxes": gt_bboxes[i],
                    "labels": gt_labels[i],
                    "size": img_metas[i]["img_shape"],
                })
        outs = self(x, img_metas, targets)
        losses = self.loss(outs, targets)
        return losses

    @force_fp32(apply_to=('pred_seq_logits'))
    def get_bboxes(self,
                   pred_seq_logits,
                   img_metas,
                   rescale=False):

        origin_img_sizes = torch.stack(
            [torch.tensor(t["ori_shape"], device='cuda:0') for t in img_metas], dim=0)

        assert len(pred_seq_logits) == len(origin_img_sizes)

        results = []
        for b_i, out_seq_logits in enumerate(pred_seq_logits):
            seq_len = out_seq_logits.shape[0]
            if seq_len < 5:
                results.append((None, None))
                continue
            out_seq_logits = out_seq_logits.softmax(dim=-1)
            num_objects = seq_len // 5
            out_seq_logits = out_seq_logits[:int(
                num_objects * 5)].reshape(num_objects, 5, -1)
            pred_boxes_logits = out_seq_logits[:, :4, :self.num_bins + 1]
            pred_class_logits = out_seq_logits[:, 4,
                                               self.num_bins + 1: self.num_bins + 1 + self.num_classes]
            scores_per_image, labels_per_image = torch.max(
                pred_class_logits, dim=1)
            boxes_per_image = pred_boxes_logits.argmax(
                dim=2) * 1333 / self.num_bins
            if rescale:
                scale_factor = img_metas[b_i]['scale_factor']
                boxes_per_image /= boxes_per_image.new_tensor(scale_factor)
            results.append((
                torch.cat((boxes_per_image, scores_per_image.unsqueeze(
                    1)), dim=-1), labels_per_image
            ))

        return results

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        # forward of this head requires img_metas
        outs = self.forward(feats, img_metas)
        results_list = self.get_bboxes(outs, img_metas, rescale=rescale)
        return results_list

    def forward_onnx(self, feats, img_metas):
        num_levels = len(feats)
        img_metas_list = [img_metas for _ in range(num_levels)]
        return multi_apply(self.forward_single_onnx, feats, img_metas_list)

    def forward_single_onnx(self, x, img_metas):
        raise NotImplementedError

    def build_target_seq(self, targets, max_objects=100):
        device = targets[0]["labels"].device
        target_seq_list = []
        for target in targets:
            label = target["labels"]
            box = target["boxes"]

            label = label.unsqueeze(1) + self.num_bins + 1
            box = (box / 1333 * self.num_bins).floor().long().clamp(min=0,
                                                                    max=self.num_bins)
            target_tokens = torch.cat([box, label], dim=1).flatten()

            num_noise = max_objects - len(label)
            fake_target_tokens = torch.zeros(
                (num_noise, 5), dtype=torch.int64).to(device)
            fake_target_tokens[:, :4] = -100
            fake_target_tokens[:, 4] = self.num_vocal - 1  # noise class
            fake_target_tokens = fake_target_tokens.flatten()
            
            target_seq = torch.cat(
                [target_tokens, fake_target_tokens], dim=0)
            target_seq_list.append(target_seq)
        return torch.stack(target_seq_list, dim=0)

    def build_input_seq(self, targets, max_objects=100, drop_token=False, random_token=False):
        device = targets[0]["labels"].device
        input_seq_list = []
        for target in targets:
            scaled_box = target["boxes"]
            label = target["labels"]
            img_size = target["size"]
            h, w, _ = torch.tensor(img_size, device=device)
            scale_factor = torch.stack([w, h, w, h], dim=0)

            label = label.unsqueeze(1) + self.num_bins + 1
            label_token = label.clone()
            if drop_token:
                drop_mask = torch.rand(label.shape) < 0.5
                if random_token:
                    random_tokens = torch.randint(
                        0, self.num_classes, (1, drop_mask.sum())).to(label_token.device)
                    label_token[drop_mask] = random_tokens + self.num_bins + 1
                else:
                    label_token[drop_mask] = self.num_vocal - 2

            box_tokens = (
                scaled_box / 1333 * self.num_bins).floor().long().clamp(min=0, max=self.num_bins)
            input_tokens = torch.cat(
                [box_tokens, label_token], dim=1)  # [N, 4+1]
            num_objects = input_tokens.shape[0]
            num_noise = max_objects - num_objects
            random_class = torch.randint(
                0, self.num_classes, (num_noise, 1), device=device) + self.num_bins + 1
            random_box_x0y0 = torch.rand(num_noise, 2, device=device)
            random_box_wh = torch.rand(num_noise, 2, device=device)
            random_box_x1y1 = (random_box_x0y0 +
                               random_box_wh).clamp(min=0, max=1)
            random_scaled_box = torch.cat(
                [random_box_x0y0, random_box_x1y1], dim=1) * scale_factor
            random_box_tokens = (random_scaled_box / 1333 *
                                 self.num_bins).floor().long().clamp(min=0, max=self.num_bins)
            random_tokens = torch.cat([random_box_tokens, random_class], dim=1)

            if num_objects > 0:
                jitter_box_idx = torch.randint(
                    0, num_objects, (num_noise,), device=device)
                jitter_class = label[jitter_box_idx]
                if drop_token:
                    drop_mask = torch.rand(jitter_class.shape) < 0.5
                    jitter_class[drop_mask] = 0
                scaled_jitter_box = scaled_box[jitter_box_idx]
                jitter_box_wh = scaled_jitter_box[:,
                                                  2:] - scaled_jitter_box[:, :2]
                jitter_box_wh = jitter_box_wh.repeat(1, 2) / scale_factor
                noise = (torch.rand((num_noise, 4), device=device) -
                         0.5) * 2 * 0.2 * jitter_box_wh
                scaled_jitter_box = noise * scale_factor + scaled_jitter_box
                jitter_box_tokens = (scaled_jitter_box / 1333 *
                                     self.num_bins).floor().long().clamp(min=0, max=self.num_bins)
                jitter_tokens = torch.cat(
                    [jitter_box_tokens, jitter_class], dim=1)

                fake_tokens = torch.stack(
                    [random_tokens, jitter_tokens], dim=1)  # torch.Size([98, 2, 5])
                select_idx = torch.randint(0, 2, (num_noise,), device=device)
                fake_tokens = fake_tokens[range(num_noise), select_idx]
            else:
                fake_tokens = random_tokens

            input_seq = torch.cat(
                [input_tokens, fake_tokens], dim=0).flatten()  # [500,]
            input_seq_list.append(input_seq)

        results = torch.stack(input_seq_list, dim=0)  # [B, 500]
        return results[:, :-1]

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        raise NotImplementedError


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

