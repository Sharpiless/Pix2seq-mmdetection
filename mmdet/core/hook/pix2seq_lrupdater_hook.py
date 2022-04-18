# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.lr_updater import (LrUpdaterHook, annealing_linear)

@HOOKS.register_module()
class LinearlyDecayLrUpdaterHook(LrUpdaterHook):

    def __init__(self, min_lr=None, min_lr_ratio=None, **kwargs):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        super(LinearlyDecayLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            warmup_epochs = self.warmup_epochs
            progress = runner.epoch
            max_progress = runner.max_epochs
            ratio = (progress-warmup_epochs) / (max_progress-warmup_epochs)
        else:
            warmup_iters = self.warmup_iters
            progress = runner.iter
            max_progress = runner.max_iters
            ratio = (progress-warmup_iters) / (max_progress-warmup_iters)

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr
        return annealing_linear(base_lr, target_lr, ratio)