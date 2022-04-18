_base_ = [
    '../_base_/datasets/coco_detection_pix2seq.py', '../_base_/default_runtime.py'
]
model = dict(
    type='Pix2seq',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=0,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,  # Frozen BN weight and bias
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    bbox_head=dict(
        type='Pix2seqHead',
        num_classes=91,
        in_channels=2048,
        num_vocal=2094,
        rand_target=True,
        drop_token=True,
        random_token=True,
        split_loss=False,
        transformer=dict(
            type='Pix2seqTransformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='Pix2seqTransformerDecoder',
                num_layers=6,
                post_norm_cfg=dict(type='LN'),
                transformerlayers=dict(
                    type='Pix2seqTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='Pix2seqAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1,
                            self_attn_dropout=0.1),
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            ),
        ),
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=False)))
# augment
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomDistortion',
        brightness=0.5,
        contrast=0.5,
        saturation=0.5,
        hue=0.5),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='LargeScaleJitter',
        desired_size=1333,
        ratio_range=(0.3, 2.0),
        keep_ratio=True,
        allow_negative_crop=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 1333),
        flip=False,
        transforms=[
            dict(
                type='LargeScaleJitter',
                desired_size=1333,
                ratio_range=(1.0, 1.0),
                keep_ratio=True,
                allow_negative_crop=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
# dataset
dataset_type = 'Pix2seqCocoDataset'
data_root = 'data/coco/'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline)
)
# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-3,
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
runner = dict(type='EpochBasedRunner', max_epochs=300)
fp16 = dict(loss_scale='dynamic')
checkpoint_config = dict(interval=50)
lr_config = dict(
    policy='LinearlyDecay',
    warmup='linear',
    by_epoch=True,
    warmup_by_epoch=True,
    warmup_ratio=0.01,
    warmup_iters=10,  # 10 epoch
    min_lr_ratio=0.01)
evaluation = dict(interval=10, metric='bbox')