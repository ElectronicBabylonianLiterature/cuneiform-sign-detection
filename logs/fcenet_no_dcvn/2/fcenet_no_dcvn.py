fourier_degree = 5
model = dict(
    type='FCENet',
    backbone=dict(
        type='mmdet.ResNet',
        depth=18,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        norm_eval=False,
        style='pytorch'),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[128, 256, 512],
        out_channels=256,
        add_extra_convs='on_output',
        num_outs=3,
        relu_before_extra_convs=True,
        act_cfg=None),
    bbox_head=dict(
        type='FCEHead',
        in_channels=256,
        scales=(8, 16, 32),
        loss=dict(type='FCELoss'),
        alpha=1.2,
        beta=1.0,
        text_repr_type='quad',
        fourier_degree=5))
train_cfg = None
test_cfg = None
dataset_type = 'IcdarDataset'
data_root = './data'
img_norm_cfg = dict(
    mean=(93.74563408650725, 72.73097058157583, 57.578964634542544),
    std=(63.56271455429814, 52.89510529933301, 45.35473046012471),
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='ColorJitter',
        brightness=0.12549019607843137,
        saturation=0.5,
        contrast=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='RandomScaling', size=800, scale=(0.75, 2.5)),
    dict(
        type='RandomCropFlip', crop_ratio=0.5, iter_num=1, min_area_ratio=0.2),
    dict(
        type='RandomCropPolyInstances',
        instance_key='gt_masks',
        crop_ratio=0.8,
        min_side_ratio=0.3),
    dict(
        type='RandomRotatePolyInstances',
        rotate_ratio=0.5,
        max_angle=30,
        pad_with_fixed_color=False),
    dict(type='SquareResizePad', target_size=800, pad_ratio=0.6),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='Pad', size_divisor=32),
    dict(
        type='FCENetTargets',
        fourier_degree=5,
        level_proportion_range=((0, 0.4), (0.3, 0.7), (0.6, 1.0))),
    dict(
        type='CustomFormatBundle',
        keys=['p3_maps', 'p4_maps', 'p5_maps'],
        visualize=dict(flag=False, boundary_key=None)),
    dict(type='Collect', keys=['img', 'p3_maps', 'p4_maps', 'p5_maps'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2260, 2260),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(1280, 800), keep_ratio=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='IcdarDataset',
        ann_file='./data/instances_training.json',
        img_prefix='./data/imgs',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadTextAnnotations',
                with_bbox=True,
                with_mask=True,
                poly2mask=False),
            dict(
                type='ColorJitter',
                brightness=0.12549019607843137,
                saturation=0.5,
                contrast=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='RandomScaling', size=800, scale=(0.75, 2.5)),
            dict(
                type='RandomCropFlip',
                crop_ratio=0.5,
                iter_num=1,
                min_area_ratio=0.2),
            dict(
                type='RandomCropPolyInstances',
                instance_key='gt_masks',
                crop_ratio=0.8,
                min_side_ratio=0.3),
            dict(
                type='RandomRotatePolyInstances',
                rotate_ratio=0.5,
                max_angle=30,
                pad_with_fixed_color=False),
            dict(type='SquareResizePad', target_size=800, pad_ratio=0.6),
            dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
            dict(type='Pad', size_divisor=32),
            dict(
                type='FCENetTargets',
                fourier_degree=5,
                level_proportion_range=((0, 0.4), (0.3, 0.7), (0.6, 1.0))),
            dict(
                type='CustomFormatBundle',
                keys=['p3_maps', 'p4_maps', 'p5_maps'],
                visualize=dict(flag=False, boundary_key=None)),
            dict(
                type='Collect', keys=['img', 'p3_maps', 'p4_maps', 'p5_maps'])
        ]),
    val=dict(
        type='IcdarDataset',
        ann_file='./data/instances_validation.json',
        img_prefix='./data/imgs',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2260, 2260),
                flip=False,
                transforms=[
                    dict(
                        type='Resize', img_scale=(1280, 800), keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='IcdarDataset',
        ann_file='./data/instances_validation.json',
        img_prefix='./data/imgs',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2260, 2260),
                flip=False,
                transforms=[
                    dict(
                        type='Resize', img_scale=(1280, 800), keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(
    interval=20,
    metric=['hmean-iou'],
    save_best='hmean-iou:hmean',
    rule='greater')
optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='poly', power=0.9)
total_epochs = 750
checkpoint_config = dict(interval=25)
log_config = dict(
    interval=10,
    hooks=[dict(type='TensorboardLoggerHook'),
           dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = './checkpoints/fcenet_cpu.pth'
resume_from = None
workflow = [('train', 10), ('val', 1)]
work_dir = 'logs/fcenet_no_dcvn/2'
gpu_ids = range(0, 1)
