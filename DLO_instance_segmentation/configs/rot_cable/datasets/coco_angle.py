dataset_type = 'CocoAngleDataset'
data_root = '../datasets/cable_dataset/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='RResize', img_scale=(800, 800)),
    dict(type='RRandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(800, 800)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='RResize', img_scale=(800, 800)),
            dict(type='RRandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(800, 800)),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        classes = ('Cable', 'CableEnd',),
        # classes = ('Cable',),
        ann_file=data_root + 'train/coco/annotations.json',
        img_prefix=data_root + 'train/coco/',
        # ann_file=data_root + 'forTesting/out/annotations.json',
        # img_prefix=data_root + 'forTesting/out/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes = ('Cable','CableEnd',),
        # classes = ('Cable',),
        ann_file=data_root + 'val/coco/annotations.json',
        img_prefix=data_root + 'val/coco/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes = ('Cable','CableEnd',),
        # classes = ('Cable',),
        ann_file=data_root + 'val/coco/annotations.json',
        img_prefix=data_root + 'val/coco/',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox'])
