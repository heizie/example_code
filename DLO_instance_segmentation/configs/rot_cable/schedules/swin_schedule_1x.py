# do not use mmdet version fp16
fp16 = None
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict(
    grad_clip=None,
    # grad_clip=dict(max_norm=35, norm_type=2),
    type='DistOptimizerHook',
    update_interval=1,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1e-05,
    step=[27, 33])

name = 'correct_class_agnostic'
runner = dict(
    type='EpochBasedRunnerAmp', 
    max_epochs=50,
    work_dir = '/home/iwb/project/trained_paths/rot_cable/{}'.format(name)
)
