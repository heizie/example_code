_base_ = [
'../datasets/coco_angle.py',
'../schedules/swin_schedule_1x.py',
'../base/default_runtime.py'
]

num_classes=2

model = dict(
    type='RHybridTaskCascade',
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=None
    ),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        num_outs=5
    ),
    # refactored rpn for RoI-transformer
    rpn_roi_transformer_head=dict(
        type='ROITransformer',
        # (H)RoI RPN
        rpn_head=dict(
            type='RPNHead',
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[8],
                ratios=[0.5, 1.0, 2.0], # original ratios
                # ratios=[0.1, 0.4, 2, 3, 12],
                strides=[4, 8, 16, 32, 64]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)
        ),
        # The regression RoI Head from HBB to OBB
        roi_regression_head=dict(
            type='StandardRRoIHead',
            # extract HBB
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', 
                    output_size=7, 
                    sampling_ratio=2),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]
            ),
            # regress HBB for OBB
            bbox_head=dict(
                type='SharedFCRBBoxHead',
                num_fcs=2,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYLSTBBoxCoder',
                    target_means=(0., 0., 0., 0., 0.),
                    target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                                loss_weight=1.0)),
        ),
        # The RoI Head for generate RBBox and Cls
        rroi_head=dict(
            type='StandardRRoIHead',
             # extract rroi for rbox
            bbox_roi_extractor=dict(
                type='RboxSingleRoIExtractor',
                roi_layer=dict(
                    type='ROIAlignRotated', 
                    output_size=7, 
                    sampling_ratio=2),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]
            ),

            # generate RBBox and Cls
            rbbox_head=dict(
                type='SharedFCRBBoxHead',
                num_fcs=2,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYLSTBBoxCoder',
                    target_means=(0., 0., 0., 0., 0.),
                    target_stds=(0.05, 0.05, 0.1, 0.1, 0.05)
                ),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
            )
        )
    ),
        # model training and testing settings
    train_cfg = dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='BboxOverlaps2D')),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_num=2000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='BboxOverlaps2D')),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='RBBoxOverlaps2D')),
                sampler=dict(
                    type='RbboxRandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)]
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            nms_pre=1000,
            score_thr=0.05,
            nms=dict(type='rnms', iou_thr=0.1),
            max_per_img=100
        )
    ) # test_cfg
) # model
 