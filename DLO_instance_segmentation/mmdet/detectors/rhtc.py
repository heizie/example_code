import torch
import torch.nn as nn
import mmcv
import numpy as np
from mmdet.core.visualization import imshow_det_rbboxes
from ..builder import DETECTORS, build_backbone, build_head, build_neck

from .base import BaseDetector

@DETECTORS.register_module()
class RHybridTaskCascade(BaseDetector):
    """Base class for the oriented object detection task
    w.r.t Mask RCNN struckture with HTC
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_roi_transformer_head=None,
                 rroi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RHybridTaskCascade, self).__init__()
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        # refactor from two-stage for roi-trans
        if rpn_roi_transformer_head is not None:
            rpn_train_cfg = train_cfg if train_cfg is not None else None
            rpn_roi_transformer_head_ = rpn_roi_transformer_head.copy()
            rpn_roi_transformer_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg)
            self.rpn_roi_transformer_head = build_head(rpn_roi_transformer_head_)

        if rroi_head is not None:
            rroi_rcnn_train_cfg = train_cfg.rcnn[1] if train_cfg is not None else None
            rroi_head.update(train_cfg=rroi_rcnn_train_cfg)
            rroi_head.update(test_cfg=test_cfg)
            self.rroi_head = build_head(rroi_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)


    def init_weights(self, pretrained=None):
        """Initialize the weights in detector with pre-trained weights.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(RHybridTaskCascade, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()

        # for roi transformer
        if self.with_rpn_roi_transformer_head:
            self.rpn_roi_transformer_head.init_weights()

        if self.with_rroi_head and self.rroi_head :
            self.rroi_head.init_weights()


    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    # TODO
    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn_roi_transformer_head:
            rpn_outs = self.rpn_roi_transformer_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.rroi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [xc, yc, longest, shortest, theta] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()
        proposal_list=None
        
        # RPN forward and loss
        if self.with_rpn_roi_transformer_head:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_roi_transformer_head.forward_train(
                                                x,
                                                img_metas,
                                                gt_bboxes, # rbbox
                                                gt_labels=gt_labels,
                                                gt_bboxes_ignore=gt_bboxes_ignore,
                                                gt_masks=gt_masks,
                                                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        if self.with_rroi_head:
            roi_losses = self.rroi_head.forward_train(
                                                x, 
                                                img_metas,
                                                proposal_list,
                                                gt_bboxes, # rbbox
                                                gt_labels,
                                                gt_bboxes_ignore,
                                                gt_masks)

            losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""

        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_roi_transformer_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        x = self.extract_feat(img)

        # get origin input shape to onnx dynamic input shape
        if torch.onnx.is_in_onnx_export():
            img_shape = torch._shape_as_tensor(img)[2:]
            img_metas[0]['img_shape_for_onnx'] = img_shape

        if proposals is None:
            proposal_list = self.rpn_roi_transformer_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals
        
        # refine phase
        if self.with_rroi_head:
            return self.rroi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
        
        else:
            return proposal_list

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_roi_transformer_head.aug_test_rpn(x, img_metas)

        if self.with_rroi_head:
            proposal_list = self.rroi_head.aug_test(
                x, proposal_list, img_metas, rescale=rescale)
        else:
            return proposal_list

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Show prediction results of the detector.

        Args:
            data (str or np.ndarray): Image filename or loaded image.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).

        Returns:
            np.ndarray: The image with bboxes drawn on it.
        """
        # if self.with_mask:
        #     ms_bbox_result, ms_segm_result = result
        #     if isinstance(ms_bbox_result, dict):
        #         result = (ms_bbox_result['ensemble'],
        #                   ms_segm_result['ensemble'])
        # else:
        #     if isinstance(result, dict):
        #         result = result['ensemble']

        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        bboxes[:,4] += np.pi/4 
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        img = imshow_det_rbboxes(
            img,
            bboxes,
            labels,
            segms,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img
