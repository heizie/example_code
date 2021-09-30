# from six.moves import xrange
from numpy.core.numeric import True_
import torch

from mmdet.core import bbox2roi, build_assigner, build_sampler, bbox_mapping
# for rbbox
from mmdet.core import (dbbox2result, dbbox2roi, roi2droi, rbboxes_to_xyxy, 
                        merge_rotate_aug_bboxes, bbox_rotate_mapping, multiclass_rnms,
                        choose_best_Rroi_batch)
from ..builder import HEADS, build_head, build_roi_extractor
from .base_rroi_head import BaseRRoIHead
from .test_mixins import MaskTestMixin
import copy

@HEADS.register_module()
class StandardRRoIHead(BaseRRoIHead, MaskTestMixin):
    """Simplest base roi head including one bbox head.
    
    This is used for regression head and the rbbox head of
    RoI-Transformer

    'bbox' is a generall name for axis-align bounding box or oriented bounding box
    """

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            # TODO there could be some better way for choosing
            # different elelment from rcnn
            assigner_cfg = self.train_cfg.assigner
            sampler_cfg = self.train_cfg.sampler
            self.bbox_assigner = build_assigner(assigner_cfg)
            self.bbox_sampler = build_sampler(
                sampler_cfg, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head=None, rbbox_head=None):
        """Initialize ``bbox_head`` or ``rbbox_head`` """
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        if bbox_head is not None:
            self.bbox_head = build_head(bbox_head)
        if rbbox_head is not None:
            self.rbbox_head = build_head(rbbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def init_weights(self):
        """Initialize the weights in head.
        """
        if self.with_bbox or self.with_rbbox:
            self.bbox_roi_extractor.init_weights()
            if self.with_bbox:
                self.bbox_head.init_weights()
            if self.with_rbbox:
                self.rbbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        if self.with_bbox:
            rois = bbox2roi([proposals])
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        if self.with_rbbox:
            rois = dbbox2roi([proposals])
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): ROTATED Ground truth bboxes for each image with
                shape (num_gts, 5) in [tl_x, tl_y, br_x, br_y, theta] format.
            gt_labels (list[Tensor]): class indices corresponding to each bbox
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                bboxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_rbbox:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                if self.with_bbox:
                    assign_result = self.bbox_assigner.assign(
                        proposal_list[i],
                        rbboxes_to_xyxy(gt_bboxes[i]),  # gt_rbbox to gt_bbox
                        gt_bboxes_ignore[i],
                        gt_labels[i])
                    sampling_result = self.bbox_sampler.sample(
                        assign_result,
                        proposal_list[i],
                        rbboxes_to_xyxy(gt_bboxes[i]),
                        gt_labels[i],
                        feats=[lvl_feat[i][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)
                elif self.with_rbbox or self.with_mask:
                    # for rbbox_head, make gt_rbbox sure h < w
                    # gt_obbs_best = choose_best_Rroi_batch(gt_bboxes[i]) 
                    assign_result = self.bbox_assigner.assign(
                        proposal_list[i],
                        gt_bboxes[i],
                        gt_bboxes_ignore[i],
                        gt_labels[i])
                    sampling_result = self.bbox_sampler.sample(
                        assign_result,
                        proposal_list[i],
                        gt_bboxes[i],
                        gt_labels[i],
                        feats=[lvl_feat[i][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)
                

        losses = dict()
        # bbox head forward and loss
        # regression roi head for ROI Transformer
        if self.with_bbox:
            bbox_results, proposal_list = self._bbox_forward_train(
                                                x, sampling_results,
                                                gt_bboxes, img_metas)
            # losses.update(bbox_results['loss_bbox'])
            # name the losses differently
            losses.update(r_Trans_loss_bbox = bbox_results['loss_bbox'],
                          r_Trans_loss_cls  = bbox_results['loss_cls'],
                          r_Trans_acc       = bbox_results['acc'])
            return losses, proposal_list
        # refine roi head
        elif self.with_rbbox:
            bbox_results = self._bbox_forward_train(
                                                x, sampling_results,
                                                gt_bboxes, img_metas)
            #losses.update(bbox_results['loss_bbox'])
            # name the losses differently
            losses.update(loss_bbox = bbox_results['loss_bbox'],
                          loss_cls  = bbox_results['loss_cls'],
                          acc       = bbox_results['acc'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x, rois):
        """BBox head forward function used in both training and testing."""
        # rois[:, 3] = rois[:, 3] * self.bbox_roi_extractor.w_enlarge
        # rois[:, 4] = rois[:, 4] * self.bbox_roi_extractor.h_enlarge
        bbox_feats = self.bbox_roi_extractor(
                    x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_bbox:
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
        if self.with_rbbox:
            cls_score, bbox_pred = self.rbbox_head(bbox_feats)
        
        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, img_metas):
        """Run forward function and calculate loss for bbox head in training."""
        # TODO: choose correct 'get_targets' for regression and rbbox_head
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_results = self._bbox_forward(x, rois)
            bbox_targets = self.bbox_head.get_targets_bboxes2rbbox(
                                        sampling_results,
                                        gt_bboxes,
                                        self.train_cfg)
            # calculated loss with (gt)box_targets and (pred)rois
            loss_bbox = self.bbox_head.loss(
                                        bbox_results['cls_score'],
                                        bbox_results['bbox_pred'],
                                        *bbox_targets)
            for name, value in loss_bbox.items():
                bbox_results['{}'.format(name)] = (value)

            # refine proposals
            # duplicated from roi_transformer
            # the regressed proposal(rbboxes) for training next branch
            pos_is_gts = [res.pos_is_gt for res in sampling_results]
            with torch.no_grad():
                proposal_list = self.bbox_head.refine_rbboxes(
                        roi2droi(rois), bbox_targets[0],
                        bbox_results['bbox_pred'], pos_is_gts, img_metas
                )
            return bbox_results, proposal_list
        elif self.with_rbbox:
            rois = dbbox2roi([res.bboxes for res in sampling_results])
            bbox_results = self._bbox_forward(x, rois)
            bbox_targets = self.rbbox_head.get_targets_rbboxes(
                                        sampling_results,
                                        self.train_cfg)

            loss_bbox = self.rbbox_head.loss(
                                        bbox_results['cls_score'],
                                        bbox_results['bbox_pred'],
                                        *bbox_targets)

            for name, value in loss_bbox.items():
                bbox_results['{}'.format(name)] = (value)
            return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = dbbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    rescale=False):
        """Adopted from original roi transformer"""
        assert self.with_bbox or self.with_rbbox, 'bbox or rbbox head must be implemented.'

        if self.with_bbox:

            rois = bbox2roi(proposal_list)
            bbox_results = self._bbox_forward(x, rois)
            bbox_label = bbox_results['cls_score'].argmax(dim=1)
            # box_head: regress_by_class
            # correct the rois with bbox_results (deltas/deviation as prediction)
            rrois = self.bbox_head.regress_by_class_rbbox(
                roi2droi(rois), bbox_label, bbox_results['bbox_pred'],img_metas[0])
            return rrois

        if self.with_rbbox:
            scale_factors = img_metas[0]['scale_factor']
            rrois = proposal_list
            rbbox_results = self._bbox_forward(x, rrois)

            det_rbboxes, det_labels = self.rbbox_head.get_rbboxes(
                                        rrois,
                                        rbbox_results['cls_score'],
                                        rbbox_results['bbox_pred'],
                                        scale_factors,
                                        rescale=rescale,
                                        cfg=self.test_cfg.rcnn)
            rbbox_results = dbbox2result(det_rbboxes, det_labels,
                                        self.rbbox_head.num_classes)
        
        if not self.with_mask:
            return rbbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_rbboxes, det_labels, rescale=rescale)
            return list(zip(rbbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test (eval) with augmentations.
        Adopted from original roi transformer
        If rescale is False, then returned bboxes will fit the scale
        of imgs[0].
        """
        assert self.with_bbox or self.with_rbbox, 'bbox or rbbox head must be implemented.'

        if self.with_bbox:
            rrois_list = []
            for x, img_meta in zip(x, img_metas):
                # only one image in the batch
                img_shape = img_meta[0]['img_shape']
                scale_factor = img_meta[0]['scale_factor']
                flip = img_meta[0]['flip']

                proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                        scale_factor, flip)

                # angle = img_meta[0]['angle']
                # # print('img shape: ', img_shape)
                # if angle != 0:
                #     proposals = bbox_rotate_mapping(proposal_list[0][:, :4], img_shape,
                #                                     angle)

                rrois_list.append(bbox2roi([proposals]))
                return rrois_list

        elif self.with_rbbox:
            rcnn_test_cfg = self.test_cfg.rcnn
            aug_rbboxes = []
            aug_rscores = []
            rrois_list = proposal_list
            for x, img_meta, rois in zip(x, img_metas, rrois_list):
                scale_factor = img_meta[0]['scale_factor']
                # recompute feature maps to save GPU memory
                roi_feats = self.bbox_roi_extractor(
                    x[:len(self.bbox_roi_extractor.featmap_strides)], rois)

                cls_score, bbox_pred = self.rbbox_head(roi_feats)

                bbox_label = cls_score.argmax(dim=1)
                rrois = self.rbbox_head.regress_by_class_rbbox(roi2droi(rois), bbox_label,
                                                            bbox_pred,
                                                            img_meta[0])
            
                rbbox_results = self._bbox_forward(x, rrois)

                rbboxes, rscores = self.rbbox_head.get_rbboxes(
                    rrois,
                    rbbox_results['cls_score'],
                    rbbox_results['bbox_pred'],
                    scale_factor,
                    rescale=rescale,
                    cfg=None) # ignore rnms
                aug_rbboxes.append(rbboxes)
                aug_rscores.append(rscores)

            merged_rbboxes, merged_rscores = merge_rotate_aug_bboxes(
                aug_rbboxes, aug_rscores, img_metas, rcnn_test_cfg
            )

            # was 'multiclass_nms_rbbox'
            det_rbboxes, det_rlabels = multiclass_rnms(
                                    merged_rbboxes, merged_rscores, rcnn_test_cfg.score_thr,
                                    rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)

            if rescale:
                _det_rbboxes = det_rbboxes
            else:
                _det_rbboxes = det_rbboxes.clone()
                _det_rbboxes[:, :4] *= img_metas[0][0]['scale_factor']

            rbbox_results = dbbox2result(_det_rbboxes, det_rlabels,
                                        self.rbbox_head.num_classes)
            return rbbox_results
