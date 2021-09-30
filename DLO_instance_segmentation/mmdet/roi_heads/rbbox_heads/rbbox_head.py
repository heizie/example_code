import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmdet.core import (choose_best_Rroi_batch, hbb2obb)
from mmdet.models.builder import HEADS, build_loss
from mmdet.core import (build_bbox_coder, multi_apply, multiclass_rnms, 
                choose_best_match_batch)
from mmdet.models.losses import accuracy

@HEADS.register_module
class RBBoxHead(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively"""

    def __init__(self,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=2,
                 bbox_coder=dict(
                    type='DeltaXYLSTBBoxCoder',
                    target_means=[0., 0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2, 0.1]),
                 reg_class_agnostic=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0)):
        super(RBBoxHead, self).__init__()
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.target_means = bbox_coder['target_means']
        self.target_stds = bbox_coder['target_stds']
        self.reg_class_agnostic = reg_class_agnostic

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        in_channels = self.in_channels
        if self.with_avg_pool:
            # TODO: finish the tuple condition
            self.avg_pool = nn.AvgPool2d(roi_feat_size)
        else:
            if isinstance(self.roi_feat_size, int):
                in_channels *= (self.roi_feat_size * self.roi_feat_size)
            elif isinstance(self.roi_feat_size, tuple):
                assert len(self.roi_feat_size) == 2
                assert isinstance(self.roi_feat_size[0], int)
                assert isinstance(self.roi_feat_size[1], int)
                in_channels *= (self.roi_feat_size[0] * self.roi_feat_size[1])
        if self.with_cls:
            # mismatch from original, num class
            # should +1 for back ground. HZ
            self.fc_cls = nn.Linear(in_channels, num_classes+1)
        if self.with_reg:
            out_dim_reg = 5 if reg_class_agnostic else 5 * num_classes
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)
        self.debug_imgs = None

    def init_weights(self):
        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)

    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred

    # bbox_target_rbbox and bbox_target_rbbox_single
    def _get_target_bboxes2rbbox_single(self,
                        pos_bboxes,
                        neg_bboxes,
                        pos_assigned_gt_inds,
                        gt_rbboxes,
                        pos_gt_labels,
                        cfg):
        """
        :param pos_bboxes: Tensor, shape (n, 4) (xyxy)
        :param neg_bboxes: Tensor, shape (m, 4)
        :param pos_assigned_gt_inds: Tensor, shape (n)
        :param gt_rbboxes: (xywht)
        :param pos_gt_labels:   Tensor, shape (n)
        :param cfg: dict, cfg.pos_weight = -1
        :return:
        """

        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        # labels = pos_bboxes.new_zeros(num_samples, dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 5)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 5)

        # no longer need to transform mask to obb       
        # choose the gt bbox acoording to sampler results
        pos_gt_bboxes = gt_rbboxes[pos_assigned_gt_inds, :]
        
        # print('pos_bboxes',pos_bboxes)
        if pos_bboxes.size(1) == 4:
            # pos_bboxes is x1y1x2y2, we need xywht
            pos_ext_bboxes = hbb2obb(pos_bboxes)
        else:
            pos_ext_bboxes = pos_bboxes
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight

            pos_bbox_targets = self.bbox_coder.encode(
                pos_ext_bboxes, pos_gt_bboxes) # xywht
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    # get_target
    def get_targets_bboxes2rbbox(self,
                                 sampling_results, 
                                 gt_rbboxes, 
                                 rcnn_train_cfg, 
                                 concat=True):
        """
        obb target hbb
        :param sampling_results:
        :param gt_rbboxes:
        :param rcnn_train_cfg:
        :param mod: 'normal' or 'best_match', 'best_match' is used for RoI Transformer
        :return:
        """
        pos_boxes_list = [res.pos_bboxes for res in sampling_results]
        neg_boxes_list = [res.neg_bboxes for res in sampling_results]
        # pos_gt_boxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_assigned_gt_inds_list = [
            res.pos_assigned_gt_inds  for res in sampling_results
        ]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        # print('rcnn_train_cfg',rcnn_train_cfg)
        labels, label_weights, box_targets, box_weights = multi_apply(
            self._get_target_bboxes2rbbox_single,
            pos_boxes_list, # xyxy
            neg_boxes_list, # xyxy
            pos_assigned_gt_inds_list, # for obb
            gt_rbboxes,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            box_targets = torch.cat(box_targets, 0)
            box_weights = torch.cat(box_weights, 0)
        return labels, label_weights, box_targets, box_weights

    def _get_target_rbbox_single(self,
                        pos_rbboxes,
                        neg_rbboxes,
                        pos_gt_rbboxes,
                        pos_gt_labels,
                        cfg):
        """

        :param pos_bboxes:
        :param neg_bboxes:
        :param gt_masks:
        :param pos_gt_labels:
        :param cfg:
        :param reg_classes:
        :param target_means:
        :param target_stds:
        :return:
        """
        assert pos_rbboxes.size(1) == 5
        num_pos = pos_rbboxes.size(0)
        num_neg = neg_rbboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        # labels = pos_rbboxes.new_zeros(num_samples, dtype=torch.long)
        labels = pos_rbboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_rbboxes.new_zeros(num_samples)
        bbox_targets = pos_rbboxes.new_zeros(num_samples, 5)
        bbox_weights = pos_rbboxes.new_zeros(num_samples, 5)

        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight

            # let the points of gt sorted as close as output bbox
            # pos_gt_rbboxes = choose_best_match_batch(pos_rbboxes, pos_gt_rbboxes)

            # output delta
            pos_bbox_targets = self.bbox_coder.encode(
                pos_rbboxes, pos_gt_rbboxes)

            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets_rbboxes(self, sampling_results, rcnn_train_cfg, concat=True):
        """
        obb target obb
        :param sampling_results:
        :param gt_bboxes:
        :param gt_labels:
        :param rcnn_train_cfg:
        :return:
        """
        pos_proposals_list = [res.pos_bboxes for res in sampling_results]
        neg_proposals_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        # reg_classes = 1 if self.reg_class_agnostic else self.num_classes

        labels, label_weights, rbbox_targets, rbbox_weights = multi_apply(
            self._get_target_rbbox_single,
            pos_proposals_list,
            neg_proposals_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            rbbox_targets = torch.cat(rbbox_targets, 0)
            rbbox_weights = torch.cat(rbbox_weights, 0)
        return labels, label_weights, rbbox_targets, rbbox_weights

    def loss(self,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
        if cls_score.numel() > 0:
            losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
            losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None: 
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 5)[pos_inds.type(torch.bool)]
                # pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 5)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        5)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                # pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                #                                5)[pos_inds, labels[pos_inds]]
            losses['loss_bbox'] = self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds.type(torch.bool)],
                bbox_weights[pos_inds.type(torch.bool)],
                avg_factor=bbox_targets.size(0))
        return losses

    # not used...
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        # TODO: check and simplify it
        if rois.size(1) == 5:
            obbs = hbb2obb(rois[:, 1:])
        elif rois.size(1) == 6:
            obbs = rois[:, 1:]
        else:
            print('strange size')
        if bbox_pred is not None:
                dbboxes = self.bbox_coder.decode(obbs, bbox_pred, self.target_means,
                                         self.target_stds)
        else:
            dbboxes = obbs

        if rescale:
            dbboxes[:, 0::5] /= scale_factor
            dbboxes[:, 1::5] /= scale_factor
            dbboxes[:, 2::5] /= scale_factor
            dbboxes[:, 3::5] /= scale_factor

        det_bboxes, det_labels = multiclass_rnms(dbboxes, scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return det_bboxes, det_labels
    # for test (eval), old version not suppport batch
    # moded for batch mode
    def get_rbboxes(self,
                    rrois,
                    cls_score,
                    rbbox_pred,
                    scale_factor,
                    rescale=False,
                    cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=-1) if cls_score is not None else None

        if rbbox_pred is not None:
            bboxes = self.bbox_coder.decode(rrois[..., 1:], rbbox_pred)
        else:
            bboxes = rrois[:, 1:].clone()

        if rescale:
            bboxes[..., 0::5] /= scale_factor
            bboxes[..., 1::5] /= scale_factor
            bboxes[..., 2::5] /= scale_factor
            bboxes[..., 3::5] /= scale_factor
        # if rescale and bboxes.size(0) > 0:
        #     scale_factor = bboxes.new_tensor(scale_factor)
        #     bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
        #         bboxes.size()[0], -1)
        if cfg is None:
            return bboxes, scores
        else:
            # det_bboxes, det_labels = bboxes, scores
            det_bboxes, det_labels = multiclass_rnms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)
        # if not batch_mode:
        #     det_bboxes = det_bboxes[0]
        #     det_labels = det_labels[0]
        return det_bboxes, det_labels

    def refine_rbboxes(self, rois, labels, rbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 5) or (n*bs, 5*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() <= len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(
                rois[:, 0] == i, as_tuple=False).squeeze(dim=1)
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            rbox_pred_ = rbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class_rbbox(bboxes_, label_, rbox_pred_,
                                           img_meta_)
            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds.type(torch.bool)])

        return bboxes_list

    def regress_by_class_rbbox(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): shape (n, 5) or (n, 6)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 5*(#class+1)) or (n, 5)
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """

        assert rois.size(1) == 5 or rois.size(1) == 6

        if not self.reg_class_agnostic:
            label = label * 5
            inds = torch.stack((label, label + 1, label + 2, label + 3, label + 4), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 5

        max_shape = img_meta['img_shape']

        if rois.size(1) == 5:
            new_rois = self.bbox_coder.decode(rois, bbox_pred)
            # choose best Rroi
            # new_rois = choose_best_Rroi_batch(new_rois)
        else:
            bboxes = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
            # bboxes = choose_best_Rroi_batch(bboxes)
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois