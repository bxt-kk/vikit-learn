from typing import List, Any, Dict, Tuple, Mapping
import math

from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import (
    sigmoid_focal_loss,
    generalized_box_iou_loss,
    boxes as box_ops,
)
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
# from torchvision.ops.misc import SqueezeExcitation
# from torchvision.models.mobilenetv3 import InvertedResidual

from PIL import Image

from .component import LinearBasicConvBD, CSENet, BasicConvBD #, LocalSqueezeExcitation
from .component import DetPredictor
from .detector import Detector


class TrimNetDet(Detector):
    '''A light-weight and easy-to-train model for object detection

    Args:
        categories: Target categories.
        bbox_limit: Maximum size limit of bounding box.
        anchors: Preset anchor boxes.
        dilation_depth: Depth of dilation module.
        dilation_range: The impact region of dilation convolution.
        num_tries: Number of attempts to guess.
        swap_size: Dimensions of the exchanged data.
        dropout: Dropout parameters in the classifier.
        backbone: Specify a basic model as a feature extraction module.
        backbone_pretrained: Whether to load backbone pretrained weights.
    '''

    def __init__(
            self,
            categories:          List[str],
            bbox_limit:          int=640,
            anchors:             List[Tuple[float, float]] | Tensor | None=None,
            dilation_depth:      int=4,
            dilation_range:      int=4,
            num_tries:           int=3,
            swap_size:           int=16,
            dropout:             float=0.1,
            backbone:            str='mobilenet_v3_small',
            backbone_pretrained: bool=True,
        ):

        super().__init__(
            categories, bbox_limit=bbox_limit, anchors=anchors)

        self.dilation_depth = dilation_depth
        self.dilation_range = dilation_range
        self.num_tries      = num_tries
        self.swap_size      = swap_size
        self.dropout        = dropout
        self.backbone       = backbone

        if backbone == 'mobilenet_v3_small':
            features = mobilenet_v3_small(
                weights=MobileNet_V3_Small_Weights.DEFAULT
                if backbone_pretrained else None,
            ).features

            # for m in features:
            #     if not isinstance(m, InvertedResidual): continue
            #     block:nn.Sequential = m.block
            #     _ids = []
            #     for idx, child in block.named_children():
            #         if not isinstance(child, SqueezeExcitation): continue
            #         _ids.append(int(idx))
            #     for idx in _ids:
            #         block[idx] = LocalSqueezeExcitation.load_from_se_module(block[idx])

            features_dim = 24 * 4 + 48 + 96
            merged_dim   = 160
            expanded_dim = 320

            self.features_d = features[:4] # 24, 64, 64
            self.features_c = features[4:9] # 48, 32, 32
            self.features_u = features[9:-1] # 96, 16, 16

        elif backbone == 'mobilenet_v2':
            features = mobilenet_v2(
                weights=MobileNet_V2_Weights.DEFAULT
                if backbone_pretrained else None,
            ).features

            features_dim = 32 * 4 + 96 + 320
            merged_dim   = 320
            expanded_dim = 640

            self.features_d = features[:7] # 32, 64, 64
            self.features_c = features[7:14] # 96, 32, 32
            self.features_u = features[14:-1] # 320, 16, 16

        self.merge = nn.Sequential(
            nn.Conv2d(features_dim, merged_dim, 1, bias=False),
            nn.BatchNorm2d(merged_dim),
        )

        self.cluster = nn.ModuleList()
        self.csenets = nn.ModuleList()
        for _ in range(dilation_depth):
            modules = []
            for r in range(dilation_range):
                modules.append(
                    LinearBasicConvBD(merged_dim, merged_dim, dilation=2**r))
            modules.append(nn.Hardswish(inplace=True))
            self.cluster.append(nn.Sequential(*modules))
            self.csenets.append(CSENet(
                merged_dim * 2, merged_dim, kernel_size=3, shrink_factor=4))

        ex_anchor_dim = (swap_size + 1) * self.num_anchors

        self.predict_conf_tries = nn.ModuleList([nn.Sequential(
            BasicConvBD(merged_dim, merged_dim, kernel_size=3),
            nn.Dropout(p=dropout, inplace=True),
            nn.Conv2d(merged_dim, ex_anchor_dim, kernel_size=1),
        )])
        for _ in range(1, num_tries):
            self.predict_conf_tries.append(nn.Sequential(
                BasicConvBD(merged_dim + ex_anchor_dim, merged_dim, kernel_size=3),
                nn.Dropout(p=dropout, inplace=True),
                nn.Conv2d(merged_dim, ex_anchor_dim, kernel_size=1),
            ))

        # object_dim = self.bbox_dim + self.num_classes
        # self.predict_objs = nn.Sequential(
        #     nn.Conv2d(merged_dim + ex_anchor_dim, expanded_dim, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(expanded_dim),
        #     nn.Hardswish(inplace=True),
        #     nn.Dropout(p=dropout, inplace=True),
        #     nn.Conv2d(expanded_dim, self.num_anchors * object_dim, kernel_size=1),
        # )
        self.predict_objs = DetPredictor(
            merged_dim + ex_anchor_dim,
            expanded_dim,
            num_anchors=self.num_anchors,
            bbox_dim=self.bbox_dim,
            num_classes=self.num_classes,
            dropout=dropout,
        )

    def forward_features(self, x:Tensor) -> Tensor:
        if not self._keep_features:
            fd = self.features_d(x)
            fc = self.features_c(fd)
            fu = self.features_u(fc)
        else:
            with torch.no_grad():
                fd = self.features_d(x)
                fc = self.features_c(fd)
                fu = self.features_u(fc)

        x = self.merge(torch.cat([
            F.pixel_unshuffle(fd, 2),
            fc,
            F.interpolate(fu, scale_factor=2, mode='bilinear'),
        ], dim=1))
        for csenet_i, cluster_i in zip(self.csenets, self.cluster):
            x = x + csenet_i(torch.cat([x, cluster_i(x)], dim=1))
        return x

    def forward(self, x:Tensor) -> Tensor:
        x = self.forward_features(x)
        confs = [self.predict_conf_tries[0](x)]
        for layer in self.predict_conf_tries[1:]:
            confs.append(layer(torch.cat([x, confs[-1]], dim=1)))
        p_objs = self.predict_objs(torch.cat([x, confs[-1]], dim=1))
        bs, _, ny, nx = p_objs.shape
        p_tryx = torch.cat([
            conf.view(bs, self.num_anchors, -1, ny, nx)[:, :, :1]
            for conf in confs], dim=2)
        p_objs = p_objs.view(bs, self.num_anchors, -1, ny, nx)
        return torch.cat([p_tryx, p_objs], dim=2).permute(0, 1, 3, 4, 2).contiguous()

    @classmethod
    def load_from_state(cls, state:Mapping[str, Any]) -> 'TrimNetDet':
        hyps = state['hyperparameters']
        model = cls(
            categories          = hyps['categories'],
            bbox_limit          = hyps['bbox_limit'],
            anchors             = hyps['anchors'],
            dilation_depth      = hyps['dilation_depth'],
            dilation_range      = hyps['dilation_range'],
            num_tries           = hyps['num_tries'],
            swap_size           = hyps['swap_size'],
            dropout             = hyps['dropout'],
            backbone            = hyps['backbone'],
            backbone_pretrained = False,
        )
        model.load_state_dict(state['model'])
        return model

    def hyperparameters(self) -> Dict[str, Any]:
        return dict(
            categories     = self.categories,
            bbox_limit     = self.bbox_limit,
            anchors        = self.anchors,
            dilation_depth = self.dilation_depth,
            dilation_range = self.dilation_range,
            num_tries      = self.num_tries,
            swap_size      = self.swap_size,
            dropout        = self.dropout,
            backbone       = self.backbone,
        )

    def load_state_dict(
            self,
            state_dict: Mapping[str, Any],
            strict:     bool=True,
            assign:     bool=False,
        ):
        CLSS_WEIGHT_KEY = 'predict_objs.predict_clss.4.weight'
        CLSS_BIAS_KEY = 'predict_objs.predict_clss.4.bias'

        clss_weight = state_dict.pop(CLSS_WEIGHT_KEY)
        clss_bias = state_dict.pop(CLSS_BIAS_KEY)
        if clss_bias.shape[0] == self.num_anchors * self.num_classes:
            state_dict[CLSS_WEIGHT_KEY] = clss_weight
            state_dict[CLSS_BIAS_KEY] = clss_bias
        super().load_state_dict(state_dict, strict, assign)

    def detect(
            self,
            image:       Image.Image,
            conf_thresh: float=0.6,
            iou_thresh:  float=0.55,
            align_size:  int=448,
            mini_side:   int=1,
        ) -> List[Dict[str, Any]]:
        # RV-240605

        device = self.get_model_device()
        x, scale, pad_x, pad_y = self.preprocess(
            image, align_size, limit_size=32, fill_value=127)
        x = x.to(device)
        x = self.forward_features(x)

        confs = [self.predict_conf_tries[0](x)]
        for layer in self.predict_conf_tries[1:]:
            confs.append(layer(torch.cat([x, confs[-1]], dim=1)))
        bs, _, ny, nx = x.shape
        p_tryx = torch.cat([
            conf.view(bs, self.num_anchors, -1, ny, nx)[:, :, :1]
            for conf in confs], dim=2).permute(0, 1, 3, 4, 2)
        mix = torch.cat([x, confs[-1]], dim=1)

        p_conf = torch.ones_like(p_tryx[..., 0])
        for conf_id in range(p_tryx.shape[-1] - 1):
            p_conf[p_tryx[..., conf_id] < 0] = 0.
        p_conf *= torch.sigmoid(p_tryx[..., -1])

        mask = p_conf.max(dim=1, keepdim=True).values > conf_thresh
        index = torch.nonzero(mask, as_tuple=True)
        if len(index[0]) == 0: return []

        p_objs = self.predict_objs(
            mix[index[0], :, index[2], index[3]].reshape(len(index[0]), -1, 1, 1))

        p_objs = p_objs.reshape(len(index[0]), self.num_anchors, -1)

        anchor_mask = p_conf[index[0], :, index[2], index[3]] > conf_thresh
        sub_ids, anchor_ids = torch.nonzero(anchor_mask, as_tuple=True)
        # bids = index[0][sub_ids]
        rids = index[2][sub_ids]
        cids = index[3][sub_ids]
        objs = p_objs[sub_ids, anchor_ids]
        conf = p_conf[index[0], :, index[2], index[3]][sub_ids, anchor_ids]

        cx = (cids + torch.tanh(objs[:, 0]) + 0.5) * self.cell_size
        cy = (rids + torch.tanh(objs[:, 1]) + 0.5) * self.cell_size

        regions = self.regions.type_as(objs)
        rw = (
            torch.tanh(objs[:, 2 + 0]) +
            (objs[:, 2 + 1:2 + 7].softmax(dim=-1) * regions).sum(dim=-1)
        ) * self.region_scale
        rh = (
            torch.tanh(objs[:, 2 + 7]) +
            (objs[:, 2 + 8:2 + 14].softmax(dim=-1) * regions).sum(dim=-1)
        ) * self.region_scale
        x1, y1 = cx - rw / 2, cy - rh / 2
        x2, y2 = x1 + rw, y1 + rh

        raw_w, raw_h = image.size
        x1 = torch.clamp((x1 - pad_x) / scale, 0, raw_w - 1)
        y1 = torch.clamp((y1 - pad_y) / scale, 0, raw_h - 1)
        x2 = torch.clamp((x2 - pad_x) / scale, 1, raw_w)
        y2 = torch.clamp((y2 - pad_y) / scale, 1, raw_h)

        boxes = torch.stack([x1, y1, x2, y2]).T
        clss = torch.softmax(objs[:, self.bbox_dim:], dim=-1).max(dim=-1)
        labels, probs = clss.indices, clss.values
        scores = conf * probs
        final_ids = box_ops.batched_nms(boxes, scores, labels, iou_thresh)
        # bids = bids[final_ids]
        boxes = boxes[final_ids]
        labels = labels[final_ids]
        probs = probs[final_ids]
        scores = scores[final_ids]

        result = []
        for score, box, label, prob in zip(scores, boxes, labels, probs):
            if score < conf_thresh: continue
            if (box[2:] - box[:2]).min() < mini_side: continue
            result.append(dict(
                score=round(score.item(), 5),
                box=box.round().tolist(),
                label=self.categories[label],
                prob=round(prob.item(), 5),
            ))
        return result

    def focal_boost(
            self,
            inputs:       Tensor,
            target_index: List[Tensor],
            sample_mask:  Tensor | None,
            conf_id:      int,
            num_confs:    int,
            alpha:        float,
            gamma:        float,
        ) -> Tuple[Tensor, Tensor, Tensor]:
        # RV-240605

        reduction = 'mean'

        pred_conf = inputs[..., conf_id]
        targ_conf = torch.zeros_like(pred_conf)
        targ_conf[target_index] = 1.

        if sample_mask is None:
            sample_mask = targ_conf >= -1

        sampled_pred = torch.masked_select(pred_conf, sample_mask)
        sampled_targ = torch.masked_select(targ_conf, sample_mask)
        sampled_loss = sigmoid_focal_loss(
            inputs=sampled_pred,
            targets=sampled_targ,
            alpha=alpha,
            gamma=gamma,
            reduction=reduction,
        )

        obj_loss = 0.
        # obj_mask = torch.logical_and(sample_mask, targ_conf > 0.5)
        obj_mask = targ_conf > 0.5
        if obj_mask.sum() > 0:
            obj_pred = torch.masked_select(pred_conf, obj_mask)
            obj_targ = torch.masked_select(targ_conf, obj_mask)
            obj_loss = F.binary_cross_entropy_with_logits(
                obj_pred, obj_targ, reduction=reduction)

            obj_pred_min = obj_pred.detach().min()
            sample_mask = pred_conf.detach() >= obj_pred_min

        alpha = (math.cos(math.pi / num_confs * conf_id) + 1) / 2
        num_foreground_per_img = (sampled_targ.sum() / len(pred_conf)).numel()

        conf_loss = (
            obj_loss * alpha / max(1, num_foreground_per_img) +
            sampled_loss) / num_confs

        return conf_loss, sampled_loss, sample_mask

    def calc_loss(
            self,
            inputs:        Tensor,
            target_index:  List[Tensor],
            target_labels: Tensor,
            target_bboxes: Tensor,
            weights:       Dict[str, float] | None=None,
            alpha:         float=0.25,
            gamma:         float=2.,
        ) -> Dict[str, Any]:

        reduction = 'mean'
        num_confs = len(self.predict_conf_tries)

        conf_loss, sampled_loss, sample_mask = self.focal_boost(
            inputs, target_index, None, 0, num_confs, alpha, gamma)
        for conf_id in range(1, num_confs):
            conf_loss_i, sampled_loss, sample_mask = self.focal_boost(
                inputs, target_index, sample_mask, conf_id, num_confs, alpha, gamma)
            conf_loss += conf_loss_i

        pred_conf = inputs[..., 0]
        targ_conf = torch.zeros_like(pred_conf)
        targ_conf[target_index] = 1.

        objects = inputs[target_index]

        bbox_loss = torch.zeros_like(conf_loss)
        clss_loss = torch.zeros_like(conf_loss)
        if objects.shape[0] > 0:
            pred_cxcywh = objects[:, num_confs:num_confs + self.bbox_dim]
            pred_xyxy = self.pred2boxes(pred_cxcywh, target_index)
            bbox_loss = generalized_box_iou_loss(
                pred_xyxy, target_bboxes, reduction=reduction)

            pred_clss = objects[:, num_confs + self.bbox_dim:]
            clss_loss = F.cross_entropy(
                pred_clss, target_labels, reduction=reduction)

        weights = weights or dict()

        loss = (
            weights.get('conf', 1.) * conf_loss +
            weights.get('bbox', 1.) * bbox_loss +
            weights.get('clss', 0.33) * clss_loss
        )

        return dict(
            loss=loss,
            conf_loss=conf_loss,
            bbox_loss=bbox_loss,
            clss_loss=clss_loss,
            sampled_loss=sampled_loss,
        )

    def calc_score(
            self,
            inputs:        Tensor,
            target_index:  List[Tensor],
            target_labels: Tensor,
            target_bboxes: Tensor,
            conf_thresh:   float=0.5,
            eps:           float=1e-5,
        ) -> Dict[str, Any]:

        num_confs = len(self.predict_conf_tries)

        targ_conf = torch.zeros_like(inputs[..., 0])
        targ_conf[target_index] = 1.

        pred_obj = targ_conf > -1
        for conf_id in range(num_confs):
            pred_conf = torch.sigmoid(inputs[..., conf_id])
            thresh = 0.5 if conf_id < num_confs - 1 else conf_thresh
            pred_obj = torch.logical_and(pred_obj, pred_conf > thresh)

        pred_obj_true = torch.masked_select(targ_conf, pred_obj).sum()
        conf_precision = pred_obj_true / torch.clamp_min(pred_obj.sum(), eps)
        conf_recall = pred_obj_true / torch.clamp_min(targ_conf.sum(), eps)
        conf_f1 = 2 * conf_precision * conf_recall / torch.clamp_min(conf_precision + conf_recall, eps)
        proposals = pred_obj.sum() / pred_conf.shape[0]

        objects = inputs[target_index]

        iou_score = torch.ones_like(conf_f1)
        clss_accuracy = torch.ones_like(conf_f1)
        obj_conf_min = torch.zeros_like(conf_f1)
        if objects.shape[0] > 0:
            pred_cxcywh = objects[:, num_confs:num_confs + self.bbox_dim]
            pred_xyxy = self.pred2boxes(pred_cxcywh, target_index)
            targ_xyxy = target_bboxes

            max_x1y1 = torch.maximum(pred_xyxy[:, :2], targ_xyxy[:, :2])
            min_x2y2 = torch.minimum(pred_xyxy[:, 2:], targ_xyxy[:, 2:])
            inter_size = torch.clamp_min(min_x2y2 - max_x1y1, 0)
            intersection = inter_size[:, 0] * inter_size[:, 1]
            pred_size = pred_xyxy[:, 2:] - pred_xyxy[:, :2]
            targ_size = targ_xyxy[:, 2:] - targ_xyxy[:, :2]
            pred_area = pred_size[:, 0] * pred_size[:, 1]
            targ_area = targ_size[:, 0] * targ_size[:, 1]
            union = pred_area + targ_area - intersection
            iou_score = (intersection / union).mean()

            pred_labels = torch.argmax(objects[:, num_confs + self.bbox_dim:], dim=-1)
            clss_accuracy = (pred_labels == target_labels).sum() / len(pred_labels)

            obj_conf = torch.sigmoid(objects[:, :num_confs])
            if num_confs == 1:
                obj_conf_min = obj_conf[:, 0].min()
            else:
                sample_mask = obj_conf[:, 0] > conf_thresh
                for conf_id in range(1, num_confs - 1):
                    sample_mask = torch.logical_and(
                        sample_mask, obj_conf[:, conf_id] > conf_thresh)
                if sample_mask.sum() > 0:
                    obj_conf_min = torch.masked_select(obj_conf[:, -1], sample_mask).min()
                else:
                    obj_conf_min = torch.zeros_like(proposals)

        return dict(
            conf_precision=conf_precision,
            conf_recall=conf_recall,
            conf_f1=conf_f1,
            iou_score=iou_score,
            clss_accuracy=clss_accuracy,
            proposals=proposals,
            obj_conf_min=obj_conf_min,
        )

    def update_metric(
            self,
            inputs:        Tensor,
            target_index:  List[Tensor],
            target_labels: Tensor,
            target_bboxes: Tensor,
            conf_thresh:   float=0.5,
            iou_thresh:    float=0.5,
        ):

        preds_mask = None
        num_confs = len(self.predict_conf_tries)

        preds_mask = torch.sigmoid(inputs[..., 0]) > 0.5
        for conf_id in range(1, num_confs):
            thresh = 0.5 if conf_id < num_confs - 1 else conf_thresh
            preds_mask = torch.logical_and(
                preds_mask, torch.sigmoid(inputs[..., conf_id]) > thresh)
        preds_index = torch.nonzero(preds_mask, as_tuple=True)

        objects = inputs[preds_index]

        pred_scores = torch.sigmoid(objects[:, num_confs - 1])
        pred_cxcywh = objects[:, num_confs:num_confs + self.bbox_dim]
        pred_bboxes = torch.clamp_min(self.pred2boxes(pred_cxcywh, preds_index), 0.)
        pred_labels = torch.argmax(objects[:, num_confs + self.bbox_dim:], dim=-1)

        preds = []
        target = []
        batch_size = inputs.shape[0]
        for batch_id in range(batch_size):
            preds_ids = (preds_index[0] == batch_id).nonzero(as_tuple=True)[0]
            target_ids = (target_index[0] == batch_id).nonzero(as_tuple=True)[0]

            scores=pred_scores[preds_ids]
            boxes=pred_bboxes[preds_ids]
            labels=pred_labels[preds_ids]
            final_ids = box_ops.batched_nms(boxes, scores, labels, iou_thresh)

            preds.append(dict(
                scores=scores[final_ids],
                boxes=boxes[final_ids],
                labels=labels[final_ids],
            ))
            target.append(dict(
                boxes=target_bboxes[target_ids],
                labels=target_labels[target_ids],
            ))
        self.m_ap_metric.update(preds, target)
