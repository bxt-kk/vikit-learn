from typing import List, Any, Dict, Tuple, Mapping
import math

from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import (
    generalized_box_iou_loss,
    boxes as box_ops,
)

from PIL import Image

from .detector import Detector
from .trimnetx2 import TrimNetX
from .component import ConvNormActive, DetPredictorV2
from ..utils.focal_boost import focal_boost_loss, focal_boost_positive


class TrimNetDet(Detector):
    '''A light-weight and easy-to-train model for object detection

    Args:
        categories: Target categories.
        bbox_limit: Maximum size limit of bounding box.
        anchors: Preset anchor boxes.
        num_waves: Number of the global wave blocks.
        wave_depth: Depth of the wave block.
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
            num_waves:           int=3,
            wave_depth:          int=3,
            num_tries:           int=3,
            swap_size:           int=16,
            dropout:             float=0.1,
            backbone:            str='mobilenet_v3_small',
            backbone_pretrained: bool=True,
        ):

        super().__init__(
            categories, bbox_limit=bbox_limit, anchors=anchors)

        assert num_tries == num_waves

        self.num_tries  = num_tries
        self.swap_size  = swap_size
        self.dropout    = dropout

        self.trimnetx = TrimNetX(
            num_waves, wave_depth, backbone, backbone_pretrained)

        self.cell_size = self.trimnetx.cell_size

        merged_dim = self.trimnetx.merged_dim
        expanded_dim = merged_dim * 4

        object_dim = (1 + self.bbox_dim + self.num_classes)
        predict_dim = object_dim * self.num_anchors

        self.predict = DetPredictorV2(
            in_planes=merged_dim,
            hidden_planes=expanded_dim,
            num_anchors=self.num_anchors,
            bbox_dim=self.bbox_dim,
            num_classes=self.num_classes,
        )

    def train_features(self, flag:bool):
        self.trimnetx.train_features(flag)

    def forward(self, x:Tensor) -> Tensor:
        hs = self.trimnetx(x)
        n, _, rs, cs = hs[0].shape

        # y = self.predict(hs[0])
        # y = y.view(n, self.num_anchors, -1, rs, cs)
        # y = y.permute(0, 1, 3, 4, 2)
        p = self.predict(hs[0])

        # p = y
        ps = [p[..., :1]]
        times = self.num_tries
        for t in range(1, times):

            # y = self.predict(hs[t])
            # y = y.view(n, self.num_anchors, -1, rs, cs)
            # y = y.permute(0, 1, 3, 4, 2)
            y = self.predict(hs[t])

            a = torch.sigmoid(ps[-1])
            p = y * a + p * (1 - a)
            ps.append(p[..., :1])
        ps.append(p[..., 1:])
        return torch.cat(ps, dim=-1)

    @classmethod
    def load_from_state(cls, state:Mapping[str, Any]) -> 'TrimNetDet':
        hyps = state['hyperparameters']
        model = cls(
            categories          = hyps['categories'],
            bbox_limit          = hyps['bbox_limit'],
            anchors             = hyps['anchors'],
            num_tries           = hyps['num_tries'],
            swap_size           = hyps['swap_size'],
            dropout             = hyps['dropout'],
            num_waves           = hyps['num_waves'],
            wave_depth          = hyps['wave_depth'],
            backbone            = hyps['backbone'],
            backbone_pretrained = False,
        )
        model.load_state_dict(state['model'])
        return model

    def hyperparameters(self) -> Dict[str, Any]:
        return dict(
            categories = self.categories,
            bbox_limit = self.bbox_limit,
            anchors    = self.anchors,
            num_tries  = self.num_tries,
            swap_size  = self.swap_size,
            dropout    = self.dropout,
            num_waves  = self.trimnetx.num_waves,
            wave_depth = self.trimnetx.wave_depth,
            backbone   = self.trimnetx.backbone,
        )

    def load_state_dict(
            self,
            state_dict: Mapping[str, Any],
            strict:     bool=True,
            assign:     bool=False,
        ):
        # CLSS_WEIGHT_KEY = 'predict_objs.predict_clss.3.weight'
        # CLSS_BIAS_KEY = 'predict_objs.predict_clss.3.bias'
        #
        # clss_weight = state_dict.pop(CLSS_WEIGHT_KEY)
        # clss_bias = state_dict.pop(CLSS_BIAS_KEY)
        # if clss_bias.shape[0] == self.num_anchors * self.num_classes:
        #     state_dict[CLSS_WEIGHT_KEY] = clss_weight
        #     state_dict[CLSS_BIAS_KEY] = clss_bias
        super().load_state_dict(state_dict, strict, assign)

    def detect(
            self,
            image:         Image.Image,
            conf_thresh:   float=0.6,
            recall_thresh: float=0.5,
            iou_thresh:    float=0.5,
            align_size:    int=448,
            mini_side:     int=1,
        ) -> List[Dict[str, Any]]:

        device = self.get_model_device()
        x, scale, pad_x, pad_y = self.preprocess(
            image, align_size, limit_size=32, fill_value=127)
        x = x.to(device)

        predicts = self.forward(x)
        conf_prob = torch.ones_like(predicts[..., 0])
        for conf_id in range(self.num_tries - 1):
            conf_prob[torch.sigmoid(predicts[..., conf_id]) < recall_thresh] = 0.
        conf_prob *= torch.sigmoid(predicts[..., self.num_tries - 1])
        pred_objs = predicts[..., self.num_tries:]

        index = torch.nonzero(conf_prob > recall_thresh, as_tuple=True)
        if len(index[0]) == 0: return []
        conf = conf_prob[index[0], index[1], index[2], index[3]]
        objs = pred_objs[index[0], index[1], index[2], index[3]]

        raw_w, raw_h = image.size
        boxes = self.pred2boxes(objs[:, :self.bbox_dim], index)
        boxes[:, 0] = torch.clamp((boxes[:, 0] - pad_x) / scale, 0, raw_w - 1)
        boxes[:, 1] = torch.clamp((boxes[:, 1] - pad_y) / scale, 0, raw_h - 1)
        boxes[:, 2] = torch.clamp((boxes[:, 2] - pad_x) / scale, 1, raw_w)
        boxes[:, 3] = torch.clamp((boxes[:, 3] - pad_y) / scale, 1, raw_h)

        clss = torch.softmax(objs[:, self.bbox_dim:], dim=-1).max(dim=-1)
        labels, probs = clss.indices, clss.values
        scores = conf * probs
        final_ids = box_ops.batched_nms(boxes, scores, labels, iou_thresh)
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
        num_confs = self.num_tries # len(self.predict_conf_tries)

        conf_loss, sampled_loss = focal_boost_loss(
            inputs, target_index, num_confs, alpha, gamma)

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
            recall_thresh: float=0.5,
            eps:           float=1e-5,
        ) -> Dict[str, Any]:

        num_confs = self.num_tries # len(self.predict_conf_tries)

        targ_conf = torch.zeros_like(inputs[..., 0])
        targ_conf[target_index] = 1.

        pred_obj = focal_boost_positive(
            inputs, num_confs, conf_thresh, recall_thresh)

        pred_obj_true = torch.masked_select(targ_conf, pred_obj).sum()
        conf_precision = pred_obj_true / torch.clamp_min(pred_obj.sum(), eps)
        conf_recall = pred_obj_true / torch.clamp_min(targ_conf.sum(), eps)
        conf_f1 = 2 * conf_precision * conf_recall / torch.clamp_min(conf_precision + conf_recall, eps)
        proposals = pred_obj.sum() / pred_obj.shape[0]

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
            recall_thresh: float=0.5,
            iou_thresh:    float=0.5,
        ):

        num_confs = self.num_tries # len(self.predict_conf_tries)

        preds_mask = focal_boost_positive(
            inputs, num_confs, conf_thresh, recall_thresh)
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
