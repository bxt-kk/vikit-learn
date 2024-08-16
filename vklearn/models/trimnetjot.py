from typing import List, Any, Dict, Mapping, Tuple

from torch import Tensor
from torchvision.ops import (
    sigmoid_focal_loss,
    generalized_box_iou_loss,
    boxes as box_ops,
)
import torch
import torch.nn.functional as F

from PIL import Image
from numpy import ndarray
import numpy as np

from .joints import Joints
from .trimnetx import TrimNetX
from .component import SegPredictor, DetPredictor
from ..utils.focal_boost import focal_boost_predict


class TrimNetJot(Joints):
    '''A light-weight and easy-to-train model for joints detection

    Args:
        categories: Target categories.
        bbox_limit: Maximum size limit of bounding box.
        num_scans: Number of the Trim-Units.
        scan_range: Range factor of the Trim-Unit convolution.
        backbone: Specify a basic model as a feature extraction module.
        backbone_pretrained: Whether to load backbone pretrained weights.
    '''

    def __init__(
            self,
            categories:          List[str],
            bbox_limit:          int=640,
            num_scans:           int=3,
            scan_range:          int=4,
            backbone:            str='mobilenet_v3_small',
            backbone_pretrained: bool=True,
        ):

        super().__init__(
            categories, bbox_limit=bbox_limit)

        self.trimnetx = TrimNetX(
            num_scans, scan_range, backbone, backbone_pretrained)

        self.cell_size = self.trimnetx.cell_size

        merged_dim = self.trimnetx.merged_dim

        self.seg_predictor = SegPredictor(merged_dim, 1, num_scans)

        self.det_predictor = DetPredictor(
            in_planes=merged_dim + 1,
            num_anchors=self.num_anchors,
            bbox_dim=self.bbox_dim,
            clss_dim=self.num_classes,
            dropout_p=0.,
        )

    def train_features(self, flag:bool):
        self.trimnetx.train_features(flag)

    def forward(self, x:Tensor) -> Tuple[Tensor, Tensor]:
        hs, _ = self.trimnetx(x)
        last_segment = self.seg_predictor(hs)[-1]
        mask = F.hardsigmoid(last_segment)
        mask = F.interpolate(
            mask,
            scale_factor=1 / 2**len(hs),
            mode='bilinear')
        objs = self.det_predictor(torch.cat([mask, hs[-1]], dim=1))
        return last_segment, objs

    @classmethod
    def load_from_state(cls, state:Mapping[str, Any]) -> 'TrimNetJot':
        hyps = state['hyperparameters']
        model = cls(
            categories          = hyps['categories'],
            bbox_limit          = hyps['bbox_limit'],
            num_scans           = hyps['num_scans'],
            scan_range          = hyps['scan_range'],
            backbone            = hyps['backbone'],
            backbone_pretrained = False,
        )
        model.load_state_dict(state['model'])
        return model

    def hyperparameters(self) -> Dict[str, Any]:
        return dict(
            categories = self.categories,
            bbox_limit = self.bbox_limit,
            num_scans  = self.trimnetx.num_scans,
            scan_range = self.trimnetx.scan_range,
            backbone   = self.trimnetx.backbone,
        )

    def load_state_dict(
            self,
            state_dict: Mapping[str, Any],
            strict:     bool=True,
            assign:     bool=False,
        ):
        CLSS_WEIGHT_KEY = 'det_predictor.clss_predict.weight'
        CLSS_BIAS_KEY   = 'det_predictor.clss_predict.bias'

        clss_weight = state_dict.pop(CLSS_WEIGHT_KEY)
        clss_bias   = state_dict.pop(CLSS_BIAS_KEY)
        if clss_bias.shape[0] == self.num_classes:
            state_dict[CLSS_WEIGHT_KEY] = clss_weight
            state_dict[CLSS_BIAS_KEY] = clss_bias
        super().load_state_dict(state_dict, strict, assign)

    def detect(
            self,
            image:         Image.Image,
            conf_thresh:   float=0.5,
            iou_thresh:    float=0.5,
            align_size:    int=448,
            mini_side:     int=1,
        ) -> List[Dict[str, Any]]:

        device = self.get_model_device()
        x, scale, pad_x, pad_y = self.preprocess(
            image, align_size, limit_size=32, fill_value=127)
        x = x.to(device)

        num_confs = self.trimnetx.num_scans

        seg_predicts, det_predicts = self.forward(x)

        pred_objs = det_predicts[..., num_confs:]
        conf_prob = focal_boost_predict(
            det_predicts, num_confs, recall_thresh=conf_thresh)

        index = torch.nonzero(conf_prob > conf_thresh, as_tuple=True)
        if len(index[0]) == 0: return []
        conf = conf_prob[index[0], index[1], index[2], index[3]]
        objs = pred_objs[index[0], index[1], index[2], index[3]]

        raw_w, raw_h = image.size
        boxes = self.pred2boxes(objs[:, :self.bbox_dim], index[2], index[3])

        clss = torch.softmax(objs[:, self.bbox_dim:], dim=-1).max(dim=-1)
        labels, probs = clss.indices, clss.values
        scores = conf * probs
        final_ids = box_ops.batched_nms(boxes, scores, labels, iou_thresh)
        boxes = boxes[final_ids]
        labels = labels[final_ids]
        probs = probs[final_ids]
        scores = scores[final_ids]
        anchors = index[1][final_ids]

        nodes = []
        for score, box, label, prob, anchor in zip(scores, boxes, labels, probs, anchors):
            # if score < conf_thresh: continue
            if (box[2:] - box[:2]).min() < mini_side: continue
            nodes.append(dict(
                score=round(score.item(), 5),
                box=box.round().tolist(),
                label=self.categories[label],
                prob=round(prob.item(), 5),
                anchor=anchor.item(),
            ))

        dst_w, dst_h = round(scale * raw_w), round(scale * raw_h)
        seg_predicts = torch.sigmoid(seg_predicts[..., -1])
        heatmap = seg_predicts[0, 0].cpu().numpy()

        objs, remains = self.joints(nodes, heatmap)

        heatmap = F.interpolate(
            seg_predicts[..., pad_y:pad_y + dst_h, pad_x:pad_x + dst_w],
            size=(raw_h, raw_w),
            mode='bilinear',
        )[0, 0].numpy()

        for ix in range(len(remains)):
            item = remains[ix]
            x1, y1, x2, y2 = item['box']
            x1 = round(min(max((x1 - pad_x) / scale, 0), raw_w - 1))
            y1 = round(min(max((y1 - pad_y) / scale, 0), raw_h - 1))
            x2 = round(min(max((x2 - pad_x) / scale, 1), raw_w))
            y2 = round(min(max((y2 - pad_y) / scale, 1), raw_h))
            item['box'] = x1, y1, x2, y2

        for ix in range(len(objs)):
            (cx, cy), (w, h), a = objs[ix]['rect']
            cx = round(min(max((cx - pad_x) / scale, 0), raw_w))
            cy = round(min(max((cy - pad_y) / scale, 0), raw_h))
            w = w / scale
            h = h / scale
            objs[ix]['rect'] = (cx, cy), (w, h), a

        return dict(
            remains=remains,
            heatmap=heatmap,
            objs=objs,
        )

    def _joint_iter(
            self,
            begin_node: List[Dict[str, Any]],
            end_nodes:  List[Dict[str, Any]],
            heatmap:    ndarray,
        ) -> Tuple[int, float]:

        begin_bbox = np.array(begin_node['box'], dtype=np.float32)
        begin_cxcy = (begin_bbox[:2] + begin_bbox[2:]) * 0.5
        max_score = 0.5
        matched_id = -1
        for end_ix, end_node in enumerate(end_nodes):
            end_bbox = np.array(end_node['box'], dtype=np.float32)
            end_cxcy = (end_bbox[:2] + end_bbox[2:]) * 0.5
            steps = round(np.linalg.norm(begin_cxcy - end_cxcy) / 1)
            cols = np.around(np.linspace(begin_cxcy[0], end_cxcy[0], steps)).astype(int)
            rows = np.around(np.linspace(begin_cxcy[1], end_cxcy[1], steps)).astype(int)
            region = heatmap[rows, cols]
            if region.size == 0:
                continue
            score = region.mean()
            if score > max_score:
                max_score = score
                matched_id = end_ix
            if max_score > 0.99: break
        return matched_id, max_score

    def joints(
            self,
            nodes:   List[Dict[str, Any]],
            heatmap: ndarray,
        ) -> Tuple[List[Any], List[Dict[str, Any]]]:

        begin_nodes = [item for item in nodes if item['anchor'] == 0]
        end_nodes = [item for item in nodes if item['anchor'] == 1]
        matched_pairs = []
        for _ in range(len(begin_nodes)):
            begin_node = begin_nodes.pop(0)
            matched_id, score = self._joint_iter(begin_node, end_nodes, heatmap)
            if matched_id < 0:
                begin_nodes.append(begin_node)
                continue
            end_node = end_nodes.pop(matched_id)
            matched_pairs.append((begin_node, end_node))
            if score < 0.9:
                end_nodes.insert(matched_id, end_node)

        objs = []
        for begin_node, end_node in matched_pairs:
            begin_bbox = np.array(begin_node['box'], dtype=np.float32)
            end_bbox = np.array(end_node['box'], dtype=np.float32)
            begin_cxcy = (begin_bbox[:2] + begin_bbox[2:]) * 0.5
            end_cxcy = (end_bbox[:2] + end_bbox[2:]) * 0.5
            vector = end_cxcy - begin_cxcy
            length = np.linalg.norm(vector)
            vector /= length
            angle = np.rad2deg(np.arccos(vector[0]))
            if vector[1] < 0: angle = -angle
            begin_width = max(begin_bbox[2:] - begin_bbox[:2])
            end_width = max(end_bbox[2:] - end_bbox[:2])
            diameter = (begin_width + end_width) * 0.5
            cx, cy = (begin_cxcy + end_cxcy) * 0.5
            rect = (cx, cy), (length, diameter), angle
            begin_score = begin_node['score']
            end_score = end_node['score']
            label = begin_node['label']
            score = begin_score
            if begin_score < end_score:
                label = end_node['label']
                score = end_score
            objs.append(dict(rect=rect, label=label, score=score))

        return objs, begin_nodes + end_nodes

    def dice_loss(
            self,
            inputs: Tensor,
            target: Tensor,
            eps:    float=1e-5,
        ) -> Tensor:

        predict = torch.sigmoid(inputs).flatten(1)
        ground = target.flatten(1)
        intersection = predict * ground
        dice = (
            intersection.sum(dim=1) * 2 /
            (predict.sum(dim=1) + ground.sum(dim=1) + eps)
        )
        dice_loss = 1 - dice
        return dice_loss

    def _calc_seg_loss(
            self,
            inputs:  Tensor,
            target:  Tensor,
            weights: Dict[str, float] | None=None,
        ) -> Dict[str, Any]:

        rows, cols = inputs.shape[2], inputs.shape[3]
        target = F.interpolate(target.type_as(inputs), (rows, cols), mode='bilinear')

        # 2 * abs(mean - 0.5) >> 0. -> alpha >> 0
        # 2 * abs(mean - 0.5) >> 1. -> alpha >> 0.5
        alpha = torch.square(target.mean(dim=(1, 2, 3)) - 0.5) * 2

        bce = F.binary_cross_entropy_with_logits(
            inputs,
            target,
            reduction='none',
        ).mean(dim=(1, 2, 3))
        dice = self.dice_loss(
            inputs,
            target)

        loss = (alpha * bce + (1 - alpha) * dice).mean()
        bce_loss = bce.mean()
        dice_loss = dice.mean()

        return dict(
            loss=loss,
            bce_loss=bce_loss,
            dice_loss=dice_loss,
        )

    def _calc_det_loss(
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

        pred_conf = inputs[..., 0]
        targ_conf = torch.zeros_like(pred_conf)
        targ_conf[target_index] = 1.

        precision_loss = sigmoid_focal_loss(
            inputs=pred_conf,
            targets=targ_conf,
            alpha=alpha,
            gamma=gamma,
            reduction=reduction,
        )

        objects = inputs[target_index]

        recall_loss = torch.zeros_like(precision_loss)
        bbox_loss   = torch.zeros_like(precision_loss)
        clss_loss   = torch.zeros_like(precision_loss)
        if objects.shape[0] > 0:
            pred_recall = objects[:, 0]
            recall_loss = F.binary_cross_entropy_with_logits(
                pred_recall, torch.ones_like(pred_recall), reduction=reduction)

            pred_cxcywh = objects[:, 1:1 + self.bbox_dim]
            pred_xyxy = self.pred2boxes(pred_cxcywh, target_index[2], target_index[3])
            bbox_loss = generalized_box_iou_loss(
                pred_xyxy, target_bboxes, reduction=reduction)

            pred_clss = objects[:, 1 + self.bbox_dim:]
            clss_loss = F.cross_entropy(
                pred_clss, target_labels, reduction=reduction)

        conf_loss = (1 - alpha) * precision_loss + alpha * recall_loss

        weights = weights or dict()

        loss = (
            weights.get('conf', 1.) * conf_loss +
            weights.get('bbox', 1.) * bbox_loss +
            weights.get('clss', 0.33) * clss_loss
        )

        return dict(
            loss=loss,
            precision_loss=precision_loss,
            recall_loss=recall_loss,
            bbox_loss=bbox_loss,
            clss_loss=clss_loss,
        )

    def calc_loss(
            self,
            inputs:        Tuple[Tensor, Tensor],
            target_index:  List[Tensor],
            target_labels: Tensor,
            target_bboxes: Tensor,
            target_masks:  Tensor,
            weights:       Dict[str, float] | None=None,
            alpha:         float=0.25,
            gamma:         float=2.,
        ) -> Dict[str, Any]:

        inputs_seg, inputs_det = inputs

        seg_losses = self._calc_seg_loss(
            inputs_seg,
            target_masks,
            weights=weights,
        )

        det_losses = self._calc_det_loss(
            inputs_det,
            target_index,
            target_labels,
            target_bboxes,
            weights,
            alpha=alpha,
            gamma=gamma,
        )

        losses = dict()
        losses['loss'] = det_losses['loss'] + seg_losses['loss']
        for _losses in (det_losses, seg_losses):
            for key in _losses.keys():
                if key == 'loss': continue
                losses[key] = _losses[key]

        return losses

    def _calc_seg_score(
            self,
            inputs: Tensor,
            target: Tensor,
            eps:    float=1e-5,
        ) -> Dict[str, Any]:

        rows, cols = inputs.shape[2], inputs.shape[3]
        target = F.interpolate(target.type_as(inputs), (rows, cols), mode='bilinear')

        predicts = torch.sigmoid(inputs)
        distance = torch.abs(predicts - target).mean()

        return dict(
            mae=distance,
        )

    def _calc_det_score(
            self,
            inputs:        Tensor,
            target_index:  List[Tensor],
            target_labels: Tensor,
            target_bboxes: Tensor,
            conf_thresh:   float=0.5,
            eps:           float=1e-5,
        ) -> Dict[str, Any]:

        pred_conf = inputs[..., 0]
        targ_conf = torch.zeros_like(pred_conf)
        targ_conf[target_index] = 1.

        pred_obj = torch.sigmoid(pred_conf) > conf_thresh

        pred_obj_true = torch.masked_select(targ_conf, pred_obj).sum()
        conf_precision = pred_obj_true / torch.clamp_min(pred_obj.sum(), eps)
        conf_recall = pred_obj_true / torch.clamp_min(targ_conf.sum(), eps)
        conf_f1 = 2 * conf_precision * conf_recall / torch.clamp_min(conf_precision + conf_recall, eps)
        proposals = pred_obj.sum() / pred_obj.shape[0]

        objects = inputs[target_index]

        iou_score = torch.ones_like(conf_f1)
        clss_accuracy = torch.ones_like(conf_f1)
        conf_min = torch.zeros_like(conf_f1)
        if objects.shape[0] > 0:
            pred_cxcywh = objects[:, 1:1 + self.bbox_dim]
            pred_xyxy = self.pred2boxes(pred_cxcywh, target_index[2], target_index[3])
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

            pred_labels = torch.argmax(objects[:, 1 + self.bbox_dim:], dim=-1)
            clss_accuracy = (pred_labels == target_labels).sum() / len(pred_labels)

            conf_min = torch.sigmoid(objects[:, 0].min())

        return dict(
            conf_precision=conf_precision,
            conf_recall=conf_recall,
            conf_f1=conf_f1,
            iou_score=iou_score,
            clss_accuracy=clss_accuracy,
            proposals=proposals,
            conf_min=conf_min,
        )

    def calc_score(
            self,
            inputs:        Tuple[Tensor, Tensor],
            target_index:  List[Tensor],
            target_labels: Tensor,
            target_bboxes: Tensor,
            target_masks:  Tensor,
            conf_thresh:   float=0.5,
            recall_thresh: float=0.5,
            eps:           float=1e-5,
        ) -> Dict[str, Any]:

        inputs_seg, inputs_det = inputs

        seg_scores = self._calc_seg_score(
            inputs_seg,
            target_masks,
            eps=eps,
        )

        det_scores = self._calc_det_score(
            inputs_det,
            target_index,
            target_labels,
            target_bboxes,
            conf_thresh=conf_thresh,
            eps=eps,
        )

        scores = dict()
        for _scores in (det_scores, seg_scores):
            for key, value in _scores.items():
                scores[key] = value
        return scores

    def _update_seg_metric(
            self,
            inputs:      Tensor,
            target:      Tensor,
            conf_thresh: float=0.5,
        ):

        rows, cols = inputs.shape[2], inputs.shape[3]
        target = F.interpolate(target.type_as(inputs), (rows, cols), mode='bilinear')

        predicts = torch.sigmoid(inputs) > conf_thresh
        self.m_iou_metric.update(predicts.to(torch.int), target.to(torch.int))

    def _update_det_metric(
            self,
            inputs:        Tensor,
            target_index:  List[Tensor],
            target_labels: Tensor,
            target_bboxes: Tensor,
            conf_thresh:   float=0.5,
            iou_thresh:    float=0.5,
        ):

        preds_mask = torch.sigmoid(inputs[..., 0]) > conf_thresh
        preds_index = torch.nonzero(preds_mask, as_tuple=True)

        objects = inputs[preds_index]

        pred_scores = torch.sigmoid(objects[:, 0])
        pred_cxcywh = objects[:, 1:1 + self.bbox_dim]
        pred_bboxes = torch.clamp_min(self.pred2boxes(pred_cxcywh, preds_index[2], preds_index[3]), 0.)
        pred_labels = torch.argmax(objects[:, 1 + self.bbox_dim:], dim=-1)

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

    def update_metric(
            self,
            inputs:        Tuple[Tensor, Tensor],
            target_index:  List[Tensor],
            target_labels: Tensor,
            target_bboxes: Tensor,
            target_masks:  Tensor,
            conf_thresh:   float=0.5,
            recall_thresh: float=0.5,
            iou_thresh:    float=0.5,
        ):

        inputs_seg, inputs_det = inputs

        self._update_seg_metric(
            inputs_seg,
            target_masks,
            conf_thresh=conf_thresh,
        )

        self._update_det_metric(
            inputs_det,
            target_index,
            target_labels,
            target_bboxes,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
        )
