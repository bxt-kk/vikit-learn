from typing import List, Any, Dict, Mapping, Tuple
import math

from torch import Tensor
from torchvision.ops import (
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
from ..utils.focal_boost import focal_boost_loss, focal_boost_positive


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
        expanded_dim = merged_dim * 4

        self.seg_predict = SegPredictor(
            merged_dim, num_classes=1, num_layers=4)

        self.det_predict = DetPredictor(
            in_planes=merged_dim + 1,
            hidden_planes=expanded_dim,
            num_anchors=self.num_anchors,
            bbox_dim=self.bbox_dim,
            num_classes=self.num_classes,
        )

    def train_features(self, flag:bool):
        self.trimnetx.train_features(flag)

    def forward_seg(self, hs:List[Tensor]) -> Tensor:
        p = self.seg_predict(hs[0])
        ps = [p]
        times = len(hs)
        for t in range(1, times):
            a = torch.sigmoid(p)
            p = self.seg_predict(hs[t]) * a + p * (1 - a)
            ps.append(p)
        return torch.cat([p[..., None] for p in ps], dim=-1)

    def forward_det(self, hs:List[Tensor]) -> Tensor:
        n, _, rs, cs = hs[0].shape

        p = self.det_predict(hs[0])
        ps = [p[..., :1]]
        times = len(hs)
        for t in range(1, times):
            y = self.det_predict(hs[t])
            a = torch.sigmoid(ps[-1])
            p = y * a + p * (1 - a)
            ps.append(p[..., :1])
        ps.append(p[..., 1:])
        return torch.cat(ps, dim=-1)

    def forward(self, x:Tensor) -> Tuple[Tensor, Tensor]:
        hs, _ = self.trimnetx(x)
        seg_p = self.forward_seg(hs)

        mask = F.max_pool2d(seg_p[..., -1], 4)
        mask = F.interpolate(mask, scale_factor=0.25, mode='bilinear')
        mask = F.hardsigmoid(mask)
        hs = [torch.cat([h, mask], dim=1) for h in hs]

        det_p = self.forward_det(hs)
        return seg_p, det_p

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
        CLSS_WEIGHT_KEY = 'det_predict.clss_predict.2.weight'
        CLSS_BIAS_KEY = 'det_predict.clss_predict.2.bias'

        clss_weight = state_dict.pop(CLSS_WEIGHT_KEY)
        clss_bias = state_dict.pop(CLSS_BIAS_KEY)
        predict_dim = self.det_predict.clss_predict[-1].bias.shape[0]
        if clss_bias.shape[0] == predict_dim:
            state_dict[CLSS_WEIGHT_KEY] = clss_weight
            state_dict[CLSS_BIAS_KEY] = clss_bias
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

        num_confs = self.trimnetx.num_scans

        seg_predicts, det_predicts = self.forward(x)
        conf_prob = torch.ones_like(det_predicts[..., 0])
        for conf_id in range(num_confs - 1):
            conf_prob[torch.sigmoid(det_predicts[..., conf_id]) < recall_thresh] = 0.
        conf_prob *= torch.sigmoid(det_predicts[..., num_confs - 1])
        pred_objs = det_predicts[..., num_confs:]

        index = torch.nonzero(conf_prob > recall_thresh, as_tuple=True)
        if len(index[0]) == 0: return []
        conf = conf_prob[index[0], index[1], index[2], index[3]]
        objs = pred_objs[index[0], index[1], index[2], index[3]]

        raw_w, raw_h = image.size
        boxes = self.pred2boxes(objs[:, :self.bbox_dim], index)

        clss = torch.softmax(objs[:, self.bbox_dim:], dim=-1).max(dim=-1)
        labels, probs = clss.indices, clss.values
        scores = conf # * probs
        final_ids = box_ops.batched_nms(boxes, scores, labels, iou_thresh)
        boxes = boxes[final_ids]
        labels = labels[final_ids]
        probs = probs[final_ids]
        scores = scores[final_ids]
        anchors = index[1][final_ids]

        nodes = []
        for score, box, label, prob, anchor in zip(scores, boxes, labels, probs, anchors):
            if score < conf_thresh: continue
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

        rects, remains = self.joints(nodes, heatmap)

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

        for ix in range(len(rects)):
            (cx, cy), (w, h), a = rects[ix]
            cx = round(min(max((cx - pad_x) / scale, 0), raw_w))
            cy = round(min(max((cy - pad_y) / scale, 0), raw_h))
            w = w / scale
            h = h / scale
            rects[ix] = (cx, cy), (w, h), a

        return dict(
            remains=remains,
            heatmap=heatmap,
            rects=rects,
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

        rects = []
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
            rects.append(rect)

        return rects, begin_nodes + end_nodes

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
            gamma:   float=0.5,
        ) -> Dict[str, Any]:

        times = inputs.shape[-1]
        F_sigma = lambda t: 1 - (math.cos((t + 1) / times * math.pi) + 1) * 0.5
        target = target.type_as(inputs)

        alpha = (target.mean(dim=(1, 2, 3))**gamma + 1) * 0.5
        grand_sigma = 0.

        loss = 0.
        for t in range(times):

            sigma = F_sigma(t)
            grand_sigma += sigma
            bce = F.binary_cross_entropy_with_logits(
                inputs[..., t],
                target,
                reduction='none',
            ).mean(dim=(1, 2, 3))
            dice = self.dice_loss(
                inputs[..., t],
                target)
            loss_t = alpha * bce + (1 - alpha) * dice
            loss = loss + loss_t.mean() * sigma

        loss = loss / grand_sigma

        return dict(
            loss=loss,
            alpha=alpha.mean(),
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
        num_confs = self.trimnetx.num_scans

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
            seg_gamma:     float=0.5,
        ) -> Dict[str, Any]:

        inputs_seg, inputs_det = inputs

        seg_losses = self._calc_seg_loss(
            inputs_seg,
            target_masks,
            weights=weights,
            gamma=seg_gamma,
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

        predicts = torch.sigmoid(inputs[..., -1])
        distance = torch.abs(predicts - target).mean(dim=(2, 3)).mean()

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
            recall_thresh: float=0.5,
            eps:           float=1e-5,
        ) -> Dict[str, Any]:

        num_confs = self.trimnetx.num_scans

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
            recall_thresh=recall_thresh,
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

        predicts = torch.sigmoid(inputs[..., -1]) > conf_thresh
        self.m_iou.update(predicts.to(torch.int), target.to(torch.int))

    def _update_det_metric(
            self,
            inputs:        Tensor,
            target_index:  List[Tensor],
            target_labels: Tensor,
            target_bboxes: Tensor,
            conf_thresh:   float=0.5,
            recall_thresh: float=0.5,
            iou_thresh:    float=0.5,
        ):

        num_confs = self.trimnetx.num_scans

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
            recall_thresh=recall_thresh,
            iou_thresh=iou_thresh,
        )
