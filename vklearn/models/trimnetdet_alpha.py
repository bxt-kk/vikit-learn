from typing import List, Any, Dict, Tuple, Mapping

from torch import Tensor
from torchvision.ops import (
    # generalized_box_iou_loss,
    # complete_box_iou_loss,
    distance_box_iou_loss,
    boxes as box_ops,
    roi_align,
)
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from .detector_alpha import Detector
from .trimnetx import TrimNetX
from .component import DetPredictor
from ..utils.focal_boost import focal_boost_loss, focal_boost_positive


class TrimNetDet(Detector):
    '''A light-weight and easy-to-train model for object detection

    Args:
        categories: Target categories.
        bbox_limit: Maximum size limit of bounding box.
        anchors: Preset anchor boxes.
        num_scans: Number of the Trim-Units.
        scan_range: Range factor of the Trim-Unit convolution.
        backbone: Specify a basic model as a feature extraction module.
        backbone_pretrained: Whether to load backbone pretrained weights.
    '''

    def __init__(
            self,
            categories:          List[str],
            bbox_limit:          int=640,
            anchors:             List[Tuple[float, float]] | Tensor | None=None,
            dropout_p:           float=0.2,
            num_scans:           int=3,
            scan_range:          int=4,
            backbone:            str='mobilenet_v3_small',
            backbone_pretrained: bool=True,
        ):

        super().__init__(
            categories, bbox_limit=bbox_limit, anchors=anchors)

        self.trimnetx = TrimNetX(
            num_scans, scan_range, backbone, backbone_pretrained)

        self.cell_size = self.trimnetx.cell_size
        self.dropout_p = dropout_p

        merged_dim   = self.trimnetx.merged_dim
        features_dim = self.trimnetx.features_dim

        self.predicts = nn.ModuleList([DetPredictor(
            in_planes=merged_dim,
            num_anchors=self.num_anchors,
            bbox_dim=self.bbox_dim,
            num_classes=self.num_classes,
            dropout_p=dropout_p,
        ) for _ in range(num_scans)])

        self.auxi_clf = nn.Sequential(
            nn.Dropout(dropout_p, inplace=False),
            # nn.Linear(merged_dim, self.num_classes),
            nn.Linear(features_dim, self.num_classes),
        )

        obj_dim = self.bbox_dim + self.num_classes
        self.alphas = nn.Parameter(torch.full(
            (1, self.num_anchors, 1, 1, obj_dim, num_scans),
            fill_value=0.1))

    def train_features(self, flag:bool):
        self.trimnetx.train_features(flag)

    def forward_old(self, x:Tensor) -> Tuple[Tensor, Tensor]:
        hs, m = self.trimnetx(x)
        n, _, rs, cs = hs[0].shape

        p = self.predicts[0](hs[0])
        ps = [p[..., :1]]
        times = len(hs)
        for t in range(1, times):
            y = self.predicts[t](hs[t])
            a = torch.sigmoid(ps[-1])
            p = y * a + p * (1 - a)
            ps.append(p[..., :1])
        ps.append(p[..., 1:])
        return torch.cat(ps, dim=-1), m

    def forward(self, x:Tensor) -> Tuple[Tensor, Tensor]:
        hs, m = self.trimnetx(x)
        n, _, rs, cs = hs[0].shape

        preds = [predict(h) for predict, h in zip(self.predicts, hs)]
        confs = [preds[0][..., :1]]
        # objvs = preds[0][..., 1:]
        alphas = torch.softmax(self.alphas, dim=-1)
        objvs = preds[0][..., 1:] * alphas[..., 0]
        times = len(preds)
        for t in range(1, times):
            c_t = preds[t][..., :1]
            a_1 = torch.sigmoid(confs[-1])
            confs.append(c_t * a_1 + confs[-1] * (1 - a_1))
            # a_2 = torch.sigmoid(c_t)
            # objvs = preds[t][..., 1:] * a_2 + objvs * (1 - a_2)
            objvs = objvs + preds[t][..., 1:] * alphas[..., t]
        return torch.cat(confs + [objvs], dim=-1), m

    @classmethod
    def load_from_state(cls, state:Mapping[str, Any]) -> 'TrimNetDet':
        hyps = state['hyperparameters']
        model = cls(
            categories          = hyps['categories'],
            bbox_limit          = hyps['bbox_limit'],
            anchors             = hyps['anchors'],
            dropout_p           = hyps['dropout_p'],
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
            anchors    = self.anchors,
            dropout_p  = self.dropout_p,
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

        # CLSS_WEIGHT_KEY = 'predict.clss_predict.weight'
        # CLSS_BIAS_KEY = 'predict.clss_predict.bias'
        # AUXI_WEIGHT_KEY = 'auxi_clf.1.weight'
        # AUXI_BIAS_KEY = 'auxi_clf.1.bias'
        #
        # clss_weight = state_dict.pop(CLSS_WEIGHT_KEY)
        # clss_bias = state_dict.pop(CLSS_BIAS_KEY)
        # auxi_weight = state_dict.pop(AUXI_WEIGHT_KEY)
        # auxi_bias = state_dict.pop(AUXI_BIAS_KEY)
        #
        # predict_dim = self.predict.clss_predict.bias.shape[0]
        # if clss_bias.shape[0] == predict_dim:
        #     state_dict[CLSS_WEIGHT_KEY] = clss_weight
        #     state_dict[CLSS_BIAS_KEY] = clss_bias
        #     state_dict[AUXI_WEIGHT_KEY] = auxi_weight
        #     state_dict[AUXI_BIAS_KEY] = auxi_bias
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

        predicts, _ = self.forward(x)
        conf_prob = torch.ones_like(predicts[..., 0])
        for conf_id in range(num_confs - 1):
            conf_prob[torch.sigmoid(predicts[..., conf_id]) < recall_thresh] = 0.
        conf_prob *= torch.sigmoid(predicts[..., num_confs - 1])
        pred_objs = predicts[..., num_confs:]

        index = torch.nonzero(conf_prob > recall_thresh, as_tuple=True)
        if len(index[0]) == 0: return []
        conf = conf_prob[index[0], index[1], index[2], index[3]]
        objs = pred_objs[index[0], index[1], index[2], index[3]]

        raw_w, raw_h = image.size
        boxes = self.pred2boxes(objs[:, :self.bbox_dim], index[2], index[3])
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

    def _random_offset_index(
            self,
            index: List[Tensor],
            xyxys: Tensor,
            scale: float=0.5,
        ) -> List[Tensor]:

        if not self.training: return index
        if index[0].shape[0] == 0: return index

        cr_w = (xyxys[:, 2] - xyxys[:, 0])
        cr_x = (
            (xyxys[:, 2] + xyxys[:, 0]) * 0.5 +
            # cr_w * (torch.rand_like(cr_w) - 0.5) * scale
            cr_w * scale * (torch.clamp(torch.randn_like(cr_w) * 0.25, -0.5, 0.5))
        )
        cr_x = torch.clamp(cr_x, xyxys[:, 0], xyxys[:, 2])
        col_index = (cr_x / self.cell_size).type(torch.int64)

        cr_h = (xyxys[:, 3] - xyxys[:, 1])
        cr_y = (
            (xyxys[:, 3] + xyxys[:, 1]) * 0.5 + 
            # cr_h * (torch.rand_like(cr_h) - 0.5) * scale
            cr_h * scale * (torch.clamp(torch.randn_like(cr_h) * 0.25, -0.5, 0.5))
        )
        cr_y = torch.clamp(cr_y, xyxys[:, 1], xyxys[:, 3])
        row_index = (cr_y / self.cell_size).type(torch.int64)
        return [index[0], index[1], row_index, col_index]

    def calc_loss(
            self,
            inputs:        Tuple[Tensor, Tensor],
            target_index:  List[Tensor],
            target_labels: Tensor,
            target_bboxes: Tensor,
            weights:       Dict[str, float] | None=None,
            alpha:         float=0.25,
            gamma:         float=2.,
            clss_gamma:    float=2.,
        ) -> Dict[str, Any]:

        reduction = 'mean'
        num_confs = self.trimnetx.num_scans

        inputs_ps, inputs_mx = inputs

        conf_loss, sampled_loss = focal_boost_loss(
            inputs_ps, target_index, num_confs, alpha, gamma)

        pred_conf = inputs_ps[..., 0]
        targ_conf = torch.zeros_like(pred_conf)
        targ_conf[target_index] = 1.

        # Lab code <<<
        offset_index = self._random_offset_index(target_index, target_bboxes)
        # >>>
        # objects = inputs_ps[target_index]
        objects = inputs_ps[offset_index]

        bbox_loss = torch.zeros_like(conf_loss)
        clss_loss = torch.zeros_like(conf_loss)
        if objects.shape[0] > 0:
            pred_cxcywh = objects[:, num_confs:num_confs + self.bbox_dim]
            # pred_xyxy = self.pred2boxes(pred_cxcywh, target_index[2], target_index[3])
            pred_xyxy = self.pred2boxes(pred_cxcywh, offset_index[2], offset_index[3])
            # bbox_loss = generalized_box_iou_loss(
            # bbox_loss = complete_box_iou_loss(
            bbox_loss = distance_box_iou_loss(
                pred_xyxy, target_bboxes, reduction=reduction)

            pred_clss = objects[:, num_confs + self.bbox_dim:]
            # clss_loss = F.cross_entropy(
            #     pred_clss, target_labels, reduction=reduction)
            pred_probs = torch.softmax(pred_clss.detach(), dim=-1)
            pred_alpha = 1 - pred_probs[range(len(target_labels)), target_labels]**clss_gamma

            clss_loss = (
                pred_alpha *
                F.cross_entropy(pred_clss, target_labels, reduction='none')
            ).mean()

        weights = weights or dict()

        loss = (
            weights.get('conf', 1.) * conf_loss +
            weights.get('bbox', 1.) * bbox_loss +
            weights.get('clss', 0.33) * clss_loss
        )

        losses = dict(
            loss=loss,
            conf_loss=conf_loss,
            bbox_loss=bbox_loss,
            clss_loss=clss_loss,
            sampled_loss=sampled_loss,
        )

        # auxiliary_clss
        auxi_weight = weights.get('auxi', 0)
        if auxi_weight == 0:
            return losses
        bboxes = torch.cat([
            target_index[0].unsqueeze(-1).type_as(target_bboxes),
            target_bboxes], dim=-1)
        aligned = roi_align(
            inputs_mx, bboxes, 1, spatial_scale=1 / self.cell_size)
        auxi_pred = self.auxi_clf(aligned.flatten(start_dim=1))

        auxi_probs = torch.softmax(auxi_pred.detach(), dim=-1)
        auxi_alpha = 1 - auxi_probs[range(len(target_labels)), target_labels]**clss_gamma

        auxi_loss = (
            auxi_alpha *
            F.cross_entropy(auxi_pred, target_labels, reduction='none')
        ).mean()

        losses['loss'] = loss + auxi_loss * auxi_weight
        losses['auxi_loss'] = auxi_loss

        return losses

    def calc_score(
            self,
            inputs:        Tuple[Tensor, Tensor],
            target_index:  List[Tensor],
            target_labels: Tensor,
            target_bboxes: Tensor,
            conf_thresh:   float=0.5,
            recall_thresh: float=0.5,
            eps:           float=1e-5,
        ) -> Dict[str, Any]:

        num_confs = self.trimnetx.num_scans

        inputs_ps, _ = inputs

        targ_conf = torch.zeros_like(inputs_ps[..., 0])
        targ_conf[target_index] = 1.

        pred_obj = focal_boost_positive(
            inputs_ps, num_confs, conf_thresh, recall_thresh)

        pred_obj_true = torch.masked_select(targ_conf, pred_obj).sum()
        conf_precision = pred_obj_true / torch.clamp_min(pred_obj.sum(), eps)
        conf_recall = pred_obj_true / torch.clamp_min(targ_conf.sum(), eps)
        conf_f1 = 2 * conf_precision * conf_recall / torch.clamp_min(conf_precision + conf_recall, eps)
        proposals = pred_obj.sum() / pred_obj.shape[0]

        objects = inputs_ps[target_index]

        iou_score = torch.ones_like(conf_f1)
        clss_accuracy = torch.ones_like(conf_f1)
        obj_conf_min = torch.zeros_like(conf_f1)
        if objects.shape[0] > 0:
            pred_cxcywh = objects[:, num_confs:num_confs + self.bbox_dim]
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
            inputs:        Tuple[Tensor, Tensor],
            target_index:  List[Tensor],
            target_labels: Tensor,
            target_bboxes: Tensor,
            conf_thresh:   float=0.5,
            recall_thresh: float=0.5,
            iou_thresh:    float=0.5,
        ):

        num_confs = self.trimnetx.num_scans

        inputs_ps, _ = inputs

        preds_mask = focal_boost_positive(
            inputs_ps, num_confs, conf_thresh, recall_thresh)
        preds_index = torch.nonzero(preds_mask, as_tuple=True)

        objects = inputs_ps[preds_index]

        # <<< Lab code.
        # pred_scores = torch.sigmoid(objects[:, num_confs - 1])
        # pred_cxcywh = objects[:, num_confs:num_confs + self.bbox_dim]
        # pred_bboxes = torch.clamp_min(self.pred2boxes(pred_cxcywh, preds_index), 0.)
        # pred_labels = torch.argmax(objects[:, num_confs + self.bbox_dim:], dim=-1)

        pred_confs = torch.sigmoid(objects[:, num_confs - 1])
        pred_probs = torch.max(torch.softmax(objects[:, num_confs + self.bbox_dim:], dim=-1), dim=-1).values
        pred_scores = pred_confs * pred_probs
        pred_cxcywh = objects[:, num_confs:num_confs + self.bbox_dim]
        pred_bboxes = torch.clamp_min(self.pred2boxes(pred_cxcywh, preds_index[2], preds_index[3]), 0.)
        pred_labels = torch.argmax(objects[:, num_confs + self.bbox_dim:], dim=-1)
        # >>>

        preds = []
        target = []
        batch_size = inputs_ps.shape[0]
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
