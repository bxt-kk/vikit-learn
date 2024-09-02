from typing import List, Any, Dict, Tuple, Mapping

from torch import Tensor
from torchvision.ops import (
    distance_box_iou_loss,
    boxes as box_ops,
    roi_align,
)
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from .detector import Detector
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
            embed_dim:           int=32,
            num_scans:           int | None=None,
            scan_range:          int | None=None,
            backbone:            str | None=None,
            backbone_pretrained: bool | None=None,
        ):

        super().__init__(
            categories, bbox_limit=bbox_limit, anchors=anchors)

        self.trimnetx = TrimNetX(
            num_scans, scan_range, backbone, backbone_pretrained)

        self.cell_size = self.trimnetx.cell_size
        self.dropout_p = dropout_p
        self.embed_dim = embed_dim

        merged_dim   = self.trimnetx.merged_dim
        features_dim = self.trimnetx.features_dim

        self.predictors = nn.ModuleList([DetPredictor(
            in_planes=merged_dim,
            num_anchors=self.num_anchors,
            bbox_dim=self.bbox_dim,
            clss_dim=embed_dim,
            dropout_p=dropout_p,
        ) for _ in range(self.trimnetx.num_scans)])

        self.auxi_clf = nn.Sequential(
            nn.Dropout(dropout_p, inplace=False),
            nn.Linear(features_dim, embed_dim),
        )

        self.decoder = nn.Linear(embed_dim, self.num_classes)

        obj_dim = self.bbox_dim + embed_dim
        self.alphas = nn.Parameter(torch.zeros(
            1, self.num_anchors, 1, 1, obj_dim, self.trimnetx.num_scans))

    def train_features(self, flag:bool):
        self.trimnetx.train_features(flag)

    def forward(self, x:Tensor) -> Tuple[Tensor, Tensor]:
        hs, m = self.trimnetx(x)
        n, _, rs, cs = hs[0].shape

        alphas = torch.softmax(self.alphas, dim=-1)
        preds = [predictor(h) for predictor, h in zip(self.predictors, hs)]
        confs = [preds[0][..., :1]]
        objs = preds[0][..., 1:] * alphas[..., 0]
        times = len(preds)
        for t in range(1, times):
            conf = preds[t][..., :1]
            mask = torch.sigmoid(confs[-1])
            confs.append(conf * mask + confs[-1] * (1 - mask))
            objs = objs + preds[t][..., 1:] * alphas[..., t]
        obj_bbox = objs[..., :self.bbox_dim]
        obj_clss = self.decoder(objs[..., self.bbox_dim:])
        return torch.cat(confs + [obj_bbox, obj_clss], dim=-1), m

    @classmethod
    def load_from_state(cls, state:Mapping[str, Any]) -> 'TrimNetDet':
        hyps = state['hyperparameters']
        model = cls(
            categories          = hyps['categories'],
            bbox_limit          = hyps['bbox_limit'],
            anchors             = hyps['anchors'],
            dropout_p           = hyps['dropout_p'],
            embed_dim           = hyps['embed_dim'],
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
            embed_dim  = self.embed_dim,
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

        DECODER_WEIGHT_KEY = 'decoder.weight'
        DECODER_BIAS_KEY   = 'decoder.bias'

        decoder_weight = state_dict.pop(DECODER_WEIGHT_KEY)
        decoder_bias   = state_dict.pop(DECODER_BIAS_KEY)

        if decoder_bias.shape[0] == self.num_classes:
            state_dict[DECODER_WEIGHT_KEY] = decoder_weight
            state_dict[DECODER_BIAS_KEY] = decoder_bias
        super().load_state_dict(state_dict, strict, assign)

    def detect(
            self,
            image:         Image.Image,
            conf_thresh:   float=0.5,
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

        index = torch.nonzero(conf_prob > conf_thresh, as_tuple=True)
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

    def calc_loss(
            self,
            inputs:          Tuple[Tensor, Tensor],
            target_index:    List[Tensor],
            target_labels:   Tensor,
            target_bboxes:   Tensor,
            weights:         Dict[str, float] | None=None,
            alpha:           float=0.25,
            gamma:           float=2.,
            clss_gamma:      float=2.,
            label_smoothing: float=0.1,
            label_weight:    Tensor | None=None,
        ) -> Dict[str, Any]:

        # reduction = 'mean'
        num_confs = self.trimnetx.num_scans

        inputs_ps, inputs_mx = inputs

        conf_loss, sampled_loss = focal_boost_loss(
            inputs_ps, target_index, num_confs, alpha, gamma)

        pred_conf = inputs_ps[..., 0]
        targ_conf = torch.zeros_like(pred_conf)
        targ_conf[target_index] = 1.

        offset_index = self.random_offset_index(target_index, target_bboxes)
        objects = inputs_ps[offset_index]

        if label_weight is not None:
            label_weight = label_weight.type_as(inputs_ps)

        # lab code <<<
        # instance_weight = (
        #     1 / torch.clamp_min(targ_conf.flatten(start_dim=1).sum(dim=1), 1)
        # )[target_index[0]]
        instance_weight = 1 / target_index[0].bincount().type_as(targ_conf)[target_index[0]]
        # >>>

        bbox_loss = torch.zeros_like(conf_loss)
        clss_loss = torch.zeros_like(conf_loss)
        if objects.shape[0] > 0:
            pred_cxcywh = objects[:, num_confs:num_confs + self.bbox_dim]
            pred_xyxy = self.pred2boxes(pred_cxcywh, offset_index[2], offset_index[3])
            # Lab code <<<
            # bbox_loss = distance_box_iou_loss(
            #     pred_xyxy, target_bboxes, reduction=reduction)
            bbox_loss = (instance_weight * distance_box_iou_loss(
                pred_xyxy, target_bboxes, reduction='none')).sum() / inputs_ps.shape[0]
            # >>>

            pred_clss = objects[:, num_confs + self.bbox_dim:]
            # Lab code <<<
            pred_probs = torch.softmax(pred_clss.detach(), dim=-1)
            pred_alpha = 1 - pred_probs[range(len(target_labels)), target_labels]**clss_gamma

            # clss_loss = (
            #     pred_alpha *
            #     F.cross_entropy(
            #         pred_clss,
            #         target_labels,
            #         label_smoothing=label_smoothing,
            #         weight=label_weight,
            #         reduction='none')
            # ).mean()

            clss_loss = (
                instance_weight *
                pred_alpha *
                F.cross_entropy(
                    pred_clss,
                    target_labels,
                    label_smoothing=label_smoothing,
                    weight=label_weight,
                    reduction='none')
            ).sum() / inputs_ps.shape[0]
            # >>>

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
            iws=instance_weight.sum(),
        )

        # auxiliary_clss
        auxi_weight = weights.get('auxi', 0)
        if auxi_weight == 0:
            return losses

        auxi_loss = torch.zeros_like(conf_loss)
        if objects.shape[0] > 0:
            bboxes = torch.cat([
                target_index[0].unsqueeze(-1).type_as(target_bboxes),
                target_bboxes], dim=-1)
            aligned = roi_align(
                inputs_mx, bboxes, 1, spatial_scale=1 / self.cell_size)
            auxi_pred = self.decoder(self.auxi_clf(aligned.flatten(start_dim=1)))

            # Lab code <<<
            auxi_probs = torch.softmax(auxi_pred.detach(), dim=-1)
            auxi_alpha = 1 - auxi_probs[range(len(target_labels)), target_labels]**clss_gamma

            # auxi_loss = (
            #     auxi_alpha *
            #     F.cross_entropy(
            #         auxi_pred,
            #         target_labels,
            #         label_smoothing=label_smoothing,
            #         weight=label_weight,
            #         reduction='none')
            # ).mean()
            auxi_loss = (
                instance_weight *
                auxi_alpha *
                F.cross_entropy(
                    auxi_pred,
                    target_labels,
                    label_smoothing=label_smoothing,
                    weight=label_weight,
                    reduction='none')
            ).sum() / inputs_ps.shape[0]
            # >>>

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
        conf_min = torch.zeros_like(conf_f1)
        recall_min = torch.zeros_like(conf_f1)
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
            conf_min = obj_conf[:, -1].min()
            recall_min = obj_conf[:, :max(1, num_confs - 1)].min()

        return dict(
            conf_precision=conf_precision,
            conf_recall=conf_recall,
            conf_f1=conf_f1,
            iou_score=iou_score,
            clss_accuracy=clss_accuracy,
            proposals=proposals,
            conf_min=conf_min,
            recall_min=recall_min,
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

        top_k = self.m_ap_metric.max_detection_thresholds[-1]
        preds_mask = focal_boost_positive(
            inputs_ps, num_confs, conf_thresh, recall_thresh, top_k)
        preds_index = torch.nonzero(preds_mask, as_tuple=True)

        objects = inputs_ps[preds_index]

        pred_confs = torch.sigmoid(objects[:, num_confs - 1])
        pred_probs = torch.max(torch.softmax(objects[:, num_confs + self.bbox_dim:], dim=-1), dim=-1).values
        pred_scores = pred_confs * pred_probs
        pred_cxcywh = objects[:, num_confs:num_confs + self.bbox_dim]
        pred_bboxes = torch.clamp_min(self.pred2boxes(pred_cxcywh, preds_index[2], preds_index[3]), 0.)
        pred_labels = torch.argmax(objects[:, num_confs + self.bbox_dim:], dim=-1)
        # # Lab code <<<
        # pred_bboxes = torch.clamp(
        #         self.pred2boxes(pred_cxcywh, preds_index[2], preds_index[3]),
        #         0.,
        #         self.cell_size * (inputs_ps.shape[2] + 1))
        # pred_labels = torch.argmax(objects[:, num_confs + self.bbox_dim:], dim=-1)
        # clss_map = inputs_ps[preds_index[0], preds_index[1]][..., num_confs + self.bbox_dim:].permute(0, 3, 1, 2)
        # center_regions = self.calc_center_regions(pred_bboxes)
        # batch_regions = torch.cat([
        #     torch.arange(len(center_regions)).unsqueeze(-1).type_as(center_regions),
        #     center_regions], dim=-1)
        # average_clss = roi_align(clss_map, batch_regions, 1, spatial_scale=1 / self.cell_size)
        # pred_labels_2stage = torch.argmax(average_clss.flatten(start_dim=1), dim=-1)
        # mask_2stage = (center_regions[:, 2:] - center_regions[:, :2]).max(dim=-1).values > self.cell_size
        # pred_labels[mask_2stage] = pred_labels_2stage[mask_2stage]
        # # >>>

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
