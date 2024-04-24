from typing import List, Any, Dict, Tuple

from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import (
    # sigmoid_focal_loss,
    generalized_box_iou_loss,
    box_convert,
)

from torchvision.transforms import v2
from torchvision import tv_tensors


from torchmetrics.detection import MeanAveragePrecision


class BasicConvBD(nn.Sequential):

    def __init__(
            self,
            in_planes:   int,
            out_planes:  int,
            kernel_size: int=3,
            stride:      int | tuple[int, int]=1
        ):

        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, in_planes, kernel_size, stride, padding, groups=in_planes, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes, 1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


class BasicConvDB(nn.Sequential):

    def __init__(
            self,
            in_planes:   int,
            out_planes:  int,
            kernel_size: int=3,
            stride:      int | tuple[int, int]=1
        ):

        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, 1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.Hardswish(inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size, stride, padding, groups=out_planes, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.Hardswish(inplace=True))


class UpSample(nn.Sequential):

    def __init__(
            self,
            in_planes:  int,
            out_planes: int,
        ):

        super().__init__(
            nn.ConvTranspose2d(in_planes, in_planes, 3, 2, 1, output_padding=1, groups=in_planes, bias=False),
            nn.BatchNorm2d(in_planes),
            BasicConvDB(in_planes, out_planes, 3),
        )


class TRBNetX(nn.Module):

    def __init__(
            self,
            num_classes: int,
            anchors:     List[Tuple[float, float]],
            cell_size:   int,
        ):
        super().__init__()

        self.num_classes = num_classes
        self.anchors     = torch.tensor(anchors, dtype=torch.float32)
        self.cell_size   = cell_size
        self.m_ap_metric = MeanAveragePrecision(iou_type='bbox')

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2, bias=False),
            nn.ReLU(inplace=True),
            BasicConvBD(16, 32, 3),
            nn.MaxPool2d(kernel_size=2),
            BasicConvBD(32, 64, 3),
            nn.MaxPool2d(kernel_size=2),
            BasicConvBD(64, 64, 3)
        ) # c, 64, 64

        self.level1 = nn.Sequential(
            BasicConvBD(64, 128, 3, stride=2),
            BasicConvBD(128, 256, 3, stride=2),
        ) # c, 16, 16

        self.level2 = nn.Sequential(
            BasicConvBD(256, 384, 3, stride=2),
            BasicConvBD(384, 384, 5, stride=1),
        ) # c, 8, 8
            
        self.cluster_l1 = nn.Sequential(
            BasicConvDB(384, 256, 3),
            UpSample(256, 128),
            UpSample(128, 64)) # c, 64, 64

        self.cluster_l2 = nn.Sequential(
            BasicConvDB(384, 256, 3),
            UpSample(256, 128)) # c, 16, 16

        self.num_anchors = len(anchors)
        self.predict_conf = nn.Conv2d(
            64 + 64,
            self.num_anchors,
            kernel_size=1)

        self.obj_dim = 4 + num_classes
        self.predict_objs = nn.Conv2d(
            64 + 64 + self.num_anchors,
            self.num_anchors * self.obj_dim,
            kernel_size=1)

    def forward_trb2(self, x:Tensor) -> Tensor:
        f0 = self.features(x)
        l1 = self.level1(f0)
        l2 = self.level2(l1)

        l2 = self.cluster_l2(l2)
        l1 = torch.cat([l2, l1], dim=1)

        l1 = self.cluster_l1(l1)
        x = torch.cat([l1, f0], dim=1)
        return x

    def forward(self, x:Tensor) -> Tensor:
        x = self.forward_trb2(x)
        p_conf = self.predict_conf(x)
        cx = torch.cat([p_conf, x], dim=1)
        p_objs = self.predict_objs(cx)
        bs, _, ny, nx = p_objs.shape
        p_objs = p_objs.view(bs, self.num_anchors, self.obj_dim, ny, nx)
        p_objs = p_objs.permute(0, 1, 3, 4, 2).contiguous()
        return torch.cat([p_conf.unsqueeze(-1), p_objs], dim=-1)

    def test_target2outputs(
            self,
            _outputs:      Tensor,
            target_index:  List[Tensor],
            target_labels: Tensor,
            target_bboxes: Tensor,
        ) -> Tensor:

        outputs = torch.full_like(_outputs, -1000)

        objects = outputs[target_index]
        objects[:, 0] = 1000.

        targ_cxcywh = box_convert(target_bboxes, 'xyxy', 'cxcywh')
        anchors = self.anchors.type_as(targ_cxcywh)
        targ_cxcywh[:, :2] = targ_cxcywh[:, :2] % self.cell_size / self.cell_size
        targ_cxcywh[:, 2:] = torch.log(targ_cxcywh[:, 2:] / anchors[target_index[1]])
        objects[:, 1:5] = targ_cxcywh

        objects[:, 5:].scatter_(-1, target_labels.unsqueeze(dim=-1), 1000.)

        outputs[target_index] = objects
        return outputs

    def calc_loss(
            self,
            inputs:        Tensor,
            target_index:  List[Tensor],
            target_labels: Tensor,
            target_bboxes: Tensor,
            weights:       List[float] | None=None,
            alpha:         float=0.25,
            gamma:         float=2,
        ) -> Dict[str, Any]:

        reduction = 'mean'

        pred_conf = inputs[..., 0]
        targ_conf = torch.zeros_like(pred_conf)
        targ_conf[target_index] = 1.
        # conf_loss = sigmoid_focal_loss(
        #     pred_conf, targ_conf, alpha=alpha, gamma=gamma, reduction=reduction)
        back_loss = F.binary_cross_entropy_with_logits(
            pred_conf, targ_conf, reduction=reduction)

        objects = inputs[target_index]

        bbox_loss = torch.zeros_like(back_loss)
        clss_loss = torch.zeros_like(back_loss)
        objs_loss = 0.
        if objects.shape[0] > 0:
            objs_loss = F.binary_cross_entropy_with_logits(
                objects[:, 0], targ_conf[target_index], reduction=reduction)
            pred_cxcywh = objects[:, 1:5]
            anchors = self.anchors.type_as(pred_cxcywh)
            pred_cxcywh[:, 0] = (pred_cxcywh[:, 0] + target_index[3].type_as(pred_cxcywh)) * self.cell_size
            pred_cxcywh[:, 1] = (pred_cxcywh[:, 1] + target_index[2].type_as(pred_cxcywh)) * self.cell_size
            pred_cxcywh[:, 2:] = torch.exp(pred_cxcywh[:, 2:]) * anchors[target_index[1]]
            pred_xyxy = box_convert(pred_cxcywh, 'cxcywh', 'xyxy')
            bbox_loss = generalized_box_iou_loss(pred_xyxy, target_bboxes, reduction=reduction)

            pred_clss = objects[:, 5:]
            clss_loss = F.cross_entropy(pred_clss, target_labels, reduction=reduction)

        conf_loss = alpha * objs_loss + (1 - alpha) * back_loss

        if weights is None:
            weights = [1] * 3

        sum_weights = sum(weights)
        weights = [w / sum_weights for w in weights]

        loss = (
            weights[0] * conf_loss +
            weights[1] * bbox_loss +
            weights[2] * clss_loss
        )

        return dict(
            loss=loss,
            conf_loss=conf_loss,
            bbox_loss=bbox_loss,
            clss_loss=clss_loss,
        )

    def calc_score(
            self,
            inputs:        Tensor,
            target_index:  List[Tensor],
            target_labels: Tensor,
            target_bboxes: Tensor,
            eps:           float=1e-5,
        ) -> Dict[str, Any]:

        pred_conf = inputs[..., 0]
        targ_conf = torch.zeros_like(pred_conf)
        targ_conf[target_index] = 1.

        pred_obj = pred_conf > 0.
        pred_obj_true = torch.masked_select(targ_conf, pred_obj).sum()
        conf_precision = pred_obj_true / torch.clamp_min(pred_obj.sum(), eps)
        conf_recall = pred_obj_true / targ_conf.sum()
        conf_f1 = 2 * conf_precision * conf_recall / torch.clamp_min(conf_precision + conf_recall, eps)

        objects = inputs[target_index]

        pred_cxcywh = objects[:, 1:5]
        anchors = self.anchors.type_as(pred_cxcywh)
        pred_cxcywh[:, 0] = (pred_cxcywh[:, 0] + target_index[3].type_as(pred_cxcywh)) * self.cell_size
        pred_cxcywh[:, 1] = (pred_cxcywh[:, 1] + target_index[2].type_as(pred_cxcywh)) * self.cell_size
        pred_cxcywh[:, 2:] = torch.exp(pred_cxcywh[:, 2:]) * anchors[target_index[1]]
        pred_xyxy = box_convert(pred_cxcywh, 'cxcywh', 'xyxy')
        targ_xyxy = target_bboxes

        max_x1y1 = torch.maximum(pred_xyxy[:, :2], targ_xyxy[:, :2])
        min_x2y2 = torch.minimum(pred_xyxy[:, 2:], targ_xyxy[:, 2:])
        inter_size = min_x2y2 - max_x1y1
        intersection = inter_size[:, 0] * inter_size[:, 1]
        pred_size = pred_xyxy[:, 2:] - pred_xyxy[:, :2]
        targ_size = targ_xyxy[:, 2:] - targ_xyxy[:, :2]
        pred_area = pred_size[:, 0] * pred_size[:, 1]
        targ_area = targ_size[:, 0] * targ_size[:, 1]
        union = pred_area + targ_area - intersection
        iou_score = (intersection / union).mean()

        pred_labels = torch.argmax(objects[:, 5:], dim=-1)
        clss_accuracy = (pred_labels == target_labels).sum() / len(pred_labels)

        return dict(
            conf_precision=conf_precision,
            conf_recall=conf_recall,
            conf_f1=conf_f1,
            iou_score=iou_score,
            clss_accuracy=clss_accuracy,
        )

    def calc_mean_ap(
            self,
            inputs:        Tensor,
            target_labels: Tensor,
            target_bboxes: Tensor,
        ) -> Dict[str, Any]:

        target_index = torch.nonzero(inputs[..., 0] > 0, as_tuple=True)

        objects = inputs[target_index]

        pred_scores = torch.sigmoid(objects[:, 0])

        pred_cxcywh = objects[:, 1:5]
        anchors = self.anchors.type_as(pred_cxcywh)
        pred_cxcywh[:, 0] = (pred_cxcywh[:, 0] + target_index[3].type_as(pred_cxcywh)) * self.cell_size
        pred_cxcywh[:, 1] = (pred_cxcywh[:, 1] + target_index[2].type_as(pred_cxcywh)) * self.cell_size
        pred_cxcywh[:, 2:] = torch.exp(pred_cxcywh[:, 2:]) * anchors[target_index[1]]
        pred_bboxes = box_convert(pred_cxcywh, 'cxcywh', 'xyxy')

        pred_labels = torch.argmax(objects[:, 5:], dim=-1)

        preds = [dict(
            scores=pred_scores,
            boxes=pred_bboxes,
            labels=pred_labels,
        )]

        target = [dict(
            boxes=target_bboxes,
            labels=target_labels,
        )]

        self.m_ap_metric.update(preds, target)
        return {k: v
            for k, v in self.m_ap_metric.compute().items()
            if k != 'classes'}

    def _select_anchor(self, boxes:Tensor) -> Tensor:
        sizes = boxes[:, 2:] - boxes[:, :2]
        inter_size = torch.minimum(sizes[:, None, ...], self.anchors)
        inter_area = inter_size[..., 0] * inter_size[..., 1]
        boxes_area = sizes[..., 0] * sizes[..., 1]
        union_area = (
            boxes_area[:, None] +
            self.anchors[..., 0] * self.anchors[..., 1] -
            inter_area)
        ious = inter_area / union_area
        anchor_ids = torch.argmax(ious, dim=1)
        return anchor_ids

    def _select_row(self, boxes:Tensor) -> Tensor:
        cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
        cell_row = (cy / self.cell_size).type(torch.int64)
        return cell_row

    def _select_column(self, boxes:Tensor) -> Tensor:
        cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
        cell_col = (cx / self.cell_size).type(torch.int64)
        return cell_col

    def collate_fn(
            self,
            batch: List[Any],
        ) -> Any:

        batch_ids   = []
        anchor_ids  = []
        row_ids     = []
        column_ids  = []
        list_image  = []
        list_labels = []
        list_bboxes = []

        for i, (image, target) in enumerate(batch):
            labels = target['labels']
            boxes = target['boxes']
            list_image.append(image.unsqueeze(dim=0))
            list_labels.append(labels)
            list_bboxes.append(boxes)

            batch_ids.append(torch.full_like(labels, i))
            anchor_ids.append(self._select_anchor(boxes))
            row_ids.append(self._select_row(boxes))
            column_ids.append(self._select_column(boxes))

        inputs = torch.cat(list_image, dim=0)
        target_labels = torch.cat(list_labels, dim=0)
        target_bboxes = torch.cat(list_bboxes, dim=0)
        target_index = [
            torch.cat(batch_ids),
            torch.cat(anchor_ids),
            torch.cat(row_ids),
            torch.cat(column_ids),
        ]
        return inputs, target_index, target_labels, target_bboxes

    @classmethod
    def get_transfomrs(
            cls,
            task_name:str,
        ) -> Tuple[v2.Transform, v2.Transform]:

        train_transforms = None
        test_transforms  = None

        if task_name == 'coco2017det':
            train_transforms = v2.Compose([
                v2.ToImage(),
                # v2.RandomIoUCrop(min_scale=0.3),
                v2.ScaleJitter(
                    target_size=(640, 640),
                    scale_range=(0.625, 1.25),
                    antialias=True),
                v2.RandomPhotometricDistort(p=1),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomCrop(
                    size=(448, 448),
                    pad_if_needed=True,
                    fill={tv_tensors.Image: 127, tv_tensors.Mask: 0}),
                v2.SanitizeBoundingBoxes(min_size=10),
                v2.ToDtype(torch.float32, scale=True),
            ])
            test_transforms = v2.Compose([
                v2.ToImage(),
                v2.Resize(
                    size=640,
                    max_size=640,
                    antialias=True),
                v2.RandomCrop(
                    size=(640, 640),
                    pad_if_needed=True,
                    fill={tv_tensors.Image: 127, tv_tensors.Mask: 0}),
                v2.SanitizeBoundingBoxes(min_size=10),
                v2.ToDtype(torch.float32, scale=True),
            ])
        else:
            raise ValueError(f'Unsupported the task `{task_name}`')

        return train_transforms, test_transforms


# Debug code.
if __name__ == "__main__":
    import time
    model = TRBNetX(
        num_classes=80,
        anchors=[(n, n) for n in [12, 24, 48, 96, 192, 384]],
        cell_size=32).eval()
    test_inputs = torch.randn(1, 3, 640, 448)

    with torch.no_grad():
        outputs = model.forward(test_inputs)

    clock = time.time()
    with torch.no_grad():
        outputs = model.forward(test_inputs)
    print(time.time() - clock)
    print(outputs.shape)
