from typing import List, Any, Dict, Tuple

from torch import Tensor

import torch

from torchvision import tv_tensors
from torchvision.ops import box_convert
from torchvision.transforms import v2

from torchmetrics.detection import MeanAveragePrecision

from PIL import Image

from .basic import Basic


class Detector(Basic):

    def __init__(
            self,
            num_classes: int,
            anchors:     List[Tuple[float, float]] | Tensor,
        ):
        super().__init__()

        load_anchors = lambda anchors: (
            anchors if isinstance(anchors, Tensor)
            else torch.tensor(anchors, dtype=torch.float32))

        self.num_classes    = num_classes
        self.anchors        = load_anchors(anchors)

        self.num_anchors = len(anchors)
        self.cell_size   = 16
        self.m_ap_metric = MeanAveragePrecision(
            iou_type='bbox', backend='faster_coco_eval')

    def pred2boxes(
            self,
            cxcywh: Tensor,
            index:  List[Tensor],
            fmt:    str='xyxy',
        ) -> Tensor:

        anchors = self.anchors.type_as(cxcywh)
        boxes_x  = (torch.tanh(cxcywh[:, 0]) + 0.5 + index[3].type_as(cxcywh)) * self.cell_size
        boxes_y  = (torch.tanh(cxcywh[:, 1]) + 0.5 + index[2].type_as(cxcywh)) * self.cell_size
        boxes_s  = torch.exp(cxcywh[:, 2:]) * anchors[index[1]]
        bboxes = torch.cat([boxes_x.unsqueeze(-1), boxes_y.unsqueeze(-1), boxes_s], dim=-1)
        return box_convert(bboxes, 'cxcywh', fmt)

    def detect(
            self,
            image:       Image.Image,
            conf_thresh: float=0.6,
            iou_thresh:  float=0.55,
            align_size:  int=448,
        ) -> List[Dict[str, Any]]:
        assert not 'this is an empty func'

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
        assert not 'this is an empty func'

    def calc_score(
            self,
            inputs:        Tensor,
            target_index:  List[Tensor],
            target_labels: Tensor,
            target_bboxes: Tensor,
            conf_thresh:   float=0.5,
            eps:           float=1e-5,
        ) -> Dict[str, Any]:
        assert not 'this is an empty func'

    def update_metric(
            self,
            inputs:        Tensor,
            target_index:  List[Tensor],
            target_labels: Tensor,
            target_bboxes: Tensor,
            conf_thresh:   float=0.5,
            iou_thresh:    float=0.5,
        ):
        assert not 'this is an empty func'

    def compute_metric(self) -> Dict[str, Any]:
        return self.m_ap_metric.compute()

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
        cell_size = self.cell_size
        noise = torch.rand_like(cy) * cell_size - cell_size / 2
        hs = boxes[:, 3] - boxes[:, 1]
        noise[hs < 2 * cell_size] = 0.
        cy = cy + noise 
        cell_row = (cy / self.cell_size).type(torch.int64)
        return cell_row

    def _select_column(self, boxes:Tensor) -> Tensor:
        cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
        cell_size = self.cell_size
        noise = torch.rand_like(cx) * cell_size - cell_size / 2
        ws = boxes[:, 2] - boxes[:, 0]
        noise[ws < 2 * cell_size] = 0.
        cx = cx + noise
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
    def get_transforms(
            cls,
            task_name: str='default',
        ) -> Tuple[v2.Transform, v2.Transform]:

        train_transforms = None
        test_transforms  = None

        if task_name in ('default', 'cocox448'):
            train_transforms = v2.Compose([
                v2.ToImage(),
                v2.ScaleJitter(
                    target_size=(448, 448),
                    scale_range=(0.9, 1.1),
                    antialias=True),
                v2.RandomPhotometricDistort(p=1),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomCrop(
                    size=(448, 448),
                    pad_if_needed=True,
                    fill={tv_tensors.Image: 127, tv_tensors.Mask: 0}),
                v2.SanitizeBoundingBoxes(min_size=5),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])
            test_transforms = v2.Compose([
                v2.ToImage(),
                v2.Resize(
                    size=447,
                    max_size=448,
                    antialias=True),
                v2.CenterCrop(448),
                v2.SanitizeBoundingBoxes(min_size=5),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])

        elif task_name == 'cocox640':
            train_transforms = v2.Compose([
                v2.ToImage(),
                v2.ScaleJitter(
                    target_size=(640, 640),
                    scale_range=(0.8, 1.25),
                    antialias=True),
                v2.RandomPhotometricDistort(p=1),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomCrop(
                    size=(640, 640),
                    pad_if_needed=True,
                    fill={tv_tensors.Image: 127, tv_tensors.Mask: 0}),
                v2.SanitizeBoundingBoxes(min_size=10),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])
            test_transforms = v2.Compose([
                v2.ToImage(),
                v2.Resize(
                    size=639,
                    max_size=640,
                    antialias=True),
                v2.CenterCrop(640),
                v2.SanitizeBoundingBoxes(min_size=10),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])

        else:
            raise ValueError(f'Unsupported the task `{task_name}`')

        return train_transforms, test_transforms
