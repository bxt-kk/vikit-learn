from typing import List, Any, Dict, Tuple

from torch import Tensor

import torch

from torchvision import tv_tensors
from torchvision.ops import box_convert
from torchvision.transforms import v2

from torchmetrics.detection import MeanAveragePrecision

from PIL import Image

from .basic import Basic


class Joints(Basic):

    def __init__(
            self,
            categories: List[str],
            bbox_limit: int,
        ):
        super().__init__()

        self.categories   = list(categories)
        self.num_classes  = len(categories)
        self.bbox_limit   = bbox_limit
        self.region_scale = bbox_limit / 32

        self.register_buffer(
            'regions', torch.tensor([[2**k for k in range(6)]]))
        self.bbox_dim    = 2 + self.regions.shape[1] + 1
        self.num_anchors = 2
        self.cell_size   = 16

        self.m_ap_metric = MeanAveragePrecision(
            iou_type='bbox',
            backend='faster_coco_eval',
            max_detection_thresholds=[1, 10, 300],
        )

    def pred2boxes(
            self,
            cxcywh: Tensor,
            index:  List[Tensor],
            fmt:    str='xyxy',
        ) -> Tensor:

        boxes_x = (
            torch.tanh(cxcywh[:, 0]) + 0.5 +
            index[3].type_as(cxcywh)
        ) * self.cell_size
        boxes_y = (
            torch.tanh(cxcywh[:, 1]) + 0.5 +
            index[2].type_as(cxcywh)
        ) * self.cell_size
        boxes_w = (
            torch.tanh(cxcywh[:, 2 + 0]) +
            (cxcywh[:, 2 + 1:2 + 7].softmax(dim=-1) * self.regions).sum(dim=-1)
        ) * self.region_scale
        bboxes = torch.cat([
            boxes_x.unsqueeze(-1),
            boxes_y.unsqueeze(-1),
            boxes_w.unsqueeze(-1),
            boxes_w.unsqueeze(-1),
            ], dim=-1)
        return box_convert(bboxes, 'cxcywh', fmt)

    def detect(
            self,
            image:         Image.Image,
            conf_thresh:   float=0.6,
            recall_thresh: float=0.5,
            iou_thresh:    float=0.5,
            align_size:    int=448,
            mini_side:     int=1,
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

    def _select_anchor(self, labels:Tensor) -> Tensor:
        anchor_ids = torch.zeros_like(labels)
        anchor_ids[::2] = 0
        anchor_ids[1::2] = 1
        return anchor_ids

    def _select_row(self, boxes:Tensor) -> Tensor:
        cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
        # cell_size = self.cell_size
        # noise = torch.rand_like(cy) * cell_size - cell_size / 2
        # hs = boxes[:, 3] - boxes[:, 1]
        # noise[hs < 2 * cell_size] = 0.
        # cy = cy + noise 
        cell_row = (cy / self.cell_size).type(torch.int64)
        return cell_row

    def _select_column(self, boxes:Tensor) -> Tensor:
        cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
        # cell_size = self.cell_size
        # noise = torch.rand_like(cx) * cell_size - cell_size / 2
        # ws = boxes[:, 2] - boxes[:, 0]
        # noise[ws < 2 * cell_size] = 0.
        # cx = cx + noise
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
        list_masks  = []

        for i, (image, target) in enumerate(batch):
            labels = target['labels']
            boxes = target['boxes']
            masks = target['masks']
            list_image.append(image.unsqueeze(dim=0))
            list_labels.append(labels)
            list_bboxes.append(boxes)
            list_masks.append(masks)

            batch_ids.append(torch.full_like(labels, i))
            anchor_ids.append(self._select_anchor(labels))
            row_ids.append(self._select_row(boxes))
            column_ids.append(self._select_column(boxes))

        inputs = torch.cat(list_image, dim=0)
        target_labels = torch.cat(list_labels, dim=0)
        target_bboxes = torch.cat(list_bboxes, dim=0)
        target_masks = torch.cat(list_masks, dim=0)
        target_index = [
            torch.cat(batch_ids),
            torch.cat(anchor_ids),
            torch.cat(row_ids),
            torch.cat(column_ids),
        ]
        return (
            inputs,
            target_index,
            target_labels,
            target_bboxes,
            target_masks,
        )

    @classmethod
    def get_transforms(
            cls,
            task_name: str='default',
        ) -> Tuple[v2.Transform, v2.Transform]:

        train_transforms = None
        test_transforms  = None

        if task_name in ('default', 'mvtecx448'):
            train_transforms = v2.Compose([
                v2.ToImage(),
                v2.Resize(
                    size=447,
                    max_size=448,
                    antialias=True),
                v2.RandomPhotometricDistort(p=1),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomCrop(
                    size=(448, 448),
                    pad_if_needed=True,
                    fill={tv_tensors.Image: 127, tv_tensors.Mask: 0}),
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
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])

        elif task_name == 'mvtecx640':
            train_transforms = v2.Compose([
                v2.ToImage(),
                v2.Resize(
                    size=639,
                    max_size=640,
                    antialias=True),
                v2.RandomPhotometricDistort(p=1),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomCrop(
                    size=(640, 640),
                    pad_if_needed=True,
                    fill={tv_tensors.Image: 127, tv_tensors.Mask: 0}),
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
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])

        else:
            raise ValueError(f'Unsupported the task `{task_name}`')

        return train_transforms, test_transforms
