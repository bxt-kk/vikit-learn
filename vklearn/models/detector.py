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
            categories: List[str],
            bbox_limit: int,
            anchors:    List[Tuple[float, float]] | Tensor | None=None,
        ):
        super().__init__()

        self.categories   = list(categories)
        self.num_classes  = len(categories)
        self.bbox_limit   = bbox_limit
        self.region_scale = bbox_limit / 32

        if anchors is None:
            anchors = []
            for k in range(3):
                anchor_base = self.region_scale * 3**k
                anchors.append((anchor_base, anchor_base))
                for aspect_ratio in (2., 3.):
                    ratio_f = aspect_ratio**0.5
                    anchors.append((anchor_base / ratio_f, anchor_base * ratio_f))
                    anchors.append((anchor_base * ratio_f, anchor_base / ratio_f))
        anchors = anchors if isinstance(anchors, Tensor) else torch.tensor(
            anchors, dtype=torch.float32).reshape(3, -1, 2)

        self.register_buffer(
            'regions', torch.tensor([[2**k for k in range(5)]]))
        self.bbox_dim    = (self.regions.shape[1] + 1) * 4 - 2
        self.anchors     = anchors
        self.num_anchors = len(anchors)
        self.cell_size   = 16

        self.m_ap_metric = MeanAveragePrecision(
            iou_type='bbox',
            backend='faster_coco_eval',
            max_detection_thresholds=[1, 10, 300],
        )

    def pred2boxes(
            self,
            inputs:    Tensor,
            row_index: Tensor,
            col_index: Tensor,
            fmt:       str='xyxy',
        ) -> Tensor:

        offsets = []
        for i in range(2):
            ptr = i * 5
            offsets.append((
                torch.tanh(inputs[:, ptr]) *
                (inputs[:, ptr + 1:ptr + 5].softmax(dim=-1) * self.regions[..., :4]).sum(dim=-1)
            ) * self.region_scale)
        for i in range(2):
            ptr = i * 6 + 10
            offsets.append((
                # torch.tanh(inputs[:, ptr]) +
                torch.sigmoid(inputs[:, ptr]) *
                (inputs[:, ptr + 1:ptr + 6].softmax(dim=-1) * self.regions).sum(dim=-1)
            ) * self.region_scale)
        ox = (col_index.type_as(inputs) + 0.5) * self.cell_size
        oy = (row_index.type_as(inputs) + 0.5) * self.cell_size
        ltx = ox + offsets[0] - offsets[2]
        lty = oy + offsets[1] - offsets[3]
        rbx = ox + offsets[0] + offsets[2]
        rby = oy + offsets[1] + offsets[3]
        bboxes = torch.cat([
            ltx.unsqueeze(-1),
            lty.unsqueeze(-1),
            rbx.unsqueeze(-1),
            rby.unsqueeze(-1),
        ], dim=-1)
        return box_convert(bboxes, 'xyxy', fmt)

    def random_offset_index(
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
            cr_w * scale * torch.clamp(torch.randn_like(cr_w) * 0.25, -0.5, 0.5)
        )
        cr_x = torch.clamp(cr_x, xyxys[:, 0], xyxys[:, 2])
        col_index = (cr_x / self.cell_size).type(torch.int64)

        cr_h = (xyxys[:, 3] - xyxys[:, 1])
        cr_y = (
            (xyxys[:, 3] + xyxys[:, 1]) * 0.5 + 
            # cr_h * (torch.rand_like(cr_h) - 0.5) * scale
            cr_h * scale * torch.clamp(torch.randn_like(cr_h) * 0.25, -0.5, 0.5)
        )
        cr_y = torch.clamp(cr_y, xyxys[:, 1], xyxys[:, 3])
        row_index = (cr_y / self.cell_size).type(torch.int64)
        return [index[0], index[1], row_index, col_index]

    def calc_center_regions(
            self,
            boxes: Tensor,
            scale: float=0.33,
        ) -> Tensor:

        pw = (boxes[:, 2] - boxes[:, 0]) * 0.5 * scale
        ph = (boxes[:, 3] - boxes[:, 1]) * 0.5 * scale
        cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
        cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
        regions = torch.zeros_like(boxes)
        regions[:, 0] = cx - pw
        regions[:, 1] = cy - ph
        regions[:, 2] = cx + pw
        regions[:, 3] = cy + ph
        return regions

    def detect(
            self,
            image:         Image.Image,
            conf_thresh:   float=0.5,
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
        metrics = self.m_ap_metric.compute()
        self.m_ap_metric.reset()
        return metrics

    def _select_anchor(self, boxes:Tensor) -> Tensor:
        derived_anchors = self.anchors.reshape(-1, 2)
        sizes = boxes[:, 2:] - boxes[:, :2]
        inter_size = torch.minimum(sizes[:, None, ...], derived_anchors)
        inter_area = inter_size[..., 0] * inter_size[..., 1]
        boxes_area = sizes[..., 0] * sizes[..., 1]
        union_area = (
            boxes_area[:, None] +
            derived_anchors[..., 0] * derived_anchors[..., 1] -
            inter_area)
        ious = inter_area / union_area
        derive_rate = len(derived_anchors) // self.num_anchors
        anchor_ids = torch.argmax(ious, dim=1) // derive_rate
        return anchor_ids

    def _select_row(self, boxes:Tensor, height:int) -> Tensor:
        cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
        # cell_size = self.cell_size
        # noise = torch.rand_like(cy) * cell_size - cell_size / 2
        # hs = boxes[:, 3] - boxes[:, 1]
        # noise[hs < 2 * cell_size] = 0.
        # cy = cy + noise 
        cell_row = (cy / self.cell_size).type(torch.int64)
        min_row = 0
        max_row = height // self.cell_size - 1
        cell_row = torch.clamp(cell_row, min_row, max_row)
        return cell_row

    def _select_column(self, boxes:Tensor, width:int) -> Tensor:
        cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
        # cell_size = self.cell_size
        # noise = torch.rand_like(cx) * cell_size - cell_size / 2
        # ws = boxes[:, 2] - boxes[:, 0]
        # noise[ws < 2 * cell_size] = 0.
        # cx = cx + noise
        cell_col = (cx / self.cell_size).type(torch.int64)
        min_col = 0
        max_col = width // self.cell_size - 1
        cell_col = torch.clamp(cell_col, min_col, max_col)
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
            row_ids.append(self._select_row(boxes, image.shape[1]))
            column_ids.append(self._select_column(boxes, image.shape[2]))

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
                    scale_range=(384 / 448, 1.1),
                    antialias=True),
                v2.RandomPhotometricDistort(p=1),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomChoice([
                    v2.GaussianBlur(7, sigma=(0.1, 2.0)),
                    v2.RandomAdjustSharpness(2, p=0.5),
                    v2.RandomEqualize(p=0.5),
                ]),
                v2.RandomCrop(
                    size=(448, 448),
                    pad_if_needed=True,
                    fill={tv_tensors.Image: 127, tv_tensors.Mask: 0}),
                v2.SanitizeBoundingBoxes(min_size=3),
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
                v2.Pad(
                    padding=448 // 4,
                    fill={tv_tensors.Image: 127, tv_tensors.Mask: 0}),
                v2.CenterCrop(448),
                v2.SanitizeBoundingBoxes(min_size=3),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])

        elif task_name == 'cocox384':
            train_transforms = v2.Compose([
                v2.ToImage(),
                v2.ScaleJitter(
                    target_size=(384, 384),
                    scale_range=(0.9, 1.1),
                    antialias=True),
                v2.RandomPhotometricDistort(p=1),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomChoice([
                    v2.GaussianBlur(7, sigma=(0.1, 2.0)),
                    v2.RandomAdjustSharpness(2, p=0.5),
                    v2.RandomEqualize(p=0.5),
                ]),
                v2.RandomCrop(
                    size=(384, 384),
                    pad_if_needed=True,
                    fill={tv_tensors.Image: 127, tv_tensors.Mask: 0}),
                v2.SanitizeBoundingBoxes(min_size=3),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])
            test_transforms = v2.Compose([
                v2.ToImage(),
                v2.Resize(
                    size=383,
                    max_size=384,
                    antialias=True),
                v2.Pad(
                    padding=384 // 4,
                    fill={tv_tensors.Image: 127, tv_tensors.Mask: 0}),
                v2.CenterCrop(384),
                v2.SanitizeBoundingBoxes(min_size=3),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])

        elif task_name == 'cocox512':
            train_transforms = v2.Compose([
                v2.ToImage(),
                v2.ScaleJitter(
                    target_size=(512, 512),
                    scale_range=(384 / 512, 1.1),
                    antialias=True),
                v2.RandomPhotometricDistort(p=1),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomChoice([
                    v2.GaussianBlur(7, sigma=(0.1, 2.0)),
                    v2.RandomAdjustSharpness(2, p=0.5),
                    v2.RandomEqualize(p=0.5),
                ]),
                v2.RandomCrop(
                    size=(512, 512),
                    pad_if_needed=True,
                    fill={tv_tensors.Image: 127, tv_tensors.Mask: 0}),
                v2.SanitizeBoundingBoxes(min_size=3),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])
            test_transforms = v2.Compose([
                v2.ToImage(),
                v2.Resize(
                    size=511,
                    max_size=512,
                    antialias=True),
                v2.Pad(
                    padding=512 // 4,
                    fill={tv_tensors.Image: 127, tv_tensors.Mask: 0}),
                v2.CenterCrop(512),
                v2.SanitizeBoundingBoxes(min_size=3),
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
                    scale_range=(384 / 640, 1.1),
                    antialias=True),
                v2.RandomPhotometricDistort(p=1),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomChoice([
                    v2.GaussianBlur(7, sigma=(0.1, 2.0)),
                    v2.RandomAdjustSharpness(2, p=0.5),
                    v2.RandomEqualize(p=0.5),
                ]),
                v2.RandomCrop(
                    size=(640, 640),
                    pad_if_needed=True,
                    fill={tv_tensors.Image: 127, tv_tensors.Mask: 0}),
                v2.SanitizeBoundingBoxes(min_size=3),
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
                v2.Pad(
                    padding=640 // 4,
                    fill={tv_tensors.Image: 127, tv_tensors.Mask: 0}),
                v2.CenterCrop(640),
                v2.SanitizeBoundingBoxes(min_size=3),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])

        elif task_name == 'documentx640':
            train_transforms = v2.Compose([
                v2.ToImage(),
                v2.ScaleJitter(
                    target_size=(640, 640),
                    scale_range=(512 / 640, 1.1),
                    antialias=True),
                v2.RandomPhotometricDistort(p=1),
                v2.RandomChoice([
                    v2.GaussianBlur(7, sigma=(0.1, 2.0)),
                    v2.RandomAdjustSharpness(2, p=0.5),
                    v2.RandomEqualize(p=0.5),
                ]),
                v2.RandomCrop(
                    size=(640, 640),
                    pad_if_needed=True,
                    fill={tv_tensors.Image: 127, tv_tensors.Mask: 0}),
                v2.RandomChoice([v2.Identity()] + [
                    v2.RandomRotation(
                        degrees=(degree - 0.1, degree + 0.1),
                        interpolation=v2.InterpolationMode.BILINEAR,
                        fill={tv_tensors.Image: 127, tv_tensors.Mask: 0})
                    for degree in (90, 180, 270)
                ]),
                v2.SanitizeBoundingBoxes(min_size=3),
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
                v2.Pad(
                    padding=640 // 4,
                    fill={tv_tensors.Image: 127, tv_tensors.Mask: 0}),
                v2.CenterCrop(640),
                v2.SanitizeBoundingBoxes(min_size=3),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])

        else:
            raise ValueError(f'Unsupported the task `{task_name}`')

        return train_transforms, test_transforms
