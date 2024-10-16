from typing import List, Any, Dict, Tuple

from torch import Tensor

import torch

from torchvision import tv_tensors
from torchvision.ops import box_convert
from torchvision.transforms import v2

from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.segmentation import MeanIoU

from PIL import Image
from numpy import ndarray
import numpy as np

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
            'regions', torch.tensor([[2**k for k in range(5)]]))
        self.bbox_dim    = (self.regions.shape[1] + 1) * 3 - 2
        self.num_anchors = 2
        self.cell_size   = 16

        self.m_ap_metric = MeanAveragePrecision(
            iou_type='bbox',
            backend='faster_coco_eval',
            max_detection_thresholds=[1, 10, 300],
        )
        self.m_iou_metric = MeanIoU(
            num_classes=self.num_classes)

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
        ptr = 10
        padding = (
            # torch.tanh(inputs[:, ptr]) +
            torch.sigmoid(inputs[:, ptr]) *
            (inputs[:, ptr + 1:ptr + 6].softmax(dim=-1) * self.regions).sum(dim=-1)
        ) * self.region_scale

        ox = (col_index.type_as(inputs) + 0.5) * self.cell_size
        oy = (row_index.type_as(inputs) + 0.5) * self.cell_size
        ltx = ox + offsets[0] - padding
        lty = oy + offsets[1] - padding
        rbx = ox + offsets[0] + padding
        rby = oy + offsets[1] + padding
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
            rows:  int,
            cols:  int,
            scale: float=0.5,
        ) -> List[Tensor]:

        if not self.training: return index
        if index[0].shape[0] == 0: return index

        cr_w = (xyxys[:, 2] - xyxys[:, 0])
        cr_x = (
            (xyxys[:, 2] + xyxys[:, 0]) * 0.5 +
            cr_w * scale * torch.clamp(torch.randn_like(cr_w) * 0.25, -0.5, 0.5)
        )
        cr_x = torch.clamp(cr_x, xyxys[:, 0], xyxys[:, 2])
        col_index = torch.clamp(cr_x / self.cell_size, 0, cols - 1).type(torch.int64)

        cr_h = (xyxys[:, 3] - xyxys[:, 1])
        cr_y = (
            (xyxys[:, 3] + xyxys[:, 1]) * 0.5 + 
            cr_h * scale * torch.clamp(torch.randn_like(cr_h) * 0.25, -0.5, 0.5)
        )
        cr_y = torch.clamp(cr_y, xyxys[:, 1], xyxys[:, 3])
        row_index = torch.clamp(cr_y / self.cell_size, 0, rows - 1).type(torch.int64)
        return [index[0], index[1], row_index, col_index]

    def _joint_iter(
            self,
            begin_node: Dict[str, Any],
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
            steps = round(np.linalg.norm(begin_cxcy - end_cxcy))
            cols = np.around(np.linspace(begin_cxcy[0], min(end_cxcy[0], heatmap.shape[1] - 1), steps)).astype(int)
            rows = np.around(np.linspace(begin_cxcy[1], min(end_cxcy[1], heatmap.shape[0] - 1), steps)).astype(int)
            region = heatmap[rows, cols]
            if region.size == 0: continue
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
            inputs:        Tuple[Tensor, Tensor],
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
            inputs:        Tuple[Tensor, Tensor],
            target_index:  List[Tensor],
            target_labels: Tensor,
            target_bboxes: Tensor,
            conf_thresh:   float=0.5,
            eps:           float=1e-5,
        ) -> Dict[str, Any]:
        assert not 'this is an empty func'

    def update_metric(
            self,
            inputs:        Tuple[Tensor, Tensor],
            target_index:  List[Tensor],
            target_labels: Tensor,
            target_bboxes: Tensor,
            conf_thresh:   float=0.5,
            iou_thresh:    float=0.5,
        ):
        assert not 'this is an empty func'

    def compute_metric(self) -> Dict[str, Any]:
        miou = self.m_iou_metric.compute() # / self.m_iou_metric.update_count
        metrics = self.m_ap_metric.compute()
        metrics['miou'] = miou
        self.m_iou_metric.reset()
        self.m_ap_metric.reset()
        metrics['mjoin'] = (
            2 * metrics['map'] * miou /
            torch.clamp_min(metrics['map'] + miou, 1e-5)
        )
        return metrics

    def _select_anchor(self, labels:Tensor) -> Tensor:
        anchor_ids = torch.zeros_like(labels)
        anchor_ids[::2] = 0
        anchor_ids[1::2] = 1
        return anchor_ids

    def _select_row(self, boxes:Tensor, height:int) -> Tensor:
        cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
        cell_row = (cy / self.cell_size).type(torch.int64)
        min_row = 0
        max_row = height // self.cell_size - 1
        cell_row = torch.clamp(cell_row, min_row, max_row)
        return cell_row

    def _select_column(self, boxes:Tensor, width:int) -> Tensor:
        cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
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
        list_masks  = []

        for i, (image, target) in enumerate(batch):
            labels = target['labels']
            boxes = target['boxes']
            masks = target['masks']
            list_image.append(image.unsqueeze(dim=0))
            list_labels.append(labels)
            list_bboxes.append(boxes)
            list_masks.append(masks.unsqueeze(dim=0))

            batch_ids.append(torch.full_like(labels, i))
            anchor_ids.append(self._select_anchor(labels))
            row_ids.append(self._select_row(boxes, image.shape[1]))
            column_ids.append(self._select_column(boxes, image.shape[2]))

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
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomChoice([
                    v2.GaussianBlur(7, sigma=(0.1, 2.0)),
                    v2.RandomAdjustSharpness(2, p=0.5),
                    v2.RandomEqualize(p=0.5),
                ]),
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
                v2.Pad(
                    padding=448 // 4,
                    fill={tv_tensors.Image: 127, tv_tensors.Mask: 0}),
                v2.CenterCrop(448),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])

        elif task_name == 'mvtecx512':
            train_transforms = v2.Compose([
                v2.ToImage(),
                v2.Resize(
                    size=511,
                    max_size=512,
                    antialias=True),
                v2.RandomPhotometricDistort(p=1),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomChoice([
                    v2.GaussianBlur(7, sigma=(0.1, 2.0)),
                    v2.RandomAdjustSharpness(2, p=0.5),
                    v2.RandomEqualize(p=0.5),
                ]),
                v2.RandomCrop(
                    size=(512, 512),
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
                    size=511,
                    max_size=512,
                    antialias=True),
                v2.Pad(
                    padding=512 // 4,
                    fill={tv_tensors.Image: 127, tv_tensors.Mask: 0}),
                v2.CenterCrop(512),
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
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomChoice([
                    v2.GaussianBlur(7, sigma=(0.1, 2.0)),
                    v2.RandomAdjustSharpness(2, p=0.5),
                    v2.RandomEqualize(p=0.5),
                ]),
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
                v2.Pad(
                    padding=640 // 4,
                    fill={tv_tensors.Image: 127, tv_tensors.Mask: 0}),
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
