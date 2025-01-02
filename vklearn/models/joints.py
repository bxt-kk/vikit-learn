from typing import List, Any, Dict, Tuple, Sequence
from collections import defaultdict
# import time

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
import cv2 as cv
import shapely

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

    def _match_nodes(
            self,
            begin_node: Dict[str, Any],
            end_nodes:  List[Dict[str, Any]],
            heatmap:    ndarray,
        ) -> Tuple[int, float]:

        begin_bbox = np.array(begin_node['box'], dtype=np.float32)
        begin_cxcy = (begin_bbox[:2] + begin_bbox[2:]) * 0.5
        max_score = 0.5
        matched_ix = -1
        begin_size = (begin_bbox[2:] - begin_bbox[:2]).mean()
        for end_ix, end_node in enumerate(end_nodes):
            end_bbox = np.array(end_node['box'], dtype=np.float32)
            end_size = (end_bbox[2:] - end_bbox[:2]).mean()
            end_cxcy = (end_bbox[:2] + end_bbox[2:]) * 0.5
            steps = round(np.linalg.norm(begin_cxcy - end_cxcy))
            cols = np.around(np.linspace(begin_cxcy[0], min(end_cxcy[0], heatmap.shape[1] - 1), steps)).astype(int)
            rows = np.around(np.linspace(begin_cxcy[1], min(end_cxcy[1], heatmap.shape[0] - 1), steps)).astype(int)
            region = heatmap[rows, cols]
            score = 1. if region.size == 0 else region.mean()
            score = score * (begin_node['score'] + end_node['score']) * 0.5
            score = score * min(begin_size, end_size) / max(begin_size, end_size)
            if score < max_score: continue
            max_score = score
            matched_ix = end_ix
        return matched_ix, max_score

    def joints_on_group(
            self,
            nodes:        List[Dict[str, Any]],
            heatmap:      ndarray,
            score_thresh: float,
        ) -> List[Dict[str, Any]]:

        begin_nodes = [item for item in nodes if item['anchor'] == 0]
        end_nodes = [item for item in nodes if item['anchor'] == 1]
        begin_nodes = sorted(begin_nodes, key=lambda n: n['score'], reverse=True)[:2]
        end_nodes = sorted(end_nodes, key=lambda n: n['score'], reverse=True)[:len(begin_nodes)]
        matched_pairs = []

        for begin_ix, begin_node in enumerate(begin_nodes):
            end_ix, max_score = self._match_nodes(begin_node, end_nodes, heatmap)
            if end_ix < 0: continue
            matched_pairs.append((
                begin_ix,
                end_ix, max_score))

        objs = []
        for begin_ix, end_ix, score in matched_pairs:
            begin_node = begin_nodes[begin_ix]
            end_node = end_nodes[end_ix]
            begin_bbox = np.array(begin_node['box'], dtype=np.float32)
            end_bbox = np.array(end_node['box'], dtype=np.float32)
            begin_cxcy = (begin_bbox[:2] + begin_bbox[2:]) * 0.5
            end_cxcy = (end_bbox[:2] + end_bbox[2:]) * 0.5
            vector = end_cxcy - begin_cxcy
            length = np.linalg.norm(vector)
            vector /= max(1e-5, length)
            angle = np.rad2deg(np.arccos(vector[0]))
            if vector[1] < 0: angle = -angle
            begin_width = max(begin_bbox[2:] - begin_bbox[:2])
            end_width = max(end_bbox[2:] - end_bbox[:2])
            diameter = (begin_width + end_width) * 0.5
            cx, cy = (begin_cxcy + end_cxcy) * 0.5
            rect = (cx, cy), (length + diameter, diameter), angle
            begin_score = begin_node['score']
            end_score = end_node['score']
            label = begin_node['label']
            if begin_score < end_score:
                label = end_node['label']
            if score < score_thresh: continue
            objs.append(dict(rect=rect, label=label, score=score))

        return objs

    def joints(
            self,
            nodes:        List[Dict[str, Any]],
            heatmap:      ndarray,
            score_thresh: float,
        ) -> List[Dict[str, Any]]:

        thresh_map = (heatmap > 0.5).astype(np.uint8)
        contours, _ = cv.findContours(thresh_map, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        boxes = shapely.polygons([cv.boxPoints(cv.minAreaRect(pts)) for pts in contours])

        node_groups = defaultdict(list)
        for node in nodes:
            node_box = shapely.box(*node['box'])
            for gid, box in enumerate(boxes):
                if shapely.intersects(node_box, box):
                    node_groups[gid].append(node)

        objs = []
        for group in node_groups.values():
            _objs = self.joints_on_group(group, thresh_map, score_thresh)
            objs.extend(_objs)

        return objs

    def joints_ocr(
            self,
            nodes:   List[Dict[str, Any]],
            heatmap: ndarray,
            params:  Sequence[Tuple[float, int]],
        ) -> List[Dict[str, Any]]:

        segment_thresh, rect_side_limit = params[0]
        next_params = params[1:]
        # print('joints ocr param:', segment_thresh, rect_side_limit)

        # clock = time.time()
        binary_map = (heatmap > segment_thresh).astype(np.uint8)
        contours, _ = cv.findContours(binary_map, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        block_rects = [cv.minAreaRect(pts) for pts in contours]
        block_rects = [rect for rect in block_rects if min(rect[1]) > rect_side_limit]
        if not block_rects: return []
        block_boxes = shapely.polygons([cv.boxPoints(rect) for rect in block_rects])

        # clock2 = time.time()
        node_groups = defaultdict(list)
        remains = []
        for node in nodes:
            node_box = node['box']
            node_poly = shapely.box(*node_box)
            node_centroid = shapely.centroid(node_poly)
            for bid, block_poly in enumerate(block_boxes):
                if shapely.contains(block_poly, node_centroid):
                    node_groups[bid].append(node)
                    break
            else:
                remains.append(node)
        # print('nodes size:', len(node), 'remains size:', len(remains))
        for node in remains:
            node_box = node['box']
            node_poly = shapely.box(*node_box)
            for bid, block_poly in enumerate(block_boxes):
                if node_groups[bid]: continue
                if shapely.intersects(node_poly, block_poly):
                    node_groups[bid].append(node)
        # print('delta2:', time.time() - clock2, params)
        objs = []
        multilines = []
        calc_diameter = lambda n: 0.5 * (n['box'][2] + n['box'][3] - n['box'][0] - n['box'][1])
        need_mask_blocks = []
        for bid, rect in enumerate(block_rects):
            diameters = [calc_diameter(node) for node in node_groups[bid]]
            if not diameters:
                need_mask_blocks.append(bid)
                continue
            mean_diameter = sum(diameters) / max(1, len(diameters))
            min_side = min(rect[1])
            if next_params and (min_side > 1.5 * mean_diameter):
                multilines.append(bid)
                continue
            need_mask_blocks.append(bid)
            if min_side < 1.5 * mean_diameter:
                xy, (w, h), a = rect
                if w > h:
                    rect = xy, (w + mean_diameter, mean_diameter), a
                else:
                    rect = xy, (mean_diameter, h + mean_diameter), a
            objs.append(dict(rect=rect, label='text', score=0.8))

        for obj in objs:
            xy, wh, a = obj['rect']
            if min(wh) / max(wh) < 0.7: continue
            # print('update src ojb', obj['rect'])
            side = sum(wh) * 0.5
            obj['rect'] = xy, (side, side), 0
            # print('update dst ojb', obj['rect'])
        # print('debug multilines:', multilines)

        if next_params and multilines:
            mask = np.ones_like(heatmap, dtype=np.uint8)
            mask_pts = [cv.boxPoints(block_rects[bid]).astype(np.int32) for bid in need_mask_blocks]
            mask = cv.fillPoly(mask, mask_pts, color=0)
            remains = []
            for bid in multilines:
                remains.extend(node_groups[bid])
            sub_objs = self.joints_ocr(remains, heatmap * mask, next_params)
            objs.extend(sub_objs)
        # print('debug remains:', len(remains), 'full nodes:', len(nodes))
        # print('delta:', time.time() - clock)
        return objs

    def detect(
            self,
            image:        Image.Image,
            joints_type:  str='normal',
            conf_thresh:  float=0.5,
            iou_thresh:   float=0.5,
            align_size:   int=448,
            score_thresh: float=0.5,
            ocr_params:   Sequence[Tuple[float, int]]=((0.7, 7), (0.9, 5)),
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

        elif task_name == 'documentx640':
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
                v2.RandomApply([
                    v2.RandomRotation(
                        degrees=5,
                        interpolation=v2.InterpolationMode.BILINEAR,
                        expand=False,
                        fill={tv_tensors.Image: 127, tv_tensors.Mask: 0}),
                ]),
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
