from typing import Any, Callable, List, Tuple, Dict
from glob import glob
import os.path
import json
# import math

from PIL import Image
import cv2 as cv
import numpy as np

import torch
from torchvision.datasets.vision import VisionDataset
from torchvision.tv_tensors import BoundingBoxes, Mask
from torchvision.ops import box_convert


class LabelmeJoints(VisionDataset):
    '''`Labelme Joints Detection dataset.

    Args:
        root: Root directory where images are downloaded to.
        transform: A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform: A function/transform that takes in the
            target and transforms it.
        transforms: A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    '''
    LINE_THICKNESS = 7

    def __init__(
        self,
        root:             str,
        split:            str='',
        transform:        Callable | None=None,
        target_transform: Callable | None=None,
        transforms:       Callable | None=None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        assert split in ['', 'train', 'valid', 'test']
        self.dataset_dir = os.path.abspath(os.path.join(root, split))
        assert os.path.isdir(self.dataset_dir)

        self.label_paths = sorted(glob(os.path.join(self.dataset_dir, '*.json')))
        self.classes = []
        classnames_file = os.path.join(root, 'classnames.txt')
        if os.path.isfile(classnames_file):
            with open(classnames_file) as f:
                for name in f:
                    name = name.strip()
                    if not name: continue
                    self.classes.append(name)

    def __len__(self) -> int:
        return len(self.label_paths)

    def _load_image(self, path:str) -> Image.Image:
        return Image.open(os.path.join(self.dataset_dir, path)).convert("RGB")

    def _format_anns(
        self,
        anns:       List[Any],
        image_size: Tuple[int, int],
    ) -> Dict[str, Any]:

        bbox_list = []
        label_list = []
        for ann in anns:
            (x, y), (w, h), a = cv.minAreaRect(np.array(ann['points'], dtype=int))
            if w < h:
                w, h = h, w
                a -= 90
            pts = cv.boxPoints(((x, y), (w, h), a))

            diameter = min(w, h)
            if diameter < self.LINE_THICKNESS: continue
            length = max(w, h)
            x1y1 = (pts[0] + pts[1]) * 0.5
            x2y2 = (pts[2] + pts[3]) * 0.5
            x1, y1 = x1y1 + (x2y2 - x1y1) * diameter / length * 0.5
            x2, y2 = x2y2 + (x1y1 - x2y2) * diameter / length * 0.5

            bbox_list.append([x1, y1, diameter, diameter])
            bbox_list.append([x2, y2, diameter, diameter])
            label_id = 0 if not self.classes else self.classes.index(ann['label'])
            label_list.append(label_id)
            label_list.append(label_id)

        return dict(
            boxes=BoundingBoxes(
                bbox_list,
                format='CXCYWH',
                canvas_size=(image_size[1], image_size[0])),
            labels=torch.LongTensor(label_list),
        )

    def _draw_masks(self, boxes:BoundingBoxes) -> Mask:
        ground = np.zeros(boxes.canvas_size, dtype=np.uint8)
        for i in range(boxes.shape[0]):
            if i % 2 == 0: continue
            pt1 = boxes[i - 1][:2].round().numpy().astype(int)
            pt2 = boxes[i][:2].round().numpy().astype(int)
            cv.line(ground, pt1, pt2, 1, self.LINE_THICKNESS, lineType=cv.LINE_AA)
        return Mask(np.expand_dims(ground, 0))

    def _load_anns(self, id:int) -> Tuple[List[Any], str]:
        label_path = self.label_paths[id]
        with open(label_path) as f:
            data = json.load(f)
        return data['shapes'], data['imagePath']

    def __getitem__(self, index:int) -> Tuple[Any, Any]:

        if not isinstance(index, int):
            raise ValueError(f'Index must be of type integer, got {type(index)} instead.')

        anns, imagePath = self._load_anns(index)
        if len(anns) == 0:
            return self.__getitem__((index + 1) % self.__len__())

        image = self._load_image(imagePath)
        target = self._format_anns(anns, image.size)
        if target['boxes'].numel() == 0:
            return self.__getitem__((index + 1) % self.__len__())

        num_boxes = len(target['boxes'])
        if self.transforms is not None:
            max_diameter = target['boxes'][:, 2:].max()
            target['boxes'][:, 2:] /= max_diameter
            image, target = self.transforms(image, target)
            target['boxes'][:, 2:] *= max_diameter

        for i in range(0, len(target['boxes']), 2):
            x1, y1 = target['boxes'][i][:2].tolist()
            x2, y2 = target['boxes'][i + 1][:2].tolist()
            if abs(x2 - x1) > abs(y2 - y1):
                if x1 <= x2: continue
            else:
                if y1 <= y2: continue
            target['boxes'][i][0] = x2
            target['boxes'][i][1] = y2
            target['boxes'][i + 1][0] = x1
            target['boxes'][i + 1][1] = y1

        target['masks'] = self._draw_masks(target['boxes'])
        target['boxes'] = box_convert(target['boxes'], 'cxcywh', 'xyxy')
        assert len(target['boxes']) == num_boxes

        return image, target
