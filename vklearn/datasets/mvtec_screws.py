from typing import Any, Callable, List, Tuple, Dict
from collections import defaultdict
import os.path
import json
import math

from PIL import Image
import cv2 as cv
import numpy as np

import torch
from torchvision.datasets.vision import VisionDataset
from torchvision.tv_tensors import BoundingBoxes, Mask
from torchvision.ops import box_convert


class MVTecScrews(VisionDataset):
    '''`MVTec-Screws OBB Detection dataset.

    Args:
        root: Root directory where images are downloaded to.
        split: The dataset split, supports ``""`` (default), ``"train"``, ``"val"`` or ``"test"``.
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
        split:            str='train',
        transform:        Callable | None=None,
        target_transform: Callable | None=None,
        transforms:       Callable | None=None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        assert split in ['train', 'val', 'test']

        self.images_dir = os.path.join(root, 'images/')

        label_path = os.path.join(root, f'mvtec_screws_{split}.json')

        with open(label_path) as f:
            data = json.load(f)

        self.id2category = {item['id'] - 1: item['name'] for item in data['categories']}
        self.classes = list(self.id2category.values())

        self.image_id2filename = {item['id']: item['file_name'] for item in data['images']}
        self.image_ids = list(self.image_id2filename.keys())

        self.anns_dict = defaultdict(list)
        for ann in data['annotations']:
            label = ann['category_id'] - 1
            bbox = ann['bbox']
            image_id = ann['image_id']
            self.anns_dict[image_id].append(dict(
                label=label,
                bbox=bbox,
            ))

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load_image(self, image_id:int) -> Image.Image:
        filename = self.image_id2filename[image_id]
        return Image.open(
            os.path.join(self.images_dir, filename)).convert('RGB')

    def _format_anns(
        self,
        anns:       List[Any],
        image_size: Tuple[int, int],
    ) -> Dict[str, Any]:

        bbox_list = []
        label_list = []
        for ann in anns:
            y, x, w, h, a = ann['bbox']
            a = - a / math.pi * 180
            pts = cv.boxPoints(((x, y), (w, h), a))

            x1, y1 = (pts[0] + pts[1]) * 0.5
            x2, y2 = (pts[2] + pts[3]) * 0.5
            diameter = min(w, h)

            bbox_list.append([x1, y1, diameter, diameter])
            bbox_list.append([x2, y2, diameter, diameter])
            label_list.append(ann['label'])
            label_list.append(ann['label'])

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

    def __getitem__(self, index:int) -> Tuple[Any, Any]:

        if not isinstance(index, int):
            raise ValueError(f'Index must be of type integer, got {type(index)} instead.')

        image_id = self.image_ids[index]
        anns = self.anns_dict[image_id]
        if len(anns) == 0:
            return self.__getitem__((index + 1) % self.__len__())

        image = self._load_image(image_id)
        target = self._format_anns(anns, image.size)

        num_boxes = len(target['boxes'])
        if self.transforms is not None:
            max_diameter = target['boxes'][:, 2:].max()
            target['boxes'][:, 2:] /= max_diameter
            image, target = self.transforms(image, target)
            target['boxes'][:, 2:] *= max_diameter
        target['masks'] = self._draw_masks(target['boxes'])
        target['boxes'] = box_convert(target['boxes'], 'cxcywh', 'xyxy')
        assert len(target['boxes']) == num_boxes

        return image, target


if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    from matplotlib.pyplot import Circle

    dataset = MVTecScrews('/media/kk/Data/dataset/image/MVTec-Screws', 'test')
    print(len(dataset))

    for i in range(3):
        image, target = dataset[i]
        ax:plt.Axes = plt.subplot()
        img_arr = np.array(image, dtype=np.uint8)
        mask = target['masks'].numpy()
        img_arr[..., 0][mask[0] == 1] = 0
        ax.imshow(img_arr)
        for bnd_id, bbox in enumerate(target['boxes']):
            is_begin = bnd_id % 2 == 0
            x, y, diameter, _ = bbox
            color = 'red' if is_begin else 'blue'
            ax.add_patch(Circle((x, y), diameter * 0.5, color=color, fill=False, linewidth=1))
            ax.add_patch(Circle((x, y), 10, color=color, fill=True))
        plt.show()
