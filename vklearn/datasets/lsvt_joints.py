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


class LSVTJoints(VisionDataset):
    '''`LSVT Joints Detection dataset.

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
        root:                str,
        ignore_illegibility: bool=True,
        transform:           Callable | None=None,
        target_transform:    Callable | None=None,
        transforms:          Callable | None=None,
    ):
        super().__init__(root, transforms, transform, target_transform)

        self.image_paths = sorted(glob(os.path.join(root, 'train_full_images_*/*/*.jpg')))
        label_path = os.path.join(root, 'train_full_labels.json')

        with open(label_path) as f:
            self.anns_dict = json.load(f)

        self.classes = ['text']
        self.ignore_illegibility = ignore_illegibility

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image(self, image_paths:str) -> Image.Image:
        return Image.open(image_paths).convert('RGB')

    def _format_anns(
        self,
        anns:       List[Any],
        image_size: Tuple[int, int],
    ) -> Dict[str, Any]:

        bbox_list = []
        for ann in anns:
            if self.ignore_illegibility and ann['illegibility']: continue
            (x, y), (w, h), a = cv.minAreaRect(np.array(ann['points']))
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

        return dict(
            boxes=BoundingBoxes(
                bbox_list,
                format='CXCYWH',
                canvas_size=(image_size[1], image_size[0])),
            labels=torch.LongTensor([0] * len(bbox_list)),
        )

    def _draw_masks(self, boxes:BoundingBoxes) -> Mask:
        ground = np.zeros(boxes.canvas_size, dtype=np.uint8)
        for i in range(boxes.shape[0]):
            if i % 2 == 0: continue
            pt1 = boxes[i - 1][:2].round().numpy().astype(int)
            pt2 = boxes[i][:2].round().numpy().astype(int)
            thickness = max(1, min(self.LINE_THICKNESS, round(0.33 * boxes[i -1][3].item())))
            cv.line(ground, pt1, pt2, 1, thickness, lineType=cv.LINE_AA)
        return Mask(np.expand_dims(ground, 0))

    def __getitem__(self, index:int) -> Tuple[Any, Any]:

        if not isinstance(index, int):
            raise ValueError(f'Index must be of type integer, got {type(index)} instead.')

        image_path = self.image_paths[index]
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        anns = self.anns_dict[image_id]
        if len(anns) == 0:
            return self.__getitem__((index + 1) % self.__len__())

        image = self._load_image(image_path)
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    from matplotlib.pyplot import Circle

    dataset = LSVTJoints('/media/kk/Data/dataset/image/LSVT')
    print(len(dataset))

    for i in range(len(dataset)):
        image, target = dataset[i]
        ax:plt.Axes = plt.subplot()
        img_arr = np.array(image, dtype=np.uint8)
        mask = target['masks'].numpy()
        img_arr[..., 1][mask[0] == 1] = 0
        ax.imshow(img_arr)
        for bnd_id, bbox in enumerate(target['boxes']):
            is_begin = bnd_id % 2 == 0
            x, y = (bbox[:2] + bbox[2:]) * 0.5
            diameter = (bbox[2:] - bbox[:2]).min()
            color = 'red' if is_begin else 'blue'
            ax.add_patch(Circle((x, y), diameter * 0.5, color=color, fill=False, linewidth=1))
            ax.add_patch(Circle((x, y), 5, color=color, fill=True))
        plt.show()
        if input('continue?>').strip() == 'q': break
