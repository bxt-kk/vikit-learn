from typing import Any, Callable, List, Tuple
import os.path
from glob import glob
import json

from PIL import Image
import numpy as np

import torch
from torchvision.datasets.vision import VisionDataset
from torchvision import tv_tensors


class LabelmeDetection(VisionDataset):
    '''`Labelme annotated format common dataset.

    Args:
        root: Root directory where images are downloaded to.
        split: The dataset split, supports ``""`` (default), ``"train"``, ``"valid"`` or ``"test"``.
        transform: A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform: A function/transform that takes in the
            target and transforms it.
        transforms: A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    '''

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
        with open(os.path.join(root, 'classnames.txt')) as f:
            for name in f:
                name = name.strip()
                if not name: continue
                self.classes.append(name)

    def __len__(self) -> int:
        return len(self.label_paths)

    def _load_image(self, path:str) -> Image.Image:
        return Image.open(os.path.join(self.dataset_dir, path)).convert('RGB')

    def _load_anns(self, id:int) -> Tuple[List[Any], str]:
        label_path = self.label_paths[id]
        with open(label_path) as f:
            data = json.load(f)
        return data['shapes'], data['imagePath']

    def _points2xyxy(self, points:List[List[float]]) -> List[float]:
        array = np.asarray(points, dtype=np.float32)
        x1, y1 = array.min(axis=0)
        x2, y2 = array.max(axis=0)
        return [x1, y1, x2, y2]

    def _format_anns(
            self,
            anns:       List[Any],
            image_size: Tuple[int, int],
        ) -> dict[str, Any]:
        boxes = tv_tensors.BoundingBoxes(
            [self._points2xyxy(ann['points']) for ann in anns],
            format='XYXY',
            canvas_size=(image_size[1], image_size[0]),
        )
        labels = torch.LongTensor([self.classes.index(ann['label']) for ann in anns])
        return dict(boxes=boxes, labels=labels)

    def __getitem__(self, index:int) -> Tuple[Any, Any]:

        if not isinstance(index, int):
            raise ValueError(f'Index must be of type integer, got {type(index)} instead.')

        anns, imagePath = self._load_anns(index)
        if len(anns) == 0:
            return self.__getitem__((index + 1) % self.__len__())

        image = self._load_image(imagePath)
        target = self._format_anns(anns, image.size)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
