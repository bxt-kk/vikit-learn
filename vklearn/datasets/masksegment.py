from typing import Callable, Tuple, Any, Dict, List
from xml.etree.ElementTree import parse as ET_parse
from glob import glob
import os

from torchvision.datasets.vision import VisionDataset
from torchvision import tv_tensors
import torch
import torch.nn.functional as F
from PIL import Image


class MaskSegment(VisionDataset):

    def __init__(
            self,
            root:             str,
            split:            str='train',
            categories:       List[str] | None=None,
            transform:        Callable | None=None,
            target_transform: Callable | None=None,
            transforms:       Callable | None=None,
        ):

        super().__init__(
            root, transforms=transforms, transform=transform, target_transform=target_transform)

        labels_file = os.path.join(root, 'labels.txt')
        self.classes = categories
        if os.path.isfile(labels_file):
            with open(labels_file, 'w') as f:
                self.classes = [label.strip() for label in f]
        assert self.classes is not None

        self._images = sorted(glob(os.path.join(root, split, 'image', '*')))
        self._masks = sorted(glob(os.path.join(root, split, 'mask', '*')))

    def __len__(self) -> int:
        return len(self._images)

    def _format_mask(self, mask:tv_tensors.Mask) -> tv_tensors.Mask:
        return tv_tensors.Mask(
            F.one_hot(mask, len(self.classes)).transpose(0, 3).squeeze(-1))

    def __getitem__(self, idx:int) -> Tuple[Any, Any]:
        image = Image.open(self._images[idx]).convert('RGB')
        target = Image.open(self._masks[idx])
        assert target.mode == 'P'
        if image.size != target.size:
            image = image.resize(target.size, resample=Image.Resampling.BICUBIC)
        target = tv_tensors.Mask(target, dtype=torch.long)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        target = self._format_mask(target)
        return image, target
