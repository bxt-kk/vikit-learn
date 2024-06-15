from typing import Callable, Tuple, Any, Dict, List
from pathlib import Path
from xml.etree.ElementTree import parse as ET_parse

from torchvision.datasets import VOCDetection as _VOCDetection
from torchvision import tv_tensors
import torch
from PIL import Image

from tqdm import tqdm


class VOCDetection(_VOCDetection):
    DEFAULT_CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
    ]

    def __init__(
        self,
        root:             str | Path,
        year:             str='2012',
        image_set:        str='train',
        sub_categories:   List[str] | None=None,
        download:         bool=False,
        transform:        Callable | None=None,
        target_transform: Callable | None=None,
        transforms:       Callable | None=None,
    ):
        super().__init__(
            root, year, image_set, download, transform, target_transform, transforms)

        self.classes = sub_categories or self.DEFAULT_CLASSES
        if len(self.classes) != len(self.DEFAULT_CLASSES):
            ids = list(range(len(self.images)))
            ids = self._drop_others(ids)
            self.images = [self.images[idx] for idx in ids]
            self.targets = [self.targets[idx] for idx in ids]

    def _drop_others(self, ids:List[int]) -> List[int]:
        new_ids = []
        for idx in tqdm(ids, ncols=80):
            anns = self.parse_voc_xml(ET_parse(self.annotations[idx]).getroot())
            anns = anns['annotation']
            for obj in anns['object']:
                name = obj['name']
                if name in self.classes:
                    new_ids.append(idx)
                    break
        return new_ids

    def _format_anns(self, anns:Dict[str, Any]) -> Dict[str, Any]:
        anns = anns['annotation']
        box_list = []
        label_list = []
        for obj in anns['object']:
            name = obj['name']
            if name not in self.classes: continue
            bbox_raw = obj['bndbox']
            bbox = [float(bbox_raw[k]) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
            box_list.append(bbox)
            label_list.append(self.classes.index(name))
        size_w = int(anns['size']['width'])
        size_h = int(anns['size']['height'])
        boxes = tv_tensors.BoundingBoxes(
            box_list,
            format='XYXY',
            canvas_size=(size_h, size_w),
        )
        labels = torch.LongTensor(label_list)
        return dict(boxes=boxes, labels=labels)

    def __getitem__(self, idx:int) -> Tuple[Any, Any]:
        img = Image.open(self.images[idx]).convert('RGB')
        anns = self.parse_voc_xml(ET_parse(self.annotations[idx]).getroot())
        target = self._format_anns(anns)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
