from typing import Tuple, Any, Dict
from xml.etree.ElementTree import parse as ET_parse

from torchvision.datasets import VOCDetection as _VOCDetection
from torchvision import tv_tensors
import torch
from PIL import Image


class VOCDetection(_VOCDetection):
    classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
    ]

    @classmethod
    def _format_anns(cls, anns:Dict[str, Any]) -> Dict[str, Any]:
        anns = anns['annotation']
        box_list = []
        label_list = []
        for obj in anns['object']:
            name = obj['name']
            bbox_raw = obj['bndbox']
            bbox = [float(bbox_raw[k]) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
            box_list.append(bbox)
            label_list.append(cls.classes.index(name))
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
