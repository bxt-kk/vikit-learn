from typing import Any, Callable, List, Tuple, Dict
import os.path
import json

from PIL import Image

import torch
from torchvision.datasets.vision import VisionDataset
from torchvision import tv_tensors


COCO_LABELS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'ms_coco_classnames.json')

with open(COCO_LABELS_PATH) as f:
    coco_classnames = {int(_id) - 1: name
        for _id, name in json.load(f).items()
        if _id != '0'}


class CocoDetection(VisionDataset):
    '''`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root: Root directory where images are downloaded to.
        annFile: Path to json annotation file.
        category_max_id: Set the maximum category ID to limit the target category range.
        max_datas_size: For large amounts of data, it is used to limit the number of samples,
            and the default is 0, which means no limit.
        transform: A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform: A function/transform that takes in the
            target and transforms it.
        transforms: A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    '''
    CLASSNAMES:Dict[int, str] = coco_classnames

    def __init__(
        self,
        root:             str,
        annFile:          str,
        category_max_id:  int=80,
        max_datas_size:   int=0,
        transform:        Callable | None=None,
        target_transform: Callable | None=None,
        transforms:       Callable | None=None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.category_max_id = category_max_id
        self.max_datas_size = max_datas_size if max_datas_size > 0 else len(self.ids)

    def __len__(self) -> int:
        return min(self.max_datas_size, len(self.ids))

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_anns(self, id: int) -> List[Any]:
        # return self.coco.loadAnns(self.coco.getAnnIds(id))
        return [
            ann for ann in self.coco.loadAnns(self.coco.getAnnIds(id))
            if (0 < ann['category_id']) and (ann['category_id'] <= self.category_max_id)]

    def _format_anns(
            self,
            anns:       List[Any],
            image_size: Tuple[int, int],
        ) -> dict[str, Any]:
        xywh2xyxy  = lambda x, y, w, h: (x, y, x + w, y + h)
        validation = lambda ann: ann['iscrowd'] == 0
        boxes = tv_tensors.BoundingBoxes(
            [xywh2xyxy(*ann['bbox']) for ann in anns if validation(ann)],
            format='XYXY',
            canvas_size=(image_size[1], image_size[0]),
        )
        labels = torch.LongTensor([ann['category_id'] - 1 for ann in anns if validation(ann)])
        return dict(boxes=boxes, labels=labels)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        if not isinstance(index, int):
            raise ValueError(f"Index must be of type integer, got {type(index)} instead.")

        id = self.ids[index]
        anns = self._load_anns(id)
        if len(anns) == 0:
            return self.__getitem__((index + 1) % self.__len__())

        image = self._load_image(id)
        target = self._format_anns(anns, image.size)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
