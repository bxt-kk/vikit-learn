from typing import Any, Callable, List, Tuple
import os.path

from PIL import Image

import torch
from torchvision.datasets.vision import VisionDataset
from torchvision import tv_tensors


class CocoDetection(VisionDataset):
    '''`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root: Root directory where images are downloaded to.
        annFile: Path to json annotation file.
        max_datas_size: For large amounts of data, it is used to limit the number of samples,
            and the default is 0, which means no limit.
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
        annFile:          str,
        category_type:    str='name',
        max_datas_size:   int=0,
        transform:        Callable | None=None,
        target_transform: Callable | None=None,
        transforms:       Callable | None=None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        assert category_type in ('name', 'supercategory')

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.category_type = category_type
        self.max_datas_size = max_datas_size if max_datas_size > 0 else len(self.ids)

        self.coid2name = {
            clss['id']: clss['name']
            for clss in self.coco.dataset['categories']}
        self.coid2supercategory = {
            clss['id']: clss['supercategory']
            for clss in self.coco.dataset['categories']}

        self.names = [
            self.coid2name[i]
            for i in range(1, len(self.coid2name) + 1)]
        self.supercategories = []
        for i in range(1, len(self.coid2supercategory) + 1):
            category = self.coid2supercategory[i]
            if category in self.supercategories: continue
            self.supercategories.append(category)

        if category_type == 'name':
            self.classes = self.names
            self.coid2class = self.coid2name
        elif category_type == 'supercategory':
            self.classes = self.supercategories
            self.coid2class = self.coid2supercategory

    def __len__(self) -> int:
        return min(self.max_datas_size, len(self.ids))

    def _load_image(self, id:int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_anns(self, id:int) -> List[Any]:
        # return self.coco.loadAnns(self.coco.getAnnIds(id))
        return [
            ann for ann in self.coco.loadAnns(self.coco.getAnnIds(id))
            if ann['category_id'] > 0]

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
        labels = torch.LongTensor([
            self.classes.index(self.coid2class[ann['category_id']])
            for ann in anns if validation(ann)])
        return dict(boxes=boxes, labels=labels)

    def __getitem__(self, index:int) -> Tuple[Any, Any]:

        if not isinstance(index, int):
            raise ValueError(f'Index must be of type integer, got {type(index)} instead.')

        id = self.ids[index]
        anns = self._load_anns(id)
        if len(anns) == 0:
            return self.__getitem__((index + 1) % self.__len__())

        image = self._load_image(id)
        target = self._format_anns(anns, image.size)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
