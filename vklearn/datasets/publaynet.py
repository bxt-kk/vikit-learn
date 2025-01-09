from typing import Any, Callable, List, Tuple, Dict
from collections import Counter
import os.path

from PIL import Image

from torch import Tensor
import torch
from torchvision.datasets.vision import VisionDataset
from torchvision import tv_tensors

from tqdm import tqdm


class PubLayNetDet(VisionDataset):
    '''`PubLayNet Detection Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root: Root directory where images are downloaded to.
        annFile: Path to json annotation file.
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
        transform:        Callable | None=None,
        target_transform: Callable | None=None,
        transforms:       Callable | None=None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.coid2class = {
            clss['id']: clss['name']
            for clss in self.coco.dataset['categories']}

        idxs = sorted(self.coid2class.keys())
        self.classes = [self.coid2class[i] for i in idxs]

    def __len__(self) -> int:
        return len(self.ids)

    def calc_balance_weight(self, gamma:float=0.1) -> Tensor:
        weight = torch.zeros(len(self.classes))
        counter = Counter()
        print('count categories...')
        for _id in tqdm(self.ids, ncols=80):
            anns = self._load_anns(_id)
            for ann in anns:
                category_id = ann['category_id']
                classname = self.coid2class[category_id]
                counter[classname] += 1
        for name, count in counter.items():
            label_id = self.classes.index(name)
            weight[label_id] = 1 / count
        weight = weight**gamma
        weight /= weight.mean()
        return weight

    def _load_image(self, id:int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]['file_name']
        return Image.open(os.path.join(self.root, path)).convert('RGB')

    def _load_anns(self, id:int) -> List[Any]:
        # return self.coco.loadAnns(self.coco.getAnnIds(id))
        return [
            ann for ann in self.coco.loadAnns(self.coco.getAnnIds(id))
            if ann['category_id'] > 0]

    def _format_anns(
            self,
            anns:       List[Any],
            image_size: Tuple[int, int],
        ) -> Dict[str, Any]:

        xywh2xyxy  = lambda x, y, w, h: (x, y, x + w, y + h)
        validation = lambda ann: ann['iscrowd'] == 0
        box_list = []
        label_list = []
        for ann in anns:
            if not validation(ann): continue
            class_name = self.coid2class[ann['category_id']]
            box_list.append(xywh2xyxy(*ann['bbox']))
            label_list.append(self.classes.index(class_name))
        boxes = tv_tensors.BoundingBoxes(
            box_list,
            format='XYXY',
            canvas_size=(image_size[1], image_size[0]),
        )
        labels = torch.LongTensor(label_list)
        return dict(boxes=boxes, labels=labels)

    def __getitem__(self, index:int) -> Tuple[Any, Any]:
        if not isinstance(index, int):
            raise ValueError(
                f'Index must be of type integer, got {type(index)} instead.')

        id = self.ids[index]
        anns = self._load_anns(id)
        if len(anns) == 0:
            return self.__getitem__((index + 1) % self.__len__())

        image = self._load_image(id)
        target = self._format_anns(anns, image.size)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
