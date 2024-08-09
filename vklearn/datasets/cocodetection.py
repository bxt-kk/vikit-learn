from typing import Any, Callable, List, Tuple, Dict
import os.path
import math
from collections import Counter

from PIL import Image

from torch import Tensor
import torch
from torchvision.datasets.vision import VisionDataset
from torchvision import tv_tensors

from tqdm import tqdm


class CocoDetection(VisionDataset):
    '''`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root: Root directory where images are downloaded to.
        annFile: Path to json annotation file.
        category_type: The type of category, can be ``name`` or ``supercategory``.
        sub_categories: Select some target categories as a List of subcategories.
        max_datas_size: For large amounts of data, it is used to limit the number of samples,
            and the default is 0, which means no limit.
        transform: A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform: A function/transform that takes in the
            target and transforms it.
        transforms: A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    '''

    NAME_OBJECT = 'object'
    NAME_OTHER  = 'other'

    def __init__(
        self,
        root:             str,
        annFile:          str,
        category_type:    str='name',
        sub_categories:   List[str] | None=None,
        lowpoly_size:     int=0,
        max_datas_size:   int=0,
        dropout_k:        int=0,
        transform:        Callable | None=None,
        target_transform: Callable | None=None,
        transforms:       Callable | None=None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        assert category_type in ('name', 'supercategory', 'object')
        assert lowpoly_size in (0, 8, 16, 24, 32)

        if sub_categories is None: sub_categories = []

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.category_type = category_type
        self.lowpoly_size = lowpoly_size

        self.coid2name = {
            clss['id']: clss['name']
            for clss in self.coco.dataset['categories']}
        self.coid2supercategory = {
            clss['id']: clss['supercategory']
            for clss in self.coco.dataset['categories']}
        self.coid2object = {
            clss['id']: self.NAME_OBJECT
            for clss in self.coco.dataset['categories']}
        subtype = category_type if category_type != 'object' else 'name'
        self.coid2subcategory = {
            clss['id']: (clss[subtype] if clss[subtype] in sub_categories else self.NAME_OTHER)
            for clss in self.coco.dataset['categories']}

        idxs = sorted(self.coid2name.keys())
        self.names = [self.coid2name[i] for i in idxs]
        # self.objcategories = [self.NAME_OBJECT, self.NAME_OTHER]
        self.objcategories = [self.NAME_OBJECT]
        self.supercategories = []
        for i in idxs:
            category = self.coid2supercategory[i]
            if category in self.supercategories: continue
            self.supercategories.append(category)
        # self.subcategories = sub_categories + [self.NAME_OTHER]
        self.subcategories = sub_categories

        self.ids = self._drop_topk(self.ids, dropout_k)

        if len(sub_categories) > 0:
            self.classes = self.subcategories
            self.coid2class = self.coid2subcategory
            self.ids = self._drop_other_images(self.ids)
        elif category_type == 'name':
            self.classes = self.names
            self.coid2class = self.coid2name
        elif category_type == 'supercategory':
            self.classes = self.supercategories
            self.coid2class = self.coid2supercategory
        elif category_type == 'object':
            self.classes = self.objcategories
            self.coid2class = self.coid2object

        self.max_datas_size = max_datas_size if max_datas_size > 0 else len(self.ids)

    def __len__(self) -> int:
        return min(self.max_datas_size, len(self.ids))

    def calc_balance_weight(self, gamma:float=0.5) -> Tensor:
        weight = torch.zeros(len(self.classes))
        counter = Counter()
        print('count categories...')
        for _id in tqdm(self.ids, ncols=80):
            anns = self._load_anns(_id)
            for ann in anns:
                category_id = ann['category_id']
                classname = self.coid2class[category_id]
                counter[classname] += 1
        names_desc = [name for name, _ in counter.most_common()]
        count_asc = [count for _, count in counter.most_common()][::-1]
        total = counter.total()
        for name, count in zip(names_desc, count_asc):
            label_id = self.classes.index(name)
            proportional = count / total
            weight[label_id] = proportional
        weight = weight**gamma
        weight /= weight.mean()
        return weight

    def _drop_topk(self, ids:List[int], k:int) -> List[int]:
        if k == 0: return ids
        count = Counter()
        print('count categories...')
        for _id in tqdm(ids, ncols=80):
            anns = self._load_anns(_id)
            for ann in anns:
                category_id = ann['category_id']
                count[category_id] += 1
        hold_categories = [cid for cid, _ in count.most_common()[k:]]
        new_ids = []
        print('drop images...')
        for _id in tqdm(ids, ncols=80):
            anns = self._load_anns(_id)
            for ann in anns:
                category_id = ann['category_id']
                if category_id in hold_categories:
                    new_ids.append(_id)
                    break
        print('ids size:', len(ids), ', new_ids size:', len(new_ids))
        return new_ids

    def _drop_other_images(self, ids:List[int]) -> List[int]:
        new_ids = []
        for _id in tqdm(ids, ncols=80):
            anns = self._load_anns(_id)
            for ann in anns:
                class_name = self.coid2class[ann['category_id']]
                if class_name != self.NAME_OTHER:
                    new_ids.append(_id)
                    break
        return new_ids

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
        # boxes = tv_tensors.BoundingBoxes(
        #     [xywh2xyxy(*ann['bbox']) for ann in anns if validation(ann)],
        #     format='XYXY',
        #     canvas_size=(image_size[1], image_size[0]),
        # )
        # labels = torch.LongTensor([
        #     self.classes.index(self.coid2class[ann['category_id']])
        #     for ann in anns if validation(ann)])
        box_list = []
        label_list = []
        for ann in anns:
            if not validation(ann): continue
            class_name = self.coid2class[ann['category_id']]
            if class_name == self.NAME_OTHER: continue
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

        if self.lowpoly_size > 0:
            labels = target['labels']
            boxes = target['boxes']
            sl = self.lowpoly_size
            for i, label_id in enumerate(labels):
                x1, y1, x2, y2 = boxes[i]
                boxes[i, 0] = x1 // sl * sl
                boxes[i, 1] = y1 // sl * sl
                boxes[i, 2] = min(math.ceil(x2 / sl) * sl, image.shape[2])
                boxes[i, 3] = min(math.ceil(y2 / sl) * sl, image.shape[1])
            target['labels'] = labels
            target['boxes'] = boxes

        return image, target
