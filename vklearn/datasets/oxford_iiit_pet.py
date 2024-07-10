from typing import Any, Callable, Sequence, Tuple, Dict
import os
import os.path
import pathlib
import xml.etree.ElementTree as ET

from PIL import Image

import torch
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset
from torchvision import tv_tensors


class OxfordIIITPet(VisionDataset):
    '''`Oxford-IIIT Pet Dataset   <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_.

    Args:
        root: Root directory of the dataset.
        split: The dataset split, supports ``"trainval"`` (default) or ``"test"``.
        target_types: Types of target to use. Can be ``category`` (default) or
            ``segmentation``. Can also be a list to output a tuple with all specified target types. The types represent:

                - ``category`` (int): Label for one of the 37 pet categories.
                - ``binary-category`` (int): Binary label for cat or dog.
                - ``segmentation`` (PIL image): Segmentation trimap of the image.
                - ``detection`` (PIL image, Labels & BoundingBoxes): Detection annotation.

            If empty, ``None`` will be returned as target.

        transform: A function/transform that takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform: A function/transform that takes in the target and transforms it.
        download: If True, downloads the dataset from the internet and puts it into
            ``root/oxford-iiit-pet``. If dataset is already downloaded, it is not downloaded again.
    '''

    _RESOURCES = (
        ('https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz', '5c4f3ee8e5d25df40f4fd59a7f44e54c'),
        ('https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz', '95a8c909bbe2e81eed6a22bccdf3f68f'),
    )
    _VALID_TARGET_TYPES = ('category', 'binary-category', 'segmentation', 'detection')

    def __init__(
        self,
        root:             str | pathlib.Path,
        split:            str='trainval',
        target_types:     Sequence[str] | str='category',
        transforms:       Callable | None=None,
        transform:        Callable | None=None,
        target_transform: Callable | None=None,
        download:         bool=False,
    ):
        self._split = verify_str_arg(split, 'split', ('trainval', 'test'))
        if isinstance(target_types, str):
            target_types = [target_types]
        self._target_types = [
            verify_str_arg(target_type, 'target_types', self._VALID_TARGET_TYPES)
            for target_type in target_types]

        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)
        self._base_folder = pathlib.Path(self.root) # / 'oxford-iiit-pet'
        self._images_folder = self._base_folder / 'images'
        self._anns_folder = self._base_folder / 'annotations'
        self._segs_folder = self._anns_folder / 'trimaps'
        self._objs_folder = self._anns_folder / 'xmls'

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it')

        image_ids = []
        self._labels = []
        self._bin_labels = []
        with open(self._anns_folder / f'{self._split}.txt') as file:
            for line in file:
                image_id, label, bin_label, _ = line.strip().split()
                image_ids.append(image_id)
                self._labels.append(int(label) - 1)
                self._bin_labels.append(int(bin_label) - 1)

        self.bin_classes = ['Cat', 'Dog']
        self.classes = [
            ' '.join(part.title() for part in raw_cls.split('_'))
            for raw_cls, _ in sorted(
                {(image_id.rsplit('_', 1)[0], label) for image_id, label in zip(image_ids, self._labels)},
                key=lambda image_id_and_label: image_id_and_label[1],
            )
        ]
        self.bin_class_to_idx = dict(zip(self.bin_classes, range(len(self.bin_classes))))
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self.tag_to_ix = dict(zip(['xmin', 'ymin', 'xmax', 'ymax'], range(4)))

        self._images = [self._images_folder / f'{image_id}.jpg' for image_id in image_ids]
        self._segs = [self._segs_folder / f'{image_id}.png' for image_id in image_ids]
        if 'detection' not in target_types:
            return
        _images = []
        _segs = []
        _labels = []
        _bin_labels = []
        self._objs = []
        for i, image_id in enumerate(image_ids):
            xml_path = self._objs_folder / f'{image_id}.xml'
            if not os.path.exists(xml_path): continue
            _images.append(self._images[i])
            _segs.append(self._segs[i])
            _labels.append(self._labels[i])
            _bin_labels.append(self._bin_labels[i])
            self._objs.append(xml_path)
        self._images = _images
        self._segs = _segs
        self._labels = _labels
        self._bin_labels = _bin_labels


    def __len__(self) -> int:
        return len(self._images)

    def _load_obj(self, idx:int, image_size: Tuple[int, int]) -> Dict[str, Any]:
        path = self._objs[idx]
        with open(path) as f:
            tree = ET.parse(f)
        root = tree.getroot()
        for cursor in root:
            if cursor.tag == 'object': break
        for bndbox in cursor:
            if bndbox.tag == 'bndbox': break

        bbox = [0] * 4
        for item in bndbox:
            bbox_ix = self.tag_to_ix[item.tag]
            bbox[bbox_ix] = float(item.text)

        return dict(
            # labels=torch.LongTensor([self._labels[idx]]),
            labels=torch.LongTensor([self._bin_labels[idx]]),
            boxes=tv_tensors.BoundingBoxes(
                [bbox],
                format='XYXY',
                canvas_size=(image_size[1], image_size[0]),
            )
        )

    def _load_mask(self, idx:int) -> tv_tensors.Mask:
        image = Image.open(self._segs[idx])
        mask = tv_tensors.Mask(image)
        mask[mask != 1] = 0
        rows, cols = mask.shape[1], mask.shape[2]
        data = torch.zeros((2, rows, cols), dtype=mask.dtype)
        data[self._bin_labels[idx]] = mask.data
        return tv_tensors.Mask(data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image = Image.open(self._images[idx]).convert('RGB')

        target: Any = []
        for target_type in self._target_types:
            if target_type == 'category':
                target.append(self._labels[idx])
            elif target_type == 'binary-category':
                target.append(self._bin_labels[idx])
            elif target_type == 'segmentation':
                target.append(self._load_mask(idx))
            elif target_type == 'detection':
                target.append(self._load_obj(idx, image.size))

        if not target:
            target = None
        elif len(target) == 1:
            target = target[0]
        else:
            target = tuple(target)

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    def _check_exists(self) -> bool:
        for folder in (self._images_folder, self._anns_folder):
            if not (os.path.exists(folder) and os.path.isdir(folder)):
                return False
        return True

    def _download(self) -> None:
        if self._check_exists(): return

        for url, md5 in self._RESOURCES:
            download_and_extract_archive(
                url, download_root=str(self._base_folder), md5=md5)
