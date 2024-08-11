from typing import List, Any, Dict, Tuple

from torch import Tensor

import torch

from torchvision import tv_tensors
from torchvision.transforms import v2

from torchmetrics.segmentation import MeanIoU

from PIL import Image

from .basic import Basic


class Segment(Basic):

    def __init__(
            self,
            categories: List[str],
        ):
        super().__init__()

        self.categories   = list(categories)
        self.num_classes  = len(categories)

        self.m_iou = MeanIoU(
            num_classes=self.num_classes)

    def segment(
            self,
            image:       Image.Image,
            conf_thresh: float=0.5,
            align_size:  int=448,
        ) -> List[Dict[str, Any]]:
        assert not 'this is an empty func'

    def calc_loss(
            self,
            inputs:  Tensor,
            target:  Tensor,
            weights: Dict[str, float] | None=None,
        ) -> Dict[str, Any]:
        assert not 'this is an empty func'

    def calc_score(
            self,
            inputs: Tensor,
            target: Tensor,
            eps:    float=1e-5,
        ) -> Dict[str, Any]:
        assert not 'this is an empty func'

    def update_metric(
            self,
            inputs:      Tensor,
            target:      Tensor,
            conf_thresh: float=0.5,
        ):
        assert not 'this is an empty func'

    def compute_metric(self) -> Dict[str, Any]:
        miou = self.m_iou.compute() / self.m_iou.update_count
        return dict(miou=miou)

    @classmethod
    def get_transforms(
            cls,
            task_name: str='default',
        ) -> Tuple[v2.Transform, v2.Transform]:

        train_transforms = None
        test_transforms  = None

        if task_name in ('default', 'cocox448'):
            train_transforms = v2.Compose([
                v2.ToImage(),
                v2.ScaleJitter(
                    target_size=(448, 448),
                    scale_range=(384 / 448, 1.1),
                    antialias=True),
                v2.RandomPhotometricDistort(p=1),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomChoice([
                    v2.GaussianBlur(7, sigma=(0.1, 2.0)),
                    v2.RandomAdjustSharpness(2, p=0.5),
                ]),
                v2.RandomCrop(
                    size=(448, 448),
                    pad_if_needed=True,
                    fill={tv_tensors.Image: 127, tv_tensors.Mask: 0}),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])
            test_transforms = v2.Compose([
                v2.ToImage(),
                v2.Resize(
                    size=447,
                    max_size=448,
                    antialias=True),
                v2.CenterCrop(448),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])

        elif task_name == 'cocox384':
            train_transforms = v2.Compose([
                v2.ToImage(),
                v2.ScaleJitter(
                    target_size=(384, 384),
                    scale_range=(0.9, 1.1),
                    antialias=True),
                v2.RandomPhotometricDistort(p=1),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomChoice([
                    v2.GaussianBlur(7, sigma=(0.1, 2.0)),
                    v2.RandomAdjustSharpness(2, p=0.5),
                ]),
                v2.RandomCrop(
                    size=(384, 384),
                    pad_if_needed=True,
                    fill={tv_tensors.Image: 127, tv_tensors.Mask: 0}),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])
            test_transforms = v2.Compose([
                v2.ToImage(),
                v2.Resize(
                    size=383,
                    max_size=384,
                    antialias=True),
                v2.CenterCrop(384),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])

        elif task_name == 'cocox512':
            train_transforms = v2.Compose([
                v2.ToImage(),
                v2.ScaleJitter(
                    target_size=(512, 512),
                    scale_range=(384 / 512, 1.1),
                    antialias=True),
                v2.RandomPhotometricDistort(p=1),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomChoice([
                    v2.GaussianBlur(7, sigma=(0.1, 2.0)),
                    v2.RandomAdjustSharpness(2, p=0.5),
                ]),
                v2.RandomCrop(
                    size=(512, 512),
                    pad_if_needed=True,
                    fill={tv_tensors.Image: 127, tv_tensors.Mask: 0}),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])
            test_transforms = v2.Compose([
                v2.ToImage(),
                v2.Resize(
                    size=511,
                    max_size=512,
                    antialias=True),
                v2.CenterCrop(512),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])

        elif task_name == 'cocox640':
            train_transforms = v2.Compose([
                v2.ToImage(),
                v2.ScaleJitter(
                    target_size=(640, 640),
                    scale_range=(384 / 640, 1.1),
                    antialias=True),
                v2.RandomPhotometricDistort(p=1),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomChoice([
                    v2.GaussianBlur(7, sigma=(0.1, 2.0)),
                    v2.RandomAdjustSharpness(2, p=0.5),
                ]),
                v2.RandomCrop(
                    size=(640, 640),
                    pad_if_needed=True,
                    fill={tv_tensors.Image: 127, tv_tensors.Mask: 0}),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])
            test_transforms = v2.Compose([
                v2.ToImage(),
                v2.Resize(
                    size=639,
                    max_size=640,
                    antialias=True),
                v2.CenterCrop(640),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])

        else:
            raise ValueError(f'Unsupported the task `{task_name}`')

        return train_transforms, test_transforms
