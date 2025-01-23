from typing import List, Any, Dict, Tuple

from torch import Tensor

import torch

from torchvision import tv_tensors
from torchvision.transforms import v2

from torchmetrics.classification import Precision, Recall

from PIL import Image

from .basic import Basic


class Classifier(Basic):

    def __init__(self, categories:List[str]):
        super().__init__()

        self.categories  = list(categories)
        self.num_classes = len(categories)
        self.precision_metric = Precision(
            task='multiclass',
            num_classes=self.num_classes,
            average='macro')
        self.recall_metric = Recall(
            task='multiclass',
            num_classes=self.num_classes,
            average='macro')

    def preprocess(
            self,
            image:      Image.Image,
            align_size: int | Tuple[int, int],
        ) -> Tensor:

        if isinstance(align_size, int):
            src_w, src_h = image.size
            scale = align_size / min(src_w, src_h)
            dst_w, dst_h = round(src_w * scale), round(src_h * scale)
            x1 = (dst_w - align_size) // 2
            y1 = (dst_h - align_size) // 2
            x2 = x1 + align_size
            y2 = y1 + align_size

            resized = image.resize((dst_w, dst_h), resample=Image.Resampling.BILINEAR)
            sampled = resized.crop((x1, y1, x2, y2))
        else:
            sampled = image.resize(align_size, resample=Image.Resampling.BILINEAR)

        return self._image2tensor(sampled).unsqueeze(dim=0)


    def classify(
            self,
            image:      Image.Image,
            top_k:      int=10,
            align_size: int | Tuple[int, int]=224,
        ) -> List[Dict[str, Any]]:
        assert not 'this is an empty func'

    def calc_loss(
            self,
            inputs:  Tensor,
            target:  Tensor,
            weights: Dict[str, float] | None=None,
            alpha:   float=0.25,
            gamma:   float=2.,
        ) -> Dict[str, Any]:
        assert not 'this is an empty func'

    def calc_score(
            self,
            inputs: Tensor,
            target: Tensor,
            thresh: float=0.5,
            eps:    float=1e-5,
        ) -> Dict[str, Any]:
        assert not 'this is an empty func'

    def update_metric(
            self,
            inputs: Tensor,
            target: Tensor,
            thresh: float=0.5,
        ):

        predict = inputs.argmax(dim=-1)
        self.precision_metric.update(predict, target)
        self.recall_metric.update(predict, target)

    def compute_metric(self) -> Dict[str, Any]:
        precision = self.precision_metric.compute()
        recall = self.recall_metric.compute()
        f1_score = (
            2 * precision * recall /
            torch.clamp_min(precision + recall, 1e-5))
        self.precision_metric.reset()
        self.recall_metric.reset()
        return dict(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
        )

    def collate_fn(
            self,
            batch: List[Any],
        ) -> Any:
        assert not 'this is an empty func'

    @classmethod
    def get_transforms(
            cls,
            task_name: str='default',
        ) -> Tuple[v2.Transform, v2.Transform]:

        train_transforms = None
        test_transforms  = None

        if task_name in ('default', 'imagenetx224'):
            aligned_size = 224
        elif task_name == 'imagenetx256':
            aligned_size = 256
        elif task_name == 'imagenetx384':
            aligned_size = 384
        elif task_name == 'imagenetx448':
            aligned_size = 448
        elif task_name == 'imagenetx512':
            aligned_size = 512
        elif task_name == 'imagenetx640':
            aligned_size = 640
        elif task_name == 'documentx224':
            aligned_size = 224
        else:
            raise ValueError(f'Unsupported the task `{task_name}`')

        if task_name.startswith('imagenet'):
            train_transforms = v2.Compose([
                v2.ToImage(),
                v2.ScaleJitter(
                    target_size=(aligned_size, aligned_size),
                    scale_range=(1., 2.),
                    antialias=True),
                v2.RandomPhotometricDistort(p=1),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomChoice([
                    v2.GaussianBlur(7, sigma=(0.1, 2.0)),
                    v2.RandomAdjustSharpness(2, p=0.5),
                    v2.RandomEqualize(p=0.5),
                ]),
                v2.RandomCrop(
                    size=(aligned_size, aligned_size),
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
                    size=aligned_size,
                    antialias=True),
                v2.CenterCrop(aligned_size),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])

        elif task_name.startswith('document'):
            train_transforms = v2.Compose([
                v2.ToImage(),
                v2.Resize(
                    size=(aligned_size, aligned_size),
                    antialias=True),
                v2.RandomPhotometricDistort(p=1),
                v2.RandomChoice([
                    v2.GaussianBlur(7, sigma=(0.1, 2.0)),
                    v2.RandomAdjustSharpness(2, p=0.5),
                    v2.RandomEqualize(p=0.5),
                ]),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])
            test_transforms = v2.Compose([
                v2.ToImage(),
                v2.Resize(
                    size=(aligned_size, aligned_size),
                    antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])

        return train_transforms, test_transforms
