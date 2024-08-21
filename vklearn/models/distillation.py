from typing import List, Any, Dict, Tuple

from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import tv_tensors
from torchvision.transforms import v2

from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError

from .basic import Basic


class Distillation(Basic):

    def __init__(
            self,
            teacher: nn.Module,
            student: nn.Module,
            project: nn.Module,
        ):

        super().__init__()

        self.teacher = teacher
        self.student = student
        self.project = project

        self.mse_metric = MeanSquaredError()
        self.mae_metric = MeanAbsoluteError()

    def align_inputs(self, x:Tensor) -> Tensor:
        return x

    def forward(self, x:Tensor) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            teacher_targ = self.teacher(self.align_inputs(x))
        student_pred = self.project(self.student(x))
        return student_pred, teacher_targ

    def calc_loss(
            self,
            inputs:  Tuple[Tensor, Tensor],
            target:  Tensor,
        ) -> Dict[str, Any]:

        teacher_targ, student_pred = inputs
        loss = F.mse_loss(student_pred, teacher_targ, reduction='mean')
        return dict(
            loss=loss,
        )

    def calc_score(
            self,
            inputs: Tuple[Tensor, Tensor],
            target: Tensor,
            eps:    float=1e-5,
        ) -> Dict[str, Any]:

        teacher_targ, student_pred = inputs
        cos_sim = F.cosine_similarity(student_pred, teacher_targ, dim=1).mean()
        return dict(
            cos_sim=cos_sim,
        )

    def update_metric(
            self,
            inputs: Tuple[Tensor, Tensor],
            target: Tensor,
        ):

        teacher_targ, student_pred = inputs
        self.mse_metric.update(student_pred, teacher_targ)
        self.mae_metric.update(student_pred, teacher_targ)

    def compute_metric(self) -> Dict[str, Any]:
        mse = self.mse_metric.compute()
        mae = self.mae_metric.compute()
        mss = 1 - mse / (1 + mse)
        self.mse_metric.reset()
        self.mae_metric.reset()
        return dict(
            mss=mss,
            mse=mse,
            mae=mae,
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

        if task_name in ('default', 'cocox512'):
            aligned_size = 512
        elif task_name == 'cocox384':
            aligned_size = 384
        elif task_name == 'cocox448':
            aligned_size = 448
        elif task_name == 'cocox640':
            aligned_size = 640
        else:
            raise ValueError(f'Unsupported the task `{task_name}`')

        train_transforms = v2.Compose([
            v2.ToImage(),
            v2.ScaleJitter(
                target_size=(aligned_size, aligned_size),
                scale_range=(min(0.9, 384 / aligned_size), 1.1),
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
            v2.SanitizeBoundingBoxes(min_size=3),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])
        test_transforms = v2.Compose([
            v2.ToImage(),
            v2.Resize(
                size=aligned_size - 1,
                max_size=aligned_size,
                antialias=True),
            v2.Pad(
                padding=aligned_size // 4,
                fill={tv_tensors.Image: 127, tv_tensors.Mask: 0}),
            v2.CenterCrop(aligned_size),
            v2.SanitizeBoundingBoxes(min_size=3),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

        return train_transforms, test_transforms
