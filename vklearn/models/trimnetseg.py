from typing import List, Any, Dict, Mapping
import math

from torch import Tensor
from torchvision.ops import (
    sigmoid_focal_loss,
)

import torch
import torch.nn as nn

from PIL import Image

from .segment import Segment
from .trimnetx import TrimNetX
from .component import ConvNormActive, UpSample


class TrimNetSeg(Segment):
    '''A light-weight and easy-to-train model for image segmentation

    Args:
        categories: Target categories.
        num_scans: Number of the Trim-Units.
        scan_range: Range factor of the Trim-Unit convolution.
        backbone: Specify a basic model as a feature extraction module.
        backbone_pretrained: Whether to load backbone pretrained weights.
    '''

    def __init__(
            self,
            categories:          List[str],
            num_scans:           int=3,
            scan_range:          int=4,
            backbone:            str='mobilenet_v3_small',
            backbone_pretrained: bool=True,
        ):
        super().__init__(categories)

        self.trimnetx = TrimNetX(
            num_scans, scan_range, backbone, backbone_pretrained)

        merged_dim = self.trimnetx.merged_dim
        # expanded_dim = merged_dim * 4

        # self.predict = nn.Sequential(
        #     ConvNormActive(merged_dim, expanded_dim, 1),
        #     nn.Conv2d(expanded_dim, self.num_classes, 1),
        # )
        self.predict = nn.Sequential(
            UpSample(merged_dim),
            ConvNormActive(merged_dim, merged_dim // 2, 1), # 80, 56
            UpSample(merged_dim // 2),
            ConvNormActive(merged_dim // 2, merged_dim // 4, 1), # 40, 112
            UpSample(merged_dim // 4),
            ConvNormActive(merged_dim // 4, merged_dim // 8, 1), # 20, 224
            UpSample(merged_dim // 8),
            ConvNormActive(merged_dim // 8, merged_dim // 16, 1), # 10, 448
            nn.Conv2d(merged_dim // 16, self.num_classes, 1),
        )

    def train_features(self, flag:bool):
        self.trimnetx.train_features(flag)

    def forward(self, x:Tensor) -> Tensor:
        hs = self.trimnetx(x)
        p = self.predict(hs[0])
        ps = [p]
        times = len(hs)
        for t in range(1, times):
            a = torch.sigmoid(p)
            p = self.predict(hs[t]) * a + p * (1 - a)
            ps.append(p)
        return torch.cat([p[..., None] for p in ps], dim=-1)

    @classmethod
    def load_from_state(cls, state:Mapping[str, Any]) -> 'TrimNetSeg':
        hyps = state['hyperparameters']
        model = cls(
            categories          = hyps['categories'],
            num_scans           = hyps['num_scans'],
            scan_range          = hyps['scan_range'],
            backbone            = hyps['backbone'],
            backbone_pretrained = False,
        )
        model.load_state_dict(state['model'])
        return model

    def hyperparameters(self) -> Dict[str, Any]:
        return dict(
            categories = self.categories,
            num_scans  = self.trimnetx.num_scans,
            scan_range = self.trimnetx.scan_range,
            backbone   = self.trimnetx.backbone,
        )

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
            alpha:   float=0.25,
            gamma:   float=2.,
        ) -> Dict[str, Any]:

        reduction = 'mean'

        times = inputs.shape[-1]
        F_sigma = lambda t: 1 - (math.cos((t + 1) / times * math.pi) + 1) * 0.5
        target = target.type_as(inputs)

        grand_sigma = 0.
        loss = 0.
        for t in range(times):
            sigma = F_sigma(t)
            grand_sigma += sigma
            loss = loss + sigmoid_focal_loss(
                inputs[..., t],
                target,
                alpha=alpha,
                gamma=gamma,
                reduction=reduction,
            ) * sigma
        loss = loss / grand_sigma

        return dict(
            loss=loss,
        )

    def calc_score(
            self,
            inputs: Tensor,
            target: Tensor,
            eps:    float=1e-5,
        ) -> Dict[str, Any]:

        predicts = torch.sigmoid(inputs[..., -1])
        distance = torch.abs(predicts - target).mean(dim=(2, 3)).mean()

        return dict(
            mad=distance,
        )

    def update_metric(
            self,
            inputs:      Tensor,
            target:      Tensor,
            conf_thresh: float=0.5,
        ):

        predicts = torch.sigmoid(inputs[..., -1]) > conf_thresh
        self.m_iou.update(predicts.to(torch.int), target.to(torch.int))
