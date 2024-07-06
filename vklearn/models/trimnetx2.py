from typing import List

from torch import Tensor

import torch
import torch.nn as nn

from .component import ConvNormActive, InvertedResidual, CSENet
from .component import MobileNetFeatures, DinoFeatures
from .basic import Basic


class TrimUnit(nn.Module):

    def __init__(
            self,
            in_planes:  int,
            out_planes: int,
            wave_depth: int=4,
        ):

        super().__init__()

        modules = []
        modules.append(CSENet(in_planes, out_planes))
        for r in range(wave_depth):
            modules.append(InvertedResidual(
                out_planes, out_planes, 1, dilation=2**r, activation=None))
        modules.append(ConvNormActive(out_planes, out_planes, 1))
        self.blocks = nn.Sequential(*modules)

    def forward(self, x:Tensor) -> Tensor:
        return self.blocks(x)


class TrimNetX(Basic):
    '''A light-weight and easy-to-train model base the mobilenetv3

    Args:
        num_waves: Number of the global wave blocks.
        wave_depth: Depth of the wave block.
        backbone: Specify a basic model as a feature extraction module.
        backbone_pretrained: Whether to load backbone pretrained weights.
    '''

    def __init__(
            self,
            num_waves:           int,
            wave_depth:          int=4,
            backbone:            str='mobilenet_v3_small',
            backbone_pretrained: bool=True,
        ):

        super().__init__()

        self.num_waves  = num_waves
        self.wave_depth = wave_depth
        self.backbone   = backbone

        if backbone == 'mobilenet_v3_small':
            self.features = MobileNetFeatures(
                backbone, backbone_pretrained)
            self.features_dim = self.features.features_dim
            self.merged_dim   = 160

        elif backbone == 'mobilenet_v3_large':
            self.features = MobileNetFeatures(
                backbone, backbone_pretrained)
            self.features_dim = self.features.features_dim
            self.merged_dim   = 320

        elif backbone == 'dinov2_vits14':
            self.features     = DinoFeatures(backbone)
            self.features_dim = self.features.features_dim
            self.merged_dim   = self.features_dim

        else:
            raise ValueError(f'Unsupported backbone `{backbone}`')

        self.cell_size = self.features.cell_size

        self.merge = ConvNormActive(
            self.features_dim, self.merged_dim, 1, activation=None)

        self.trim_units = nn.ModuleList()
        for wave_id in range(num_waves):
            in_planes = self.merged_dim
            if wave_id > 0:
                in_planes = 2 * self.merged_dim
            self.trim_units.append(TrimUnit(
                in_planes, self.merged_dim, wave_depth=wave_depth))

    def forward(self, x:Tensor) -> List[Tensor]:
        if not self._keep_features:
            f = self.features(x)
        else:
            with torch.no_grad():
                f = self.features(x)

        m = self.merge(f)
        h = self.trim_units[0](m)
        ht = [h]
        times = len(self.trim_units)
        for t in range(1, times):
            h = self.trim_units[t](torch.cat([m, h], dim=1))
            ht.append(h)
        return ht
