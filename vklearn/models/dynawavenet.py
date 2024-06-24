from typing import List

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from .basic import Basic
from .component import ConvNormActive, InvertedResidual, LSENet


class DynawaveNet(Basic):
    '''A light-weight and easy-to-train model

    Args:
        num_waves: Number of the global wave blocks.
        wave_depth: Depth of the wave block.
    '''

    def __init__(
            self,
            num_waves:  int=2,
            wave_depth: int=4,
        ):

        super().__init__()

        self.num_waves  = num_waves
        self.wave_depth = wave_depth

        features = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT
        ).features

        self.features_d = features[:9] # 48, 32, 32
        self.features_u = features[9:-1] # 96, 16, 16

        features_ddim = 48
        features_udim = 96

        self.features_dim = 160

        self.merge = nn.Sequential(
            nn.Conv2d(features_ddim + features_udim, self.features_dim, 1, bias=False),
            nn.BatchNorm2d(self.features_dim),
        )

        expand_ratio = 2
        expanded_dim = self.features_dim * expand_ratio

        self.cluster = nn.ModuleList()
        self.csenets = nn.ModuleList()
        self.normals = nn.ModuleList()
        for _ in range(num_waves):
            modules = []
            modules.append(InvertedResidual(
                self.features_dim, expanded_dim, expand_ratio, stride=2))
            for r in range(wave_depth):
                modules.append(
                    InvertedResidual(
                        expanded_dim, expanded_dim, 1, dilation=2**r, activation=None))
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(expanded_dim, expanded_dim, 3, 2, 1, output_padding=1, groups=expanded_dim, bias=False),
                nn.BatchNorm2d(expanded_dim),
                nn.GELU(),
                ConvNormActive(expanded_dim, self.features_dim, 1),
            ))
            self.cluster.append(nn.Sequential(*modules))
            self.csenets.append(LSENet(
                self.features_dim * 2, self.features_dim, kernel_size=3, shrink_factor=4))
            self.normals.append(nn.BatchNorm2d(self.features_dim))

    def forward(self, x:Tensor) -> List[Tensor]:
        # x = self.features(x)
        if not self._keep_features:
            fd = self.features_d(x)
            fu = self.features_u(fd)
        else:
            with torch.no_grad():
                fd = self.features_d(x)
                fu = self.features_u(fd)
        x = self.merge(torch.cat([
            fd,
            F.interpolate(fu, scale_factor=2, mode='bilinear'),
        ], dim=1))
        fs = [x]
        for normal_i, csenet_i, cluster_i in zip(self.normals, self.csenets, self.cluster):
            x = normal_i(x + csenet_i(torch.cat([x, cluster_i(x)], dim=1)))
            fs.append(x)
        return fs
