from torch import Tensor
import torch
import torch.nn as nn

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
            num_waves:  int=3,
            wave_depth: int=4,
        ):

        super().__init__()

        self.num_waves  = num_waves
        self.wave_depth = wave_depth

        self.features = nn.Sequential(
            nn.PixelUnshuffle(2),
            ConvNormActive(12, 24, stride=2),
            InvertedResidual(24, 24, 4),
            ConvNormActive(24, 48, stride=2),
            InvertedResidual(48, 48, 4),
            ConvNormActive(48, 96, stride=2),
            InvertedResidual(96, 96, 4),
            InvertedResidual(96, 192, 4),
        ) # 192, 32, 32

        self.features_dim = 192

        self.cluster = nn.ModuleList()
        self.csenets = nn.ModuleList()
        for _ in range(num_waves):
            modules = []
            for r in range(wave_depth):
                modules.append(
                    InvertedResidual(
                        self.features_dim, self.features_dim, 2, dilation=2**r, heads=8, activation=None))
            modules.append(ConvNormActive(self.features_dim, self.features_dim, 1))
            self.cluster.append(nn.Sequential(*modules))
            self.csenets.append(LSENet(
                self.features_dim * 2, self.features_dim, kernel_size=3, shrink_factor=4))

    def forward(self, x:Tensor) -> Tensor:
        x = self.features(x)
        for csenet_i, cluster_i in zip(self.csenets, self.cluster):
            x = x + csenet_i(torch.cat([x, cluster_i(x)], dim=1))
        return x
