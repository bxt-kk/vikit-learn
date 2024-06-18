from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.mobilenetv3 import InvertedResidual
from torchvision.ops.misc import SqueezeExcitation

from .component import LinearBasicConvBD, CSENet, LocalSqueezeExcitation
from .basic import Basic


class TrimNetX(Basic):
    '''A light-weight and easy-to-train model base the mobilenetv3

    Args:
        dilation_depth: Depth of dilation module.
        dilation_range: The impact region of dilation convolution.
        backbone: Specify a basic model as a feature extraction module.
        backbone_pretrained: Whether to load backbone pretrained weights.
    '''

    def __init__(
            self,
            dilation_depth:      int=2,
            dilation_range:      int=4,
            backbone:            str='mobilenet_v3_small',
            backbone_pretrained: bool=True,
            pop_features_se:     bool=True,
        ):

        super().__init__()

        self.dilation_depth = dilation_depth
        self.dilation_range = dilation_range
        self.backbone       = backbone

        if backbone == 'mobilenet_v3_small':
            features = mobilenet_v3_small(
                weights=MobileNet_V3_Small_Weights.DEFAULT
                if backbone_pretrained else None,
            ).features
            if pop_features_se:
                features = self._pop_se_blocks(features)

            self.features_dim = 48 + 96
            self.merged_dim   = 160

            self.features_d = features[:9] # 48, 32, 32
            self.features_u = features[9:-1] # 96, 16, 16

        elif backbone == 'mobilenet_v3_large':
            features = mobilenet_v3_large(
                weights=MobileNet_V3_Large_Weights.DEFAULT
                if backbone_pretrained else None,
            ).features
            if pop_features_se:
                features = self._pop_se_blocks(features)

            self.features_dim = 112 + 160
            self.merged_dim   = 320

            self.features_d = features[:13] # 112, 32, 32
            self.features_u = features[13:-1] # 160, 16, 16

        self.merge = nn.Sequential(
            nn.Conv2d(self.features_dim, self.merged_dim, 1, bias=False),
            nn.BatchNorm2d(self.merged_dim),
        )

        self.cluster = nn.ModuleList()
        self.csenets = nn.ModuleList()
        for _ in range(dilation_depth):
            modules = []
            for r in range(dilation_range):
                modules.append(
                    LinearBasicConvBD(self.merged_dim, self.merged_dim, dilation=2**r))
            modules.append(nn.Sequential(
                nn.Conv2d(self.merged_dim, self.merged_dim, 1, bias=False),
                nn.BatchNorm2d(self.merged_dim),
                nn.Hardswish(inplace=True),
            ))
            self.cluster.append(nn.Sequential(*modules))
            self.csenets.append(CSENet(
                self.merged_dim * 2, self.merged_dim, kernel_size=3, shrink_factor=4))

    def forward(self, x:Tensor) -> Tensor:
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
        for csenet_i, cluster_i in zip(self.csenets, self.cluster):
            x = x + csenet_i(torch.cat([x, cluster_i(x)], dim=1))
        return x

    def _pop_se_blocks(self, features:nn.Module) -> nn.Module:
        for m in features:
            if not isinstance(m, InvertedResidual): continue
            block:nn.Sequential = m.block
            remove_ids = []
            for idx, child in block.named_children():
                if not isinstance(child, SqueezeExcitation): continue
                remove_ids.append(int(idx))
            for idx in remove_ids[::-1]:
                # block.pop(idx)
                block[idx] = LocalSqueezeExcitation.load_from_se_module(
                    block[idx], kernel_size=5)
        return features
