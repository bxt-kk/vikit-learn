from typing import List, Mapping, Any, Dict
import math

from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from .component import ConvNormActive, InvertedResidual, CSENet
from .component import MobileNetFeatures, DinoFeatures, DEFAULT_NORM_LAYER
from .basic import Basic


class MixtureHead(nn.Module):

    def __init__(
            self,
            in_planes: int,
            num_layers: int,
        ):

        super().__init__()

        self.num_layers = num_layers

        project_dim = in_planes

        self.projects = nn.ModuleList()
        self.samples = nn.ModuleList()
        for t in range(num_layers):
            out_planes = in_planes * 2
            self.samples.append(InvertedResidual(
                in_planes, out_planes, 1, stride=2))
            self.projects.insert(0, nn.Conv2d(
                in_planes + project_dim, project_dim, 1))
            in_planes = out_planes
        self.project_final = nn.Conv2d(in_planes, project_dim, 1)

        self.norm_layer = DEFAULT_NORM_LAYER(project_dim)

    def forward(self, x:Tensor) -> Tensor:
        hs = [x]
        for t in range(self.num_layers):
            hs.append(self.samples[t](hs[-1]))
        p = self.project_final(hs.pop())
        for t in range(self.num_layers):
            p = F.interpolate(p, scale_factor=2, mode='bilinear')
            x = hs.pop()
            x = torch.cat([x, p], dim=1)
            p = p + self.projects[t](x)
        return self.norm_layer(p)


class TrimUnit(nn.Module):

    def __init__(
            self,
            in_planes:  int,
            out_planes: int,
            scan_range: int=4,
            dropout_p:  float=0,
        ):

        super().__init__()

        modules = []
        modules.append(CSENet(in_planes, out_planes))
        for r in range(scan_range):
            modules.append(InvertedResidual(
                out_planes, out_planes, 1, dilation=2**r, activation=None))
        modules.append(ConvNormActive(out_planes, out_planes, 1))
        if dropout_p > 0:
            modules.append(nn.Dropout(dropout_p))
        self.blocks = nn.Sequential(*modules)

    def forward(self, x:Tensor) -> Tensor:
        return self.blocks(x)


class TrimNetX(Basic):
    '''A light-weight and easy-to-train model base the mobilenetv3

    Args:
        num_scans: Number of the Trim-Units.
        scan_range: Range factor of the Trim-Unit convolution.
        backbone: Specify a basic model as a feature extraction module.
        backbone_pretrained: Whether to load backbone pretrained weights.
    '''

    def __init__(
            self,
            num_scans:           int,
            scan_range:          int=4,
            backbone:            str='mobilenet_v3_small',
            backbone_pretrained: bool=True,
        ):

        super().__init__()

        self.num_scans  = num_scans
        self.scan_range = scan_range
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

        self.mixture = MixtureHead(self.merged_dim, 2)

        self.project = ConvNormActive(
            self.merged_dim, self.merged_dim, 1, activation=None)

        self.trim_units = nn.ModuleList()
        for t in range(num_scans):
            in_planes = self.merged_dim
            if t > 0:
                in_planes = 2 * self.merged_dim
            sigma = (math.cos((t + 1) / num_scans * math.pi) + 1) / 4
            self.trim_units.append(TrimUnit(
                in_planes,
                self.merged_dim,
                scan_range=scan_range,
                dropout_p=sigma,
            ))

    def forward(self, x:Tensor) -> List[Tensor]:
        if not self._keep_features:
            f = self.features(x)
        else:
            with torch.no_grad():
                f = self.features(x)

        m = self.merge(f)
        # Lab
        m = self.mixture(m)
        # >>>
        h = self.trim_units[0](m)
        ht = [h]
        times = len(self.trim_units)
        for t in range(1, times):
            e = self.project(h)
            h = self.trim_units[t](torch.cat([m, e], dim=1))
            ht.append(h)
        return ht

    @classmethod
    def load_from_state(cls, state:Mapping[str, Any]) -> 'TrimNetX':
        hyps = state['hyperparameters']
        model = cls(
            num_scans           = hyps['num_scans'],
            scan_range          = hyps['scan_range'],
            backbone            = hyps['backbone'],
            backbone_pretrained = False,
        )
        model.load_state_dict(state['model'])
        return model

    def hyperparameters(self) -> Dict[str, Any]:
        return dict(
            num_scans  = self.num_scans,
            scan_range = self.scan_range,
            backbone   = self.backbone,
        )
