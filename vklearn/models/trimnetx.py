from typing import List, Mapping, Any, Dict, Tuple
import math

from torch import Tensor

import torch
import torch.nn as nn

from .component import ConvNormActive, ConvNormActiveRes, CSENet
from .component import MobileNetFeatures, DinoFeatures
from .basic import Basic


class TrimUnit(nn.Module):

    def __init__(
            self,
            in_planes:  int,
            out_planes: int,
            head_dim:   int,
            scan_range: int=4,
            dropout_p:  float=0.,
        ):

        super().__init__()

        assert out_planes % head_dim == 0
        groups = out_planes // head_dim

        modules = []
        modules.append(CSENet(in_planes, out_planes))
        for r in range(scan_range):
            modules.append(ConvNormActiveRes(
                out_planes,
                out_planes,
                dilation=2**r,
                groups=groups,
                norm_layer=None,
                activation=None,
            ))
        modules.append(ConvNormActive(out_planes, out_planes, 1))
        if dropout_p > 0:
            modules.append(nn.Dropout(dropout_p))
        self.blocks = nn.Sequential(*modules)

    # def train(self, mode:bool=True):
    #     super().train(mode)
    #     if isinstance(self.blocks[-1], nn.Dropout):
    #         self.blocks[-1].train()

    def forward(self, x:Tensor) -> Tensor:
        return self.blocks(x)


class TrimUnit2(nn.Module):

    def __init__(
            self,
            in_planes:  int,
            out_planes: int,
            head_dim:   int,
            scan_range: int=4,
            dropout_p:  float=0.,
        ):

        super().__init__()

        assert out_planes % head_dim == 0
        groups = out_planes // head_dim
        dense_dim = out_planes // scan_range

        self.csenet = CSENet(in_planes, out_planes)
        self.convs = nn.ModuleList()
        self.denses = nn.ModuleList()
        for r in range(scan_range):
            self.convs.append(ConvNormActive(
                out_planes,
                out_planes,
                dilation=2**r,
                groups=groups,
                norm_layer=None,
                activation=None,
            ))
            self.denses.append(ConvNormActive(out_planes, dense_dim, 1))
        modules = []
        modules.append(ConvNormActive(dense_dim * scan_range, out_planes, 1))
        if dropout_p > 0:
            modules.append(nn.Dropout(dropout_p))
        self.merge = nn.Sequential(*modules)

    # def train(self, mode:bool=True):
    #     super().train(mode)
    #     if isinstance(self.blocks[-1], nn.Dropout):
    #         self.blocks[-1].train()

    def forward(self, x:Tensor) -> Tensor:
        x = self.csenet(x)
        ds = []
        for conv, dense in zip(self.convs, self.denses):
            x = x + conv(x)
            ds.append(dense(x))
        m = self.merge(torch.cat(ds, dim=1))
        return m


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
            self.merged_dim   = 128

        elif backbone == 'mobilenet_v3_large':
            self.features = MobileNetFeatures(
                backbone, backbone_pretrained)
            self.features_dim = self.features.features_dim
            self.merged_dim   = 192

        elif backbone == 'mobilenet_v3_larges':
            self.features = MobileNetFeatures(
                backbone, backbone_pretrained)
            self.features_dim = self.features.features_dim
            self.merged_dim   = 192

        elif backbone == 'mobilenet_v2':
            self.features = MobileNetFeatures(
                backbone, backbone_pretrained)
            self.features_dim = self.features.features_dim
            self.merged_dim   = 192

        elif backbone == 'dinov2_vits14':
            self.features     = DinoFeatures(backbone)
            self.features_dim = self.features.features_dim
            self.merged_dim   = self.features_dim # 384

        else:
            raise ValueError(f'Unsupported backbone `{backbone}`')

        self.cell_size = self.features.cell_size

        # self.merge = ConvNormActive(
        #     self.features_dim, self.merged_dim, 1, activation=None)
        # self.merge = CSENet(self.features_dim, self.merged_dim)

        self.project = ConvNormActive(
            self.merged_dim, self.merged_dim, 1, activation=None)

        self.trim_units = nn.ModuleList()
        for t in range(num_scans):
            # in_planes = self.merged_dim
            in_planes = self.features_dim
            if t > 0:
                # in_planes = 2 * self.merged_dim
                in_planes = self.features_dim + self.merged_dim
            sigma = (math.cos((t + 1) / num_scans * math.pi) + 1) / 4
            self.trim_units.append(TrimUnit2(
                in_planes,
                self.merged_dim,
                head_dim=16,
                scan_range=scan_range,
                dropout_p=sigma,
            ))

    def forward(self, x:Tensor) -> Tuple[List[Tensor], Tensor]:
        if not self._keep_features:
            f = self.features(x)
        else:
            with torch.no_grad():
                f = self.features(x)

        # m = self.merge(f)
        m = f
        h = self.trim_units[0](m)
        ht = [h]
        times = len(self.trim_units)
        for t in range(1, times):
            e = self.project(h)
            h = self.trim_units[t](torch.cat([m, e], dim=1))
            ht.append(h)
        return ht, m

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
