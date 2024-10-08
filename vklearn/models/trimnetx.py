from typing import List, Mapping, Any, Dict, Tuple

from torch import Tensor

import torch
import torch.nn as nn

from .component import ConvNormActive
from .component import MobileNetFeatures, DinoFeatures
from .component import CBANet
from .basic import Basic


class TrimUnit(nn.Module):

    def __init__(
            self,
            in_planes:  int,
            out_planes: int,
            head_dim:   int,
            scan_range: int=4,
        ):

        super().__init__()

        assert out_planes % head_dim == 0
        groups = out_planes // head_dim
        dense_dim = out_planes // scan_range

        self.cbanet = CBANet(in_planes, out_planes)
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
        self.merge = ConvNormActive(dense_dim * scan_range, out_planes, 1)

    def forward(self, x:Tensor) -> Tensor:
        x = self.cbanet(x)
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
            num_scans:           int | None=None,
            scan_range:          int | None=None,
            backbone:            str | None=None,
            backbone_pretrained: bool | None=None,
        ):

        super().__init__()

        if num_scans is None:
            num_scans = 3
        if scan_range is None:
            scan_range = 4
        if backbone is None:
            backbone = 'mobilenet_v3_small'
        if backbone_pretrained is None:
            backbone_pretrained = True

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

        elif backbone == 'dinov2_vits14_h192':
            self.features     = DinoFeatures(backbone.rstrip('_h192'))
            self.features_dim = self.features.features_dim
            self.merged_dim   = 192

        else:
            raise ValueError(f'Unsupported backbone `{backbone}`')

        self.cell_size = self.features.cell_size

        self.projects = nn.ModuleList([
            ConvNormActive(self.merged_dim, self.merged_dim, 1, activation=None)
            for _ in range(num_scans - 1)])

        self.trim_units = nn.ModuleList()
        for t in range(num_scans):
            in_planes = self.features_dim
            if t > 0:
                in_planes = self.features_dim + self.merged_dim
            self.trim_units.append(TrimUnit(
                in_planes,
                self.merged_dim,
                head_dim=16,
                scan_range=scan_range,
            ))

    def forward(self, x:Tensor) -> Tuple[List[Tensor], Tensor]:
        if not self._keep_features:
            f = self.features(x)
        else:
            with torch.no_grad():
                f = self.features(x)

        h = self.trim_units[0](f)
        ht = [h]
        times = len(self.trim_units)
        for t in range(1, times):
            e = self.projects[t - 1](h)
            h = self.trim_units[t](torch.cat([f, e], dim=1))
            ht.append(h)
        return ht, f

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
