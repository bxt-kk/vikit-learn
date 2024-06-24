from typing import Callable

from torch import Tensor
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torchvision.ops.misc import SqueezeExcitation


# class BasicConvBD(nn.Sequential):
#
#     def __init__(
#             self,
#             in_planes:   int,
#             out_planes:  int,
#             kernel_size: int=3,
#             stride:      int | tuple[int, int]=1
#         ):
#
#         padding = (kernel_size - 1) // 2
#         super().__init__(
#             nn.Conv2d(in_planes, in_planes, kernel_size, stride, padding, groups=in_planes, bias=False),
#             nn.BatchNorm2d(in_planes),
#             nn.Hardswish(inplace=True),
#             nn.Conv2d(in_planes, out_planes, 1, bias=False),
#             nn.BatchNorm2d(out_planes),
#             nn.Hardswish(inplace=True))
#
#
# class LinearBasicConvBD(nn.Module):
#
#     def __init__(
#             self,
#             in_planes:   int,
#             out_planes:  int,
#             kernel_size: int=3,
#             dilation:    int=1,
#             stride:      int | tuple[int, int]=1
#         ):
#
#         super().__init__()
#
#         padding = (kernel_size + 2 * (dilation - 1) - 1) // 2
#         self.layers = nn.Sequential(
#             nn.Conv2d(
#                 in_planes, in_planes, kernel_size, stride, padding,
#                 dilation=dilation, groups=in_planes, bias=False),
#             nn.BatchNorm2d(in_planes),
#             nn.Conv2d(in_planes, out_planes, 1, bias=False),
#             nn.BatchNorm2d(out_planes))
#
#         self.use_res_connect = in_planes == out_planes
#
#     def forward(self, x:Tensor) -> Tensor:
#         result = self.layers(x)
#         if self.use_res_connect:
#             result = result + x
#         return result
#
#
# class LinearBasicConvDBD(nn.Module):
#
#     def __init__(
#             self,
#             in_planes:   int,
#             expand_rate: int,
#             kernel_size: int=3,
#             dilation:    int=1,
#             stride:      int | tuple[int, int]=1
#         ):
#
#         super().__init__()
#
#         padding = (kernel_size + 2 * (dilation - 1) - 1) // 2
#         expanded_dim = in_planes * expand_rate
#         self.layers = nn.Sequential(
#             nn.Conv2d(in_planes, expanded_dim, 1, bias=False),
#             nn.BatchNorm2d(expanded_dim),
#             nn.Conv2d(
#                 expanded_dim, expanded_dim, kernel_size, stride, padding,
#                 dilation=dilation, groups=expanded_dim, bias=False),
#             nn.BatchNorm2d(expanded_dim),
#             nn.Conv2d(expanded_dim, in_planes, 1, bias=False),
#             nn.BatchNorm2d(in_planes),
#         )
#
#     def forward(self, x:Tensor) -> Tensor:
#         return x + self.layers(x)
#
#
# class BasicConvDB(nn.Sequential):
#
#     def __init__(
#             self,
#             in_planes:   int,
#             out_planes:  int,
#             kernel_size: int=3,
#             stride:      int | tuple[int, int]=1
#         ):
#
#         padding = (kernel_size - 1) // 2
#         super().__init__(
#             nn.Conv2d(in_planes, out_planes, 1, bias=False),
#             nn.BatchNorm2d(out_planes),
#             nn.Hardswish(inplace=True),
#             nn.Conv2d(out_planes, out_planes, kernel_size, stride, padding, groups=out_planes, bias=False),
#             nn.BatchNorm2d(out_planes),
#             nn.Hardswish(inplace=True))
#
#
# class LinearBasicConvDB(nn.Sequential):
#
#     def __init__(
#             self,
#             in_planes:   int,
#             out_planes:  int,
#             kernel_size: int=3,
#             dilation:    int=1,
#             stride:      int | tuple[int, int]=1
#         ):
#
#         padding = (kernel_size + 2 * (dilation - 1) - 1) // 2
#         super().__init__(
#             nn.Conv2d(in_planes, out_planes, 1, bias=False),
#             nn.BatchNorm2d(out_planes),
#             nn.Conv2d(out_planes, out_planes, kernel_size, stride, padding,
#                 dilation=dilation, groups=out_planes, bias=False),
#             nn.BatchNorm2d(out_planes))
#
#
# class PixelShuffleSample(nn.Module):
#
#     def __init__(
#             self,
#             in_planes:  int,
#         ):
#
#         super().__init__()
#
#         assert in_planes % 4 == 0
#         self.block = nn.Sequential(
#             nn.Conv2d(in_planes, in_planes, 1, bias=False),
#             nn.PixelShuffle(2),
#             nn.BatchNorm2d(in_planes // 4),
#         )
#
#     def forward(self, x:Tensor) -> Tensor:
#         return torch.cat([
#             self.block(x),
#             F.interpolate(x, scale_factor=2, mode='bilinear')
#             ], dim=1)


class LocalSqueezeExcitation(nn.Module):

    def __init__(
            self,
            input_channels:   int,
            squeeze_channels: int,
            kernel_size:      int=3,
        ):

        super().__init__()
        padding = (kernel_size - 1) // 2
        self.avgpool = nn.AvgPool2d(kernel_size, 1, padding=padding)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = nn.ReLU(inplace=True)
        self.scale_activation = nn.Hardsigmoid(inplace=True)

    @classmethod
    def load_from_se_module(
            cls,
            se_module:   SqueezeExcitation,
            kernel_size: int=3,
        ) -> 'LocalSqueezeExcitation':

        squeeze_channels, input_channels, _, _ = se_module.fc1.weight.shape
        lse_module = cls(input_channels, squeeze_channels, kernel_size)
        lse_module.fc1.load_state_dict(se_module.fc1.state_dict())
        lse_module.fc2.load_state_dict(se_module.fc2.state_dict())
        return lse_module

    def _scale(self, x:Tensor) -> Tensor:
        scale = self.fc1(x)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, x:Tensor) -> Tensor:
        scale = self._scale(x)
        return scale * x


class ConvNormActive(nn.Sequential):

    def __init__(
            self,
            in_planes:   int,
            out_planes:  int,
            kernel_size: int=3,
            stride:      int | tuple[int, int]=1,
            dilation:    int=1,
            groups:      int=1,
            norm_layer:  Callable[..., nn.Module] | None=nn.BatchNorm2d,
            activation:  Callable[..., nn.Module] | None=nn.GELU,
        ):

        padding = (kernel_size + 2 * (dilation - 1) - 1) // 2
        layers = [nn.Conv2d(
            in_planes, out_planes, kernel_size, stride, padding, dilation, groups=groups),
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_planes))
        if activation is not None:
            layers.append(activation())
        super().__init__(*layers)


class InvertedResidual(nn.Module):

    def __init__(
            self,
            in_planes:       int,
            out_planes:      int,
            expand_ratio:    int,
            kernel_size:     int=3,
            stride:          int | tuple[int, int]=1,
            dilation:        int=1,
            heads:           int=1,
            activation:      Callable[..., nn.Module] | None=nn.GELU,
            use_res_connect: bool=True,
        ):

        super().__init__()

        layers = []
        expanded_dim = in_planes * expand_ratio
        if expand_ratio != 1:
            layers.append(ConvNormActive(
                in_planes, expanded_dim, kernel_size=1, groups=heads, activation=activation))
        layers.extend([
            ConvNormActive(
                expanded_dim, expanded_dim,
                stride=stride, dilation=dilation, groups=expanded_dim, activation=activation),
            ConvNormActive(
                expanded_dim, out_planes, kernel_size=1, groups=heads, activation=None),
        ])
        self.blocks = nn.Sequential(*layers)
        self.use_res_connect = (
            use_res_connect and
            (in_planes == out_planes) and
            (stride == 1))

    def forward(self, x:Tensor) -> Tensor:
        out = self.blocks(x)
        if self.use_res_connect:
            out += x
        return out


class UpSample(nn.Sequential):

    def __init__(
            self,
            in_planes:  int,
            out_planes: int,
            activation: Callable[..., nn.Module] | None=nn.GELU,
        ):

        super().__init__(
            nn.ConvTranspose2d(in_planes, in_planes, 3, 2, 1, output_padding=1, groups=in_planes, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.GELU(),
        )


class CSENet(nn.Module):

    def __init__(
            self,
            in_planes:     int,
            out_planes:    int,
            kernel_size:   int=3,
            shrink_factor: int=4,
        ):

        super().__init__()

        shrink_dim = in_planes // shrink_factor
        self.fusion = nn.Sequential(
            ConvNormActive(
                in_planes, shrink_dim, 1),
            ConvNormActive(
                shrink_dim, shrink_dim, 3, groups=shrink_dim),
            ConvNormActive(
                shrink_dim, in_planes, 1, norm_layer=None, activation=nn.Sigmoid),
        )
        self.project = ConvNormActive(
            in_planes, out_planes, 1, norm_layer=nn.BatchNorm2d, activation=None)

    def forward(self, x:Tensor) -> Tensor:
        return self.project(x * self.fusion(x))


class DetPredictor(nn.Module):

    def __init__(
            self,
            in_planes:     int,
            hidden_planes: int,
            num_anchors:   int,
            bbox_dim:      int,
            num_classes:   int,
            dropout:       float,
            dropout_bbox:  float=0.,
        ):

        super().__init__()

        self.predict_bbox = nn.Sequential(
            ConvNormActive(in_planes, in_planes, kernel_size=1),
            nn.Dropout(p=dropout_bbox, inplace=True),
            nn.Conv2d(in_planes, num_anchors * bbox_dim, kernel_size=1),
        )

        self.predict_clss = nn.Sequential(
            ConvNormActive(in_planes, hidden_planes, kernel_size=1),
            nn.Dropout(p=dropout, inplace=True),
            nn.Conv2d(hidden_planes, num_anchors * num_classes, kernel_size=1),
        )

        self.num_anchors = num_anchors

    def forward(self, x:Tensor) -> Tensor:
        bs, _, ny, nx = x.shape
        p_bbox = self.predict_bbox(x).view(bs, self.num_anchors, -1, ny, nx)
        p_clss = self.predict_clss(x).view(bs, self.num_anchors, -1, ny, nx)
        return torch.cat([p_bbox, p_clss], dim=2).view(bs, -1, ny, nx)
