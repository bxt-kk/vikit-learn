from typing import Callable, List, Tuple

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops.misc import SqueezeExcitation
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights

import timm


class LayerNorm2d(nn.GroupNorm):

    def __init__(self, num_channels:int):
        super().__init__(1, num_channels)


class LayerNorm2dChannel(nn.Module):
    def __init__(
            self,
            dim: int,
            eps: float=1e-5,
        ):

        super().__init__()

        self.weight = nn.Parameter(torch.ones(dim), requires_grad=True)
        self.bias   = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.shape  = (dim, )
        self.eps    = eps

    def forward(self, x:Tensor) -> Tensor:
        return F.layer_norm(
            x.transpose(1, 3),
            normalized_shape=self.shape,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps).transpose(1, 3)


DEFAULT_NORM_LAYER = nn.BatchNorm2d # LayerNorm2d
DEFAULT_ACTIVATION = nn.Hardswish
DEFAULT_SIGMOID    = nn.Hardsigmoid


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
            kernel_size: int | Tuple[int, int]=3,
            stride:      int | Tuple[int, int]=1,
            dilation:    int | Tuple[int, int]=1,
            groups:      int=1,
            norm_layer:  Callable[..., nn.Module] | None=DEFAULT_NORM_LAYER,
            activation:  Callable[..., nn.Module] | None=DEFAULT_ACTIVATION,
        ):

        if isinstance(kernel_size, int):
            kernel_size = kernel_size, kernel_size
        if isinstance(dilation, int):
            dilation = dilation, dilation
        padding = tuple(
            (ks + 2 * (dl - 1) - 1) // 2 for ks, dl in zip(kernel_size, dilation))
        layers = [nn.Conv2d(
            in_planes, out_planes, kernel_size, stride, padding, dilation, groups=groups),
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_planes))
        if activation is not None:
            layers.append(activation())
        super().__init__(*layers)


class ConvNormActiveRes(ConvNormActive):

    def forward(self, x:Tensor) -> Tensor:
        return super().forward(x) + x


class MultiKernelConvNormActive(nn.Module):

    def __init__(
            self,
            in_planes:   int,
            kernel_size: int | List[int]=3,
            stride:      int | Tuple[int, int]=1,
            dilation:    int=1,
            norm_layer:  Callable[..., nn.Module] | None=DEFAULT_NORM_LAYER,
            activation:  Callable[..., nn.Module] | None=DEFAULT_ACTIVATION,
        ):

        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]

        self.kernel_groups = len(kernel_size)
        assert in_planes % self.kernel_groups == 0

        part_dim = in_planes // self.kernel_groups
        self.convs = nn.ModuleList([
            ConvNormActive(
                part_dim,
                part_dim,
                kernel_size=ksize,
                groups=part_dim,
                norm_layer=None,
                activation=None,
            )
            for ksize in kernel_size])

        self.normal_active = nn.Sequential(
            norm_layer(in_planes),
            activation(),
        )

    def forward(self, x:Tensor) -> Tensor:
        bs, _, ny, nx = x.shape
        x = x.reshape(bs, self.kernel_groups, -1, ny, nx)
        x = torch.cat([
            self.convs[t](x[:, t])
            for t in range(self.kernel_groups)
        ], dim=1)
        return self.normal_active(x)


class InvertedResidual(nn.Module):

    def __init__(
            self,
            in_planes:       int,
            out_planes:      int,
            expand_ratio:    int,
            kernel_size:     int=3,
            stride:          int | Tuple[int, int]=1,
            dilation:        int=1,
            norm_layer:      Callable[..., nn.Module] | None=DEFAULT_NORM_LAYER,
            activation:      Callable[..., nn.Module] | None=DEFAULT_ACTIVATION,
            use_sse:         bool=False,
            use_res_connect: bool=True,
        ):

        super().__init__()

        layers = []
        expanded_dim = in_planes * expand_ratio
        if expand_ratio != 1:
            layers.append(ConvNormActive(
                in_planes,
                expanded_dim,
                kernel_size=1,
                norm_layer=norm_layer,
                activation=activation))
        layers.extend([
            ConvNormActive(
                expanded_dim,
                expanded_dim,
                stride=stride,
                dilation=dilation,
                groups=expanded_dim,
                norm_layer=norm_layer,
                activation=activation),
            ConvNormActive(
                expanded_dim,
                out_planes,
                kernel_size=1,
                norm_layer=norm_layer,
                activation=None),
        ])

        if use_sse:
            layers.insert(
                -1, SSELayer(expanded_dim, max(expanded_dim // 4, 8)))

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
            norm_layer: Callable[..., nn.Module] | None=DEFAULT_NORM_LAYER,
            activation: Callable[..., nn.Module] | None=DEFAULT_ACTIVATION,
        ):

        super().__init__(
            nn.ConvTranspose2d(in_planes, in_planes, 3, 2, 1, output_padding=1, groups=in_planes, bias=False),
            norm_layer(in_planes),
            activation(),
        )


# class PoolWithPosCode(nn.Module):
#
#     def __init__(
#             self,
#             stride:    int,
#             scale_dim: int=1,
#         ):
#         super().__init__()
#
#         self.stride = stride
#         self.scale = 1 / scale_dim**0.5
#
#     def forward(self, x:Tensor) -> Tensor:
#         in_planes = x.shape[1]
#
#         # l = (x * x).sum(dim=1, keepdim=True)
#         l = (torch.square(x.detach())).sum(dim=1, keepdim=True)
#         u = F.pixel_unshuffle(l, self.stride)
#         I = u.argmax(dim=1, keepdim=True)
#         cr = (I // self.stride).type_as(x) / (0.5 * (self.stride - 1)) * self.scale - self.scale
#         cc = (I % self.stride).type_as(x) / (0.5 * (self.stride - 1)) * self.scale - self.scale
#
#         bs, _, ny, nx = u.shape
#         x = F.pixel_unshuffle(x, self.stride)
#         x = x.reshape(bs, in_planes, -1, ny, nx)
#         J = I.unsqueeze(1).expand(-1, in_planes, -1, -1, -1)
#         x = x.gather(dim=2, index=J).squeeze(2)
#         return torch.cat([cr, cc, x], dim=1)


class PoolWithPosCode(nn.Module):

    def __init__(
            self,
            in_planes: int,
            stride:    int,
        ):

        super().__init__()

        assert stride > 1

        self.pos_code = nn.Sequential(
            nn.Conv2d(in_planes, 1, 1),
            DEFAULT_ACTIVATION(inplace=False),
            nn.Conv2d(1, 2, kernel_size=stride, stride=stride),
            DEFAULT_ACTIVATION(inplace=False),
        )

        self.stride = stride

    def forward(self, x:Tensor) -> Tensor:
        return torch.cat([
            self.pos_code(x),
            F.max_pool2d(x, self.stride),
        ], dim=1)


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
                shrink_dim, in_planes, 1, norm_layer=None, activation=DEFAULT_SIGMOID),
        )
        self.project = ConvNormActive(
            in_planes, out_planes, 1, activation=None)

    def forward(self, x:Tensor) -> Tensor:
        return self.project(x * self.fusion(x))


class SSELayer(nn.Module):

    def __init__(
            self,
            in_channels: int,
            sq_channels: int,
        ):

        super().__init__()

        self._scale = nn.Sequential(
            nn.Conv2d(in_channels, sq_channels, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(sq_channels, in_channels, 1),
            nn.Hardsigmoid(inplace=False),
        )

    def forward(self, x:Tensor) -> Tensor:
        return x * self._scale(x)


class ChannelAttention(nn.Module):

    def __init__(
            self,
            in_planes:     int,
            shrink_factor: int=4,
        ):

        super().__init__()

        shrink_dim = in_planes // shrink_factor
        self.dense = nn.Sequential(
            nn.Conv2d(in_planes, shrink_dim, 1),
            DEFAULT_ACTIVATION(inplace=False),
            nn.Conv2d(shrink_dim, in_planes, 1),
            DEFAULT_SIGMOID(inplace=False),
        )

    def forward(self, x:Tensor) -> Tensor:
        f = F.adaptive_avg_pool2d(x, 1)
        return self.dense(f) * x


class SpatialAttention(nn.Module):

    def __init__(
            self,
            in_planes:   int,
            kernel_size: int=7,
        ):

        super().__init__()

        self.fc = ConvNormActive(in_planes, 1, 1, norm_layer=None)
        self.dense = ConvNormActive(
            2, 1, kernel_size, norm_layer=None, activation=DEFAULT_SIGMOID)

    def forward(self, x:Tensor) -> Tensor:
        f = torch.cat([
            x.mean(dim=1, keepdim=True),
            self.fc(x),
        ], dim=1)
        return self.dense(f) * x


class CBANet(nn.Module):

    def __init__(
            self,
            in_planes:     int,
            out_planes:    int,
            kernel_size:   int=7,
            shrink_factor: int=4,
            norm_layer:    Callable[..., nn.Module]=DEFAULT_NORM_LAYER,
        ):

        super().__init__()

        self.channel_attention = ChannelAttention(in_planes, shrink_factor)
        self.spatial_attention = SpatialAttention(in_planes, kernel_size)
        self.project = ConvNormActive(
            in_planes, out_planes, 1, norm_layer=norm_layer, activation=None)

    def forward(self, x:Tensor) -> Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return self.project(x)


class SqueezeAttention(nn.Module):

    def __init__(
            self,
            in_channels:int,
            in_rows:int,
        ):

        super().__init__()

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_rows, kernel_size=(in_rows, 1)),
            nn.Softmax(dim=1))

    def forward(self, x:Tensor) -> Tensor:
        # x shape: n, c, h, w
        m = self.attention(x).transpose(1, 2) # n, h, 1, w -> n, 1, h, w
        n, _, _, w = x.shape
        x = (x * m).reshape(n, -1, 1, w) # n, c, 1, w
        return x


class DetPredictor(nn.Module):

    def __init__(
            self,
            features_dim: int,
            in_planes:    int,
            num_anchors:  int,
            bbox_dim:     int,
            clss_dim:     int,
            dropout_p:    float,
        ):

        super().__init__()

        self.num_anchors = num_anchors

        # Lab code <<<
        # self.conf_predict = nn.Sequential(
        #     ConvNormActive(in_planes, in_planes, 1),
        #     ConvNormActive(in_planes, in_planes, 3, groups=in_planes),
        #     nn.Conv2d(in_planes, num_anchors, kernel_size=1))
        conf_hidden = max(in_planes // num_anchors * num_anchors, num_anchors)
        self.conf_predict = nn.Sequential(
            ConvNormActive(in_planes, conf_hidden, 1),
            MultiKernelConvNormActive(
                conf_hidden, [3 + t * 2 for t in range(num_anchors)]),
            ConvNormActive(conf_hidden, conf_hidden, 1),
            MultiKernelConvNormActive(
                conf_hidden, [3 + t * 2 for t in range(num_anchors)]),
            nn.Conv2d(conf_hidden, num_anchors, kernel_size=1),
        )
        # >>>

        ex_bbox_dims = bbox_dim * num_anchors
        self.bbox_predict = nn.Sequential(
            ConvNormActive(in_planes, in_planes, 1),
            ConvNormActive(in_planes, in_planes, 3, groups=in_planes),
            # nn.Conv2d(in_planes, ex_bbox_dims, kernel_size=1),
            ConvNormActive(in_planes, ex_bbox_dims, 1, norm_layer=None),
            nn.Conv2d(ex_bbox_dims, ex_bbox_dims, kernel_size=1, groups=num_anchors))

        clss_hidden = in_planes * num_anchors

        self.expansion = nn.Sequential(
            ConvNormActive(in_planes + features_dim, conf_hidden, 1),
            MultiKernelConvNormActive(
                conf_hidden, [3 + t * 2 for t in range(num_anchors)]),
            ConvNormActive(conf_hidden, clss_hidden, 1),
            MultiKernelConvNormActive(
                clss_hidden, [3 + t * 2 for t in range(num_anchors)]),
        )

        self.dropout = nn.Dropout(dropout_p, inplace=False)

        self.clss_predict = nn.Conv2d(
            clss_hidden, clss_dim * num_anchors, kernel_size=1, groups=num_anchors)

    def forward(self, x:Tensor, features:Tensor) -> Tensor:
        bs, _, ny, nx = x.shape

        p_conf = self.conf_predict(x)
        p_bbox = self.bbox_predict(x)

        e = self.expansion(torch.cat([x, features], dim=1)) # .reshape(bs * self.num_anchors, -1, ny, nx)
        e = self.dropout(e) # .reshape(bs, -1, ny, nx)
        p_clss = self.clss_predict(e)

        return torch.cat([
            part.reshape(bs, self.num_anchors, -1, ny, nx).permute(0, 1, 3, 4, 2)
            for part in (p_conf, p_bbox, p_clss)
        ], dim=-1)


class SegPredictor(nn.Module):

    def __init__(
            self,
            in_planes:   int,
            num_classes: int,
            num_scans:  int,
        ):

        super().__init__()

        self.embeded_dim = int((num_classes + 2)**0.5)
        self.decoder = nn.Conv2d(self.embeded_dim, num_classes, 1)
        self.upsamples = nn.ModuleDict()
        self.predicts = nn.ModuleList()
        for t in range(num_scans):
            hidden_dim = in_planes
            out_planes = max(hidden_dim // 2, self.embeded_dim)
            if t > 0:
                self.upsamples[f'{t}'] = nn.Sequential(
                    UpSample(hidden_dim),
                    ConvNormActive(hidden_dim, out_planes, 1),
                )
                hidden_dim = out_planes + self.embeded_dim
                out_planes = max(out_planes // 2, self.embeded_dim)
            for k in range(t - 1):
                self.upsamples[f'{t}_{k}'] = nn.Sequential(
                    UpSample(hidden_dim),
                    ConvNormActive(hidden_dim, out_planes, 1),
                )
                hidden_dim = out_planes + self.embeded_dim
                out_planes = max(out_planes // 2, self.embeded_dim)
            self.predicts.append(nn.Sequential(
                UpSample(hidden_dim),
                ConvNormActive(hidden_dim, out_planes, 1),
                ConvNormActive(out_planes, out_planes, groups=out_planes),
                ConvNormActive(out_planes, self.embeded_dim, kernel_size=1),
            ))

    def forward(self, hs:List[Tensor]) -> List[Tensor]:
        pt = self.predicts[0](hs[0])
        ps = [pt]
        times = len(hs)
        for t in range(1, times):
            u = self.upsamples[f'{t}'](hs[t])
            for k in range(t - 1):
                u = self.upsamples[f'{t}_{k}'](torch.cat([
                    u, ps[k]], dim=1))
            pt = (
                self.predicts[t](torch.cat([u, pt], dim=1)) +
                F.interpolate(pt, scale_factor=2, mode='bilinear')
            )
            ps.append(pt)
        return ps


class MobileNetFeatures(nn.Module):

    def __init__(self, arch:str, pretrained:bool):

        super().__init__()

        if arch == 'mobilenet_v3_small':
            features = mobilenet_v3_small(
                weights=MobileNet_V3_Small_Weights.DEFAULT
                if pretrained else None,
            ).features

            layer_dims = (16, 16, 24, 48, 96)

            self.layers = nn.ModuleList([
                features[0], # 16, 256, 256
                features[1], # 16, 128, 128
                features[2:4], # 24, 64, 64
                features[4:8], # 48, 32, 32
                features[8:-1], # 96, 16, 16
            ])

        elif arch == 'mobilenet_v3_large':
            features = mobilenet_v3_large(
                weights=MobileNet_V3_Large_Weights.DEFAULT
                if pretrained else None,
            ).features

            layer_dims = (16, 24, 40, 112, 160)

            self.layers = nn.ModuleList([
                features[:2], # 16, 256, 256
                features[2:4], # 24, 128, 128
                features[4:7], # 40, 64, 64
                features[7:12], # 112, 32, 32
                features[12:-1], # 160, 16, 16
            ])

        elif arch == 'mobilenet_v3_large_dl':
            weights_state = deeplabv3_mobilenet_v3_large(
                weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
            ).backbone.state_dict()
            features = mobilenet_v3_large().features
            features.load_state_dict(weights_state)

            layer_dims = (16, 24, 40, 112, 160)

            self.layers = nn.ModuleList([
                features[:2], # 16, 256, 256
                features[2:4], # 24, 128, 128
                features[4:7], # 40, 64, 64
                features[7:12], # 112, 32, 32
                features[12:-1], # 160, 16, 16
            ])

        elif arch == 'mobilenet_v2':
            features = mobilenet_v2(
                weights=MobileNet_V2_Weights.DEFAULT
                if pretrained else None,
            ).features

            layer_dims = (16, 24, 32, 96, 160)

            self.layers = nn.ModuleList([
                features[:2], # 16, 256, 256
                features[2:4], # 24, 128, 128
                features[4:7], # 32, 64, 64
                features[7:12], # 96, 32, 32
                features[12:-2], # 160, 16, 16
            ])

        elif arch.startswith('mobilenet_v4_'):
            tag = arch.lstrip('mobilenet_v4_')

            layer_dims = (32, 32, 64, 96, 128)
            if 'medium' in tag:
                layer_dims = (32, 48, 80, 160, 256)
            elif 'large' in tag:
                layer_dims = (24, 48, 96, 192, 512)

            backbone = timm.create_model(
                'mobilenetv4_' + tag, pretrained=pretrained)

            self.layers = nn.ModuleList([
                nn.Sequential(backbone.conv_stem, backbone.bn1), # 256, 256
                backbone.blocks[0], # 128, 128
                backbone.blocks[1], # 64, 64
                backbone.blocks[2], # 32, 32
                backbone.blocks[3], # 16, 16
            ])

        else:
            raise ValueError(f'Unsupported arch `{arch}`')

        self.features_dim = sum(layer_dims) + 2 * 3
        self.cell_size    = 16

        # self.pools2 = PoolWithPosCode(stride=2, scale_dim=self.features_dim)
        # self.pools4 = PoolWithPosCode(stride=4, scale_dim=self.features_dim)
        # self.pools8 = PoolWithPosCode(stride=8, scale_dim=self.features_dim)

        self.pools2 = PoolWithPosCode(layer_dims[2], stride=2)
        self.pools4 = PoolWithPosCode(layer_dims[1], stride=4)
        self.pools8 = PoolWithPosCode(layer_dims[0], stride=8)

    def forward(self, x:Tensor) -> Tensor:
        f0 = self.layers[0](x)
        f1 = self.layers[1](f0)
        f2 = self.layers[2](f1)
        f3 = self.layers[3](f2)
        f4 = self.layers[4](f3)

        return torch.cat([
            self.pools8(f0),
            self.pools4(f1),
            self.pools2(f2),
            f3,
            F.interpolate(f4, scale_factor=2, mode='nearest'),
        ], dim=1)


class DinoFeatures(nn.Module):

    def __init__(self, arch:str):

        super().__init__()

        if arch == 'dinov2_vits14':
            self.features = torch.hub.load(
                'facebookresearch/dinov2', arch)
            self.features_dim = 384
            self.cell_size    = 14

        else:
            raise ValueError(f'Unsupported arch `{arch}`')

    def forward(self, x:Tensor) -> Tensor:
        dr, dc = x.shape[2] // self.cell_size, x.shape[3] // self.cell_size
        x = self.features.forward_features(x)['x_norm_patchtokens']
        return x.transpose(1, 2).reshape(-1, self.features_dim, dr, dc)


class CaresFeatures(nn.Module):

    def __init__(self, arch:str):

        super().__init__()

        self.cell_size = 8
        f2d_rows = 4

        if arch == 'small':
            self.features_dim= 192

            self.features2d = nn.Sequential(
                ConvNormActive(3, 8, kernel_size=3, stride=2, activation=nn.ReLU, norm_layer=None),
                ConvNormActive(8, 16, kernel_size=1, activation=nn.ReLU),

                InvertedResidual(16, 16, 1, stride=2, activation=nn.ReLU),
                InvertedResidual(16, 24, 4, stride=2, activation=nn.ReLU),
                InvertedResidual(24, 24, 4, stride=1, activation=nn.ReLU),
            )

            features2d_channel = 24

        elif arch == 'large':
            self.features_dim= 320

            self.features2d = nn.Sequential(
                ConvNormActive(3, 8, kernel_size=3, stride=2, activation=nn.ReLU, norm_layer=None),
                ConvNormActive(8, 16, kernel_size=1, activation=nn.ReLU),

                InvertedResidual(16, 16, 1, stride=1, activation=nn.ReLU),
                InvertedResidual(16, 24, 4, stride=2, activation=nn.ReLU),
                InvertedResidual(24, 24, 3, stride=1, activation=nn.ReLU),
                InvertedResidual(24, 40, 3, kernel_size=5, stride=2, activation=nn.ReLU),
                InvertedResidual(40, 40, 3, kernel_size=5, stride=1, activation=nn.ReLU),
                InvertedResidual(40, 40, 3, kernel_size=5, stride=1, activation=nn.ReLU),
            )

            features2d_channel = 40

        elif arch in ('largex48', 'largex64'):
            self.features_dim= 320
            self.cell_size = 16

            self.features2d = nn.Sequential(
                ConvNormActive(3, 8, kernel_size=3, stride=2, activation=nn.ReLU, norm_layer=None),
                ConvNormActive(8, 16, kernel_size=1, activation=nn.ReLU),

                InvertedResidual(16, 16, 1, stride=1, activation=nn.ReLU),
                InvertedResidual(16, 24, 4, stride=2, activation=nn.ReLU),
                InvertedResidual(24, 24, 3, stride=1, activation=nn.ReLU),
                InvertedResidual(24, 40, 3, kernel_size=5, stride=2, activation=nn.ReLU),
                InvertedResidual(40, 40, 3, kernel_size=5, stride=1, activation=nn.ReLU),
                InvertedResidual(40, 40, 3, kernel_size=5, stride=2, activation=nn.ReLU),
                InvertedResidual(40, 80, 6, kernel_size=5, stride=1, activation=nn.ReLU),
            )

            features2d_channel = 80
            if arch.endswith('x48'):
                f2d_rows = 3
            elif arch.endswith('x64'):
                f2d_rows = 4

        else:
            raise ValueError(f'Unsupported arch `{arch}`')

        hidden_dim = features2d_channel * f2d_rows

        self.sq_attation = SqueezeAttention(
            features2d_channel, in_rows=f2d_rows)

        self.feedforward = nn.Sequential(
            InvertedResidual(
                hidden_dim,
                hidden_dim,
                6,
                kernel_size=(1, 5),
                norm_layer=LayerNorm2dChannel,
                use_sse=True,
                use_res_connect=False,
            ),
            # InvertedResidual(
            #     hidden_dim,
            #     hidden_dim,
            #     6,
            #     kernel_size=(1, 5),
            #     norm_layer=LayerNorm2dChannel,
            #     use_sse=True,
            #     use_res_connect=True,
            # ),
            ConvNormActive(
                hidden_dim,
                self.features_dim,
                kernel_size=1,
                norm_layer=LayerNorm2dChannel,
            ),
        )

    def forward(self, x:Tensor) -> Tensor:
        x = self.features2d(x)
        x = self.sq_attation(x)
        x = self.feedforward(x)
        return x
