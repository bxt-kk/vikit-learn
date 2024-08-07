from typing import Callable

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


class LayerNorm2d(nn.GroupNorm):

    def __init__(self, num_channels:int):
        super().__init__(1, num_channels)


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
            kernel_size: int=3,
            stride:      int | tuple[int, int]=1,
            dilation:    int=1,
            groups:      int=1,
            norm_layer:  Callable[..., nn.Module] | None=DEFAULT_NORM_LAYER,
            activation:  Callable[..., nn.Module] | None=DEFAULT_ACTIVATION,
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


class ConvNormActiveRes(ConvNormActive):

    def forward(self, x:Tensor) -> Tensor:
        return super().forward(x) + x


class InvertedResidual(nn.Module):

    def __init__(
            self,
            in_planes:       int,
            out_planes:      int,
            expand_ratio:    int,
            kernel_size:     int=3,
            stride:          int | tuple[int, int]=1,
            dilation:        int=1,
            norm_layer:      Callable[..., nn.Module] | None=DEFAULT_NORM_LAYER,
            activation:      Callable[..., nn.Module] | None=DEFAULT_ACTIVATION,
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


class PoolWithPosCode(nn.Module):

    def __init__(self, stride:int):
        super().__init__()

        self.stride = stride

    def forward(self, x:Tensor) -> Tensor:
        in_planes = x.shape[1]

        l = (x * x).sum(dim=1, keepdim=True)
        u = F.pixel_unshuffle(l, self.stride)
        I = u.argmax(dim=1, keepdim=True)
        cr = (I // self.stride).type_as(x) / (self.stride - 1)
        cc = (I % self.stride).type_as(x) / (self.stride - 1)

        # x = F.pixel_unshuffle(x, self.stride)
        # x = x.transpose(1, 3).reshape(-1, in_planes, self.stride**2)
        # J = I.transpose(1, 3).flatten(0, -1)
        # x = x[range(len(J)), ..., J]
        #
        # bs, _, ny, nx = u.shape
        # x = x.reshape(bs, nx, ny, in_planes).transpose(1, 3)
        # return torch.cat([cr, cc, x], dim=1)
        bs, _, ny, nx = u.shape
        x = F.pixel_unshuffle(x, self.stride)
        x = x.reshape(bs, in_planes, -1, ny, nx)
        J = I.unsqueeze(1).expand(-1, in_planes, -1, -1, -1)
        x = x.gather(dim=2, index=J).squeeze(2)
        return torch.cat([cr, cc, x], dim=1)


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
        )
        self.sigmoid = DEFAULT_SIGMOID(inplace=False)

    def forward(self, x:Tensor) -> Tensor:
        fa = self.dense(F.adaptive_avg_pool2d(x, 1))
        fm = self.dense(F.adaptive_max_pool2d(x, 1))
        return self.sigmoid(fa + fm) * x


class SpatialAttention(nn.Module):

    def __init__(
            self,
            kernel_size: int=7,
        ):

        super().__init__()

        self.dense = ConvNormActive(
            2, 1, kernel_size, norm_layer=None, activation=DEFAULT_SIGMOID)

    def forward(self, x:Tensor) -> Tensor:
        f = torch.cat([
            x.mean(dim=1, keepdim=True),
            x.max(dim=1, keepdim=True).values,
        ], dim=1)
        return self.dense(f) * x


class CBANet(nn.Module):

    def __init__(
            self,
            in_planes:     int,
            out_planes:    int,
            kernel_size:   int=7,
            shrink_factor: int=4,
        ):

        super().__init__()

        self.channel_attention = ChannelAttention(in_planes, shrink_factor)
        self.spatial_attention = SpatialAttention(kernel_size)
        self.project = ConvNormActive(
            in_planes, out_planes, 1, activation=None)

    def forward(self, x:Tensor) -> Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return self.project(x)


class DetPredictor(nn.Module):

    def __init__(
            self,
            in_planes:     int,
            num_anchors:   int,
            bbox_dim:      int,
            num_classes:   int,
            dropout_p:     float,
        ):

        super().__init__()

        self.num_anchors = num_anchors

        self.conf_predict = nn.Sequential(
            ConvNormActive(in_planes, in_planes, 1),
            ConvNormActive(in_planes, in_planes, 3, groups=in_planes),
            nn.Conv2d(in_planes, num_anchors, kernel_size=1))

        self.bbox_predict = nn.Sequential(
            ConvNormActive(in_planes, in_planes, 1),
            ConvNormActive(in_planes, in_planes, 3, groups=in_planes),
            nn.Conv2d(in_planes, bbox_dim * num_anchors, kernel_size=1))

        clss_hidden = in_planes * num_anchors

        self.expansion = nn.Sequential(
            ConvNormActive(in_planes, clss_hidden, 1),
            ConvNormActive(clss_hidden, clss_hidden, 3, groups=clss_hidden))

        self.dropout2d = nn.Dropout2d(dropout_p, inplace=False)

        self.clss_predict = nn.Conv2d(
            clss_hidden, num_classes * num_anchors, kernel_size=1, groups=num_anchors)

    def forward(self, x:Tensor) -> Tensor:
        bs, _, ny, nx = x.shape

        p_conf = self.conf_predict(x)
        p_bbox = self.bbox_predict(x)

        e = self.expansion(x).view(bs * self.num_anchors, -1, ny, nx)
        e = self.dropout2d(e).view(bs, -1, ny, nx)
        p_clss = self.clss_predict(e)

        return torch.cat([
            part.view(bs, self.num_anchors, -1, ny, nx).permute(0, 1, 3, 4, 2)
            for part in (p_conf, p_bbox, p_clss)
        ], dim=-1)


class SegPredictor(nn.Module):

    def __init__(
            self,
            in_planes:   int,
            num_classes: int,
            num_layers:  int,
        ):

        super().__init__()

        self.num_layers = num_layers

        project_dim = min(
            max(in_planes // 2**num_layers, num_classes**0.5),
            num_classes,
        )

        self.projects = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for t in range(num_layers):
            self.projects.append(nn.Conv2d(in_planes, project_dim, 1))
            out_planes = in_planes
            if out_planes > num_classes**0.5:
                out_planes //= 2
            self.upsamples.append(nn.Sequential(
                ConvNormActive(in_planes + project_dim, out_planes, 1),
                UpSample(out_planes),
            ))
            in_planes = out_planes
        self.projects.append(nn.Conv2d(in_planes, project_dim, 1))

        self.norm_layer = DEFAULT_NORM_LAYER(project_dim)
        self.classifier = nn.Conv2d(project_dim, num_classes, 1)

    def forward(self, x:Tensor) -> Tensor:
        p = 0.
        for t in range(self.num_layers):
            p = p + self.projects[t](x)
            x = self.upsamples[t](torch.cat([x, p], dim=1))
            p = F.interpolate(p, scale_factor=2, mode='bilinear')
        x = p + self.projects[-1](x)
        return self.classifier(self.norm_layer(x))


class MobileNetFeatures(nn.Module):

    def __init__(self, arch:str, pretrained:bool):

        super().__init__()

        if arch == 'mobilenet_v3_small':
            features = mobilenet_v3_small(
                weights=MobileNet_V3_Small_Weights.DEFAULT
                if pretrained else None,
            ).features

            fd_dim = 24
            fm_dim = 48
            fu_dim = 96

            self.features_d = features[:3] # 24, 64, 64
            self.features_m = features[3:8] # 48, 32, 32
            self.features_u = features[8:-1] # 96, 16, 16

        elif arch == 'mobilenet_v3_large':
            features = mobilenet_v3_large(
                weights=MobileNet_V3_Large_Weights.DEFAULT
                if pretrained else None,
            ).features

            fd_dim = 40
            fm_dim = 112
            fu_dim = 160

            self.features_d = features[:5] # 40, 64, 64
            self.features_m = features[5:12] # 112, 32, 32
            self.features_u = features[12:-1] # 160, 16, 16

        elif arch == 'mobilenet_v3_larges':
            weights_state = deeplabv3_mobilenet_v3_large(
                weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
            ).backbone.state_dict()
            features = mobilenet_v3_large().features
            features.load_state_dict(weights_state)

            fd_dim = 40
            fm_dim = 112
            fu_dim = 160

            self.features_d = features[:5] # 40, 64, 64
            self.features_m = features[5:12] # 112, 32, 32
            self.features_u = features[12:-1] # 160, 16, 16

        elif arch == 'mobilenet_v2':
            features = mobilenet_v2(
                weights=MobileNet_V2_Weights.DEFAULT
                if pretrained else None,
            ).features

            fd_dim = 32
            fm_dim = 96
            fu_dim = 160

            self.features_d = features[:5] # 32, 64, 64
            self.features_m = features[5:12] # 96, 32, 32
            self.features_u = features[12:-2] # 160, 16, 16

        else:
            raise ValueError(f'Unsupported arch `{arch}`')

        self.features_dim = fd_dim + fm_dim + fu_dim
        self.cell_size    = 16

    def forward(self, x:Tensor) -> Tensor:
        fd = self.features_d(x)
        fm = self.features_m(fd)
        fu = self.features_u(fm)

        return torch.cat([
            F.max_pool2d(fd, kernel_size=3, stride=2, padding=1),
            fm,
            F.interpolate(fu, scale_factor=2, mode='nearest'),
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
        return x.transpose(1, 2).view(-1, self.features_dim, dr, dc)
