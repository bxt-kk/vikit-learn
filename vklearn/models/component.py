from typing import Callable, List

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.misc import SqueezeExcitation
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

import clip


class LayerNorm2d(nn.GroupNorm):

    def __init__(self, num_channels:int):
        super().__init__(1, num_channels)


DEFAULT_LAYER_NORM = LayerNorm2d # nn.BatchNorm2d
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
            norm_layer:  Callable[..., nn.Module] | None=DEFAULT_LAYER_NORM,
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


class InvertedResidual(nn.Module):

    def __init__(
            self,
            in_planes:       int,
            out_planes:      int,
            expand_ratio:    int,
            kernel_size:     int=3,
            stride:          int | tuple[int, int]=1,
            dilation:        int=1,
            norm_layer:      Callable[..., nn.Module] | None=DEFAULT_LAYER_NORM,
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
            norm_layer: Callable[..., nn.Module] | None=DEFAULT_LAYER_NORM,
            activation: Callable[..., nn.Module] | None=DEFAULT_ACTIVATION,
        ):

        super().__init__(
            nn.ConvTranspose2d(in_planes, in_planes, 3, 2, 1, output_padding=1, groups=in_planes, bias=False),
            norm_layer(in_planes),
            activation(),
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
                shrink_dim, in_planes, 1, norm_layer=None, activation=DEFAULT_SIGMOID),
        )
        self.project = ConvNormActive(
            in_planes, out_planes, 1, activation=None)

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
            # ConvNormActive(in_planes, in_planes, kernel_size=1),
            # InvertedResidual(in_planes, in_planes, 1, use_res_connect=False),
            # DEFAULT_ACTIVATION(),
            ConvNormActive(in_planes, in_planes, 1),
            ConvNormActive(in_planes, in_planes, 3, groups=in_planes),
            nn.Dropout(p=dropout_bbox, inplace=True),
            nn.Conv2d(in_planes, num_anchors * bbox_dim, kernel_size=1),
        )

        self.predict_clss = nn.Sequential(
            # ConvNormActive(in_planes, hidden_planes, kernel_size=1),
            # InvertedResidual(in_planes, hidden_planes, 1),
            # DEFAULT_ACTIVATION(),
            ConvNormActive(in_planes, hidden_planes, 1),
            ConvNormActive(hidden_planes, hidden_planes, 3, groups=hidden_planes),
            nn.Dropout(p=dropout, inplace=True),
            nn.Conv2d(hidden_planes, num_anchors * num_classes, kernel_size=1),
        )

        self.num_anchors = num_anchors

    def forward(self, x:Tensor) -> Tensor:
        bs, _, ny, nx = x.shape
        p_bbox = self.predict_bbox(x).view(bs, self.num_anchors, -1, ny, nx)
        p_clss = self.predict_clss(x).view(bs, self.num_anchors, -1, ny, nx)
        return torch.cat([p_bbox, p_clss], dim=2).view(bs, -1, ny, nx)


class DetPredictorV2(nn.Module):

    def __init__(
            self,
            in_planes:     int,
            hidden_planes: int,
            num_anchors:   int,
            bbox_dim:      int,
            num_classes:   int,
        ):

        super().__init__()

        object_dim = (1 + bbox_dim + num_classes)
        predict_dim = object_dim * num_anchors
        self.predict = nn.Sequential(
            ConvNormActive(in_planes, hidden_planes, 1),
            ConvNormActive(hidden_planes, hidden_planes, 3, groups=hidden_planes),
            nn.Conv2d(hidden_planes, predict_dim, kernel_size=1))

        self.num_anchors = num_anchors

    def forward(self, x:Tensor) -> Tensor:
        bs, _, ny, nx = x.shape
        x = self.predict(x)
        x = x.view(bs, self.num_anchors, -1, ny, nx)
        x = x.permute(0, 1, 3, 4, 2)
        return x


class ClipConv2d1x1(nn.Conv2d):

    CODE_LENGTH = 512

    def __init__(
            self,
            in_planes:  int,
            out_planes: int,
            prompts:    List[str] | None=None,
        ):

        super().__init__(in_planes, out_planes, 1)

        priori = torch.zeros(out_planes, in_planes, 1, 1)
        scale = 1
        if prompts is not None:
            print('enable clip encoding:', prompts)
            num_classes = len(prompts)
            assert num_classes <= out_planes

            clip_device = 'cpu'
            clip_inputs = clip.tokenize(prompts).to(clip_device)
            clip_model, _ = clip.load('ViT-B/32', device=clip_device)
            with torch.no_grad():
                codes = clip_model.encode_text(clip_inputs)
            if in_planes < self.CODE_LENGTH:
                codes, _ = self._code_align_weight(codes, in_planes)
            scale = 1 / ((codes**2).sum(dim=1)**0.5).mean().item()
            num_codes = len(codes)
            for i in range(out_planes):
                code = codes[i % num_codes]
                priori[i, :len(code), 0, 0] = code
        self.register_buffer('priori', priori)
        self.scale = scale

    @classmethod
    def category_to_prompt(self, categories:List[str]) -> List[str]:
        prompts = []
        for category in categories:
            name = category.lower()
            prompt = 'an' if name[0] in 'aeiou' else 'a'
            prompts.append(prompt + ' ' + name)
        return prompts

    def _code_align_weight(self, code:Tensor, in_planes:int) -> Tensor:
        mean = code.mean(dim=0)
        X = code - mean
        cov = torch.cov(X.T)
        eigval, eigvect = torch.linalg.eig(cov)
        eigval_idxs = torch.argsort(eigval.real, descending=True)[:in_planes]
        red_eigvect = eigvect.real[:, eigval_idxs]
        short_code = X @ red_eigvect
        recon = (short_code @ red_eigvect.T) + mean
        return short_code, recon

    def forward(self, x:Tensor) -> Tensor:
        return self._conv_forward(
            x, self.weight + self.priori, self.bias) * self.scale


class ClipDetPredictor(nn.Module):

    def __init__(
            self,
            in_planes:     int,
            hidden_planes: int,
            num_anchors:   int,
            bbox_dim:      int,
            num_classes:   int,
            dropout:       float,
            prompts:       List[str],
            dropout_bbox:  float=0.,
        ):

        super().__init__()

        self.predict_bbox = nn.Sequential(
            ConvNormActive(in_planes, in_planes, 1),
            ConvNormActive(in_planes, in_planes, 3, groups=in_planes),
            nn.Dropout(p=dropout_bbox, inplace=True),
            nn.Conv2d(in_planes, num_anchors * bbox_dim, kernel_size=1),
        )

        self.predict_clss = nn.Sequential(
            ConvNormActive(in_planes, hidden_planes, 1),
            ConvNormActive(hidden_planes, hidden_planes, 3, groups=hidden_planes),
            nn.Dropout(p=dropout, inplace=True),
            # nn.Conv2d(hidden_planes, num_anchors * num_classes, kernel_size=1),
            ClipConv2d1x1(hidden_planes, num_anchors * num_classes, prompts),
        )

        self.num_anchors = num_anchors

    def forward(self, x:Tensor) -> Tensor:
        bs, _, ny, nx = x.shape
        p_bbox = self.predict_bbox(x).view(bs, self.num_anchors, -1, ny, nx)
        p_clss = self.predict_clss(x).view(bs, self.num_anchors, -1, ny, nx)
        return torch.cat([p_bbox, p_clss], dim=2).view(bs, -1, ny, nx)


class MobileNetFeatures(nn.Module):

    def __init__(self, arch:str, pretrained:bool):

        super().__init__()

        if arch == 'mobilenet_v3_small':
            features = mobilenet_v3_small(
                weights=MobileNet_V3_Small_Weights.DEFAULT
                if pretrained else None,
            ).features

            self.features_dim = 48 + 96

            self.features_d = features[:9] # 48, 32, 32
            self.features_u = features[9:-1] # 96, 16, 16

        elif arch == 'mobilenet_v3_large':
            features = mobilenet_v3_large(
                weights=MobileNet_V3_Large_Weights.DEFAULT
                if pretrained else None,
            ).features

            self.features_dim = 112 + 160

            self.features_d = features[:13] # 112, 32, 32
            self.features_u = features[13:-1] # 160, 16, 16

        elif arch == 'efficientnet_v2_s':
            features = efficientnet_v2_s(
                weights=EfficientNet_V2_S_Weights.DEFAULT
                if pretrained else None,
            ).features

            self.features_dim = 128 + 256

            self.features_d = features[:5] # 128, 32, 32
            self.features_u = features[5:-1] # 256, 16, 16

        else:
            raise ValueError(f'Unsupported arch `{arch}`')

        self.cell_size = 16

    def forward(self, x:Tensor) -> Tensor:
        # Lab code
        with torch.no_grad():
            fd = self.features_d(x)
        # >>>
        # fd = self.features_d(x)
        fu = self.features_u(fd)
        return torch.cat([
            fd, F.interpolate(fu, scale_factor=2, mode='bilinear')], dim=1)


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
