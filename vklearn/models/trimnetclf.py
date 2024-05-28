from typing import List, Any, Dict, Mapping

from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from PIL import Image

from .component import LinearBasicConvBD
from .classifier import Classifier


class TrimNetClf(Classifier):
    '''A light-weight and easy-to-train model for image classification

    Args:
        num_classes: Number of target categories.
        dilation_depth: Depth of dilation module.
        dilation_range: The impact region of dilation convolution.
        num_tries: Number of attempts to guess.
        swap_size: Dimensions of the exchanged data.
        dropout: Dropout parameters in the classifier.
        backbone: Specify a basic model as a feature extraction module.
        backbone_pretrained: Whether to load backbone pretrained weights.
    '''

    def __init__(
            self,
            num_classes:         int,
            dilation_depth:      int=4,
            dilation_range:      int=4,
            dropout:             float=0.2,
            backbone:            str='mobilenet_v3_small',
            backbone_pretrained: bool=True,
        ):
        super().__init__(num_classes)

        self.dilation_depth = dilation_depth
        self.dilation_range = dilation_range
        self.dropout        = dropout
        self.backbone       = backbone

        if backbone == 'mobilenet_v3_small':
            features = mobilenet_v3_small(
                weights=MobileNet_V3_Small_Weights.DEFAULT
                if backbone_pretrained else None,
            ).features

            features_dim = 24 * 4 + 48 + 96
            merged_dim   = 160
            expanded_dim = 320

            self.features_d = features[:4] # 24, 64, 64
            self.features_c = features[4:9] # 48, 32, 32
            self.features_u = features[9:-1] # 96, 16, 16

        elif backbone == 'mobilenet_v2':
            features = mobilenet_v2(
                weights=MobileNet_V2_Weights.DEFAULT
                if backbone_pretrained else None,
            ).features

            features_dim = 32 * 4 + 96 + 320
            merged_dim   = 320
            expanded_dim = 640

            self.features_d = features[:7] # 32, 64, 64
            self.features_c = features[7:14] # 96, 32, 32
            self.features_u = features[14:-1] # 320, 16, 16

        self.merge = nn.Sequential(
            nn.Conv2d(features_dim, merged_dim, 1, bias=False),
            nn.BatchNorm2d(merged_dim),
        )

        self.cluster = nn.ModuleList()
        for _ in range(dilation_depth):
            modules = []
            for r in range(dilation_range):
                modules.append(
                    LinearBasicConvBD(merged_dim, merged_dim, dilation=2**r))
            modules.append(nn.Sequential(
                nn.Hardswish(inplace=True),
                nn.Conv2d(merged_dim, merged_dim, 1, bias=False),
                nn.BatchNorm2d(merged_dim),
            ))
            self.cluster.append(nn.Sequential(*modules))

        self.predict_clss = nn.Sequential(
            nn.Conv2d(merged_dim, expanded_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(expanded_dim),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(expanded_dim, expanded_dim * 2),
            nn.LayerNorm(expanded_dim * 2),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(expanded_dim * 2, num_classes)
        )

    def forward_features(
            self,
            x:              Tensor,
            train_features: bool,
        ) -> Tensor:

        if train_features:
            fd = self.features_d(x)
            fc = self.features_c(fd)
            fu = self.features_u(fc)
        else:
            with torch.no_grad():
                fd = self.features_d(x)
                fc = self.features_c(fd)
                fu = self.features_u(fc)

        x = self.merge(torch.cat([
            F.pixel_unshuffle(fd, 2),
            fc,
            F.interpolate(fu, scale_factor=2, mode='bilinear'),
        ], dim=1))
        for layer in self.cluster:
            x = x + layer(x)
        return x

    def forward(
            self,
            x:              Tensor,
            train_features: bool=True,
        ) -> Tensor:

        x = self.forward_features(x, train_features)
        return self.predict_clss(x)

    @classmethod
    def load_from_state(cls, state:Mapping[str, Any]) -> 'TrimNetClf':
        hyps = state['hyperparameters']
        model = cls(
            num_classes         = hyps['num_classes'],
            dilation_depth      = hyps['dilation_depth'],
            dilation_range      = hyps['dilation_range'],
            dropout             = hyps['dropout'],
            backbone            = hyps['backbone'],
            backbone_pretrained = False,
        )
        model.load_state_dict(state['model'])
        return model

    def hyperparameters(self) -> Dict[str, Any]:
        return dict(
            num_classes    = self.num_classes,
            dilation_depth = self.dilation_depth,
            dilation_range = self.dilation_range,
            dropout        = self.dropout,
            backbone       = self.backbone,
        )

    def classify(
            self,
            image:      Image.Image,
            top_k:      int=10,
            align_size: int=224,
        ) -> List[Dict[str, Any]]:

        device = next(self.parameters()).device
        x, scale, pad_x, pad_y = self.preprocess(
            image, align_size, limit_size=32, fill_value=127)
        x = x.to(device)
        x = self.forward(x, train_features=False)
        topk = x.squeeze(dim=0).softmax(dim=-1).topk(top_k)
        probs = [round(v, 5) for v in topk.values.tolist()]
        index = topk.indices.tolist()
        return dict(
            probs=dict(zip(index, probs)),
            predict=index[0],
        )

    def calc_loss(
            self,
            inputs:  Tensor,
            target:  Tensor,
            weights: Dict[str, float] | None=None,
            alpha:   float=0.25,
            gamma:   float=2.,
        ) -> Dict[str, Any]:

        reduction = 'mean'

        loss = F.cross_entropy(inputs, target, reduction=reduction)

        return dict(
            loss=loss,
        )

    def calc_score(
            self,
            inputs: Tensor,
            target: Tensor,
            thresh: float=0.5,
            eps:    float=1e-5,
        ) -> Dict[str, Any]:

        predict = torch.argmax(inputs, dim=-1)
        accuracy = (predict == target).sum() / len(predict)

        return dict(
            accuracy=accuracy,
        )
