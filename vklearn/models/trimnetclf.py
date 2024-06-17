from typing import List, Any, Dict, Mapping

from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

from PIL import Image

from .component import LinearBasicConvBD, CSENet
from .classifier import Classifier


class TrimNetClf(Classifier):
    '''A light-weight and easy-to-train model for image classification

    Args:
        categories: Target categories.
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
            categories:          List[str],
            dilation_depth:      int=2,
            dilation_range:      int=4,
            dropout:             float=0.2,
            backbone:            str='mobilenet_v3_small',
            backbone_pretrained: bool=True,
        ):
        super().__init__(categories)

        self.dilation_depth = dilation_depth
        self.dilation_range = dilation_range
        self.dropout        = dropout
        self.backbone       = backbone

        if backbone == 'mobilenet_v3_small':
            features = mobilenet_v3_small(
                weights=MobileNet_V3_Small_Weights.DEFAULT
                if backbone_pretrained else None,
            ).features

            features_dim = 48 + 96
            merged_dim   = 160
            expanded_dim = merged_dim * 4

            self.features_d = features[:9] # 48, 32, 32
            self.features_u = features[9:-1] # 96, 16, 16

        elif backbone == 'mobilenet_v3_large':
            features = mobilenet_v3_large(
                weights=MobileNet_V3_Large_Weights.DEFAULT
                if backbone_pretrained else None,
            ).features

            features_dim = 112 + 160
            merged_dim   = 320
            expanded_dim = merged_dim * 4

            self.features_d = features[:13] # 112, 32, 32
            self.features_u = features[13:-1] # 160, 16, 16

        self.merge = nn.Sequential(
            nn.Conv2d(features_dim, merged_dim, 1, bias=False),
            nn.BatchNorm2d(merged_dim),
        )

        self.cluster = nn.ModuleList()
        self.csenets = nn.ModuleList()
        for _ in range(dilation_depth):
            modules = []
            for r in range(dilation_range):
                modules.append(
                    LinearBasicConvBD(merged_dim, merged_dim, dilation=2**r))
            modules.append(nn.Sequential(
                nn.Conv2d(merged_dim, merged_dim, 1, bias=False),
                nn.BatchNorm2d(merged_dim),
                nn.Hardswish(inplace=True),
            ))
            self.cluster.append(nn.Sequential(*modules))
            self.csenets.append(CSENet(
                merged_dim * 2, merged_dim, kernel_size=3, shrink_factor=4))

        self.predict_clss = nn.Sequential(
            # nn.BatchNorm2d(merged_dim),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(merged_dim, expanded_dim),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(expanded_dim, self.num_classes)
        )

    def forward_features(self, x:Tensor) -> Tensor:
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

    def forward(self, x:Tensor) -> Tensor:
        x = self.forward_features(x)
        return self.predict_clss(x)

    @classmethod
    def load_from_state(cls, state:Mapping[str, Any]) -> 'TrimNetClf':
        hyps = state['hyperparameters']
        model = cls(
            categories          = hyps['categories'],
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
            categories     = self.categories,
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

        device = self.get_model_device()
        x, scale, pad_x, pad_y = self.preprocess(
            image, align_size, limit_size=32, fill_value=127)
        x = x.to(device)
        x = self.forward(x)
        top_k = min(self.num_classes, top_k)
        topk = x.squeeze(dim=0).softmax(dim=-1).topk(top_k)
        probs = [round(v, 5) for v in topk.values.tolist()]
        labels = [self.categories[cid] for cid in topk.indices]
        return dict(
            probs=dict(zip(labels, probs)),
            predict=labels[0],
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
