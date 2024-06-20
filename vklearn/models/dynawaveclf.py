from typing import List, Any, Dict, Mapping

from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from .component import WaveBase, InvWaveBase
from .classifier import Classifier


class DynawaveClf(Classifier):
    '''A light-weight and easy-to-train model for image classification

    Args:
        categories: Target categories.
        dropout: Dropout parameters in the classifier.
    '''

    def __init__(
            self,
            categories: List[str],
            num_global: int=3,
            dropout:    float=0.2,
        ):
        super().__init__(categories)

        self.dropout = dropout

        self.features = nn.Sequential(
            WaveBase(),
            WaveBase(),
            WaveBase(),
            nn.Conv2d(192, 96, 1),
            WaveBase(),
            nn.Conv2d(384, 192, 1),
            nn.BatchNorm2d(192),
        ) # 192, 16, 16

        features_dim = 192

        self.global_wave = nn.ModuleList([
            nn.Sequential(
                WaveBase(), # c1, 8, 8
                nn.Conv2d(192 * 4, 192 * 2, 1, groups=192),
                nn.BatchNorm2d(192 * 2),
                nn.Hardswish(inplace=False),
                WaveBase(), # c2, 4, 4
                nn.Conv2d(192 * 8, 192 * 4, 1, groups=192),
                nn.BatchNorm2d(192 * 4),
                nn.Hardswish(inplace=False),
                WaveBase(), # c3, 2, 2
                nn.Conv2d(192 * 16, 192 * 16, 3, padding=1, groups=192 * 16),
                nn.BatchNorm2d(192 * 16),
                nn.Hardswish(inplace=False),
                nn.Conv2d(192 * 16, 192 * 16, 1, groups=192),
                nn.BatchNorm2d(192 * 16),
                nn.Hardswish(inplace=False),
                InvWaveBase(), # c3, 2, 2
                nn.Conv2d(192 * 4, 192 * 8, 1, groups=192),
                nn.BatchNorm2d(192 * 8),
                nn.Hardswish(inplace=False),
                InvWaveBase(), # c2, 4, 4
                nn.Conv2d(192 * 2, 192 * 4, 1, groups=192),
                nn.BatchNorm2d(192 * 4),
                nn.Hardswish(inplace=False),
                InvWaveBase(), # c1, 8, 8
                nn.Conv2d(192, 192, 1),
                nn.BatchNorm2d(192),
                nn.Hardswish(inplace=False),
            ) for _ in range(num_global)])

        expanded_dim = features_dim * 4

        self.predict_clss = nn.Sequential(
            # nn.BatchNorm2d(merged_dim),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(features_dim, expanded_dim),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(expanded_dim, self.num_classes)
        )

    def forward_features(self, x:Tensor) -> Tensor:
        x = self.features(x)
        for layer in self.global_wave:
            x = x + layer(x)
        return x

    def forward(self, x:Tensor) -> Tensor:
        x = self.forward_features(x)
        return self.predict_clss(x)

    @classmethod
    def load_from_state(cls, state:Mapping[str, Any]) -> 'DynawaveClf':
        hyps = state['hyperparameters']
        model = cls(
            categories     = hyps['categories'],
            dropout        = hyps['dropout'],
        )
        model.load_state_dict(state['model'])
        return model

    def hyperparameters(self) -> Dict[str, Any]:
        return dict(
            categories     = self.categories,
            dropout        = self.dropout,
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
