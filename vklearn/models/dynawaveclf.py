from typing import List, Any, Dict, Mapping

from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

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
            nn.PixelUnshuffle(2),
            nn.Conv2d(12, 48, 3, padding=1, stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 96, 3, padding=1, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 192, 3, padding=1, stride=2),
            nn.BatchNorm2d(192),
        ) # 192, 32, 32

        features_dim = 192

        self.global_wave = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(192, 192 * 2, 3, padding=1, stride=2, groups=192),
                    nn.BatchNorm2d(192 * 2)), # 16
                nn.Sequential(
                    nn.Conv2d(192 * 2, 192 * 4, 3, padding=1, stride=2, groups=192),
                    nn.BatchNorm2d(192 * 4)), # 8
                nn.Sequential(
                    nn.Conv2d(192 * 4, 192 * 8, 3, padding=1, stride=2, groups=192),
                    nn.BatchNorm2d(192 * 8)), # 4

                nn.Sequential(
                    nn.Conv2d(192 * 8, 192 * 8, 3, padding=1, stride=1, groups=192),
                    nn.BatchNorm2d(192 * 8),
                ), # 2

                nn.Sequential(
                    nn.ConvTranspose2d(192 * 8, 192 * 4, 3, 2, 1, output_padding=1, groups=192),
                    nn.BatchNorm2d(192 * 4)),
                nn.Sequential(
                    nn.ConvTranspose2d(192 * 4, 192 * 2, 3, 2, 1, output_padding=1, groups=192),
                    nn.BatchNorm2d(192 * 2)),
                nn.Sequential(
                    nn.ConvTranspose2d(192 * 2, 192 * 1, 3, 2, 1, output_padding=1, groups=192),
                    nn.BatchNorm2d(192 * 1)),

                nn.Sequential(
                    nn.Conv2d(192, 192, 1),
                    nn.BatchNorm2d(192),
                    nn.ReLU(),
                    nn.Conv2d(192, 192, 1),
                    nn.BatchNorm2d(192),
                ),

                nn.BatchNorm2d(192),
            ]) for _ in range(num_global)])


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
        for block in self.global_wave:
            x0 = x
            vs = [x]
            for n, layer in enumerate(block[:-2]):
                x = layer(x)
                if n < 3:
                    vs.append(x)
                else:
                    x = x + vs.pop()
            x = block[-1](x0 + block[-2](x))
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
