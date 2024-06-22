from typing import List, Any, Dict, Mapping

from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from .classifier import Classifier
from .dynawavenet import DynawaveNet


class DynawaveClf(Classifier):
    '''A light-weight and easy-to-train model for image classification

    Args:
        categories: Target categories.
        num_waves: Number of the global wave blocks.
        wave_depth: Depth of the wave block.
        dropout: Dropout parameters in the classifier.
    '''

    def __init__(
            self,
            categories: List[str],
            num_waves:  int=3,
            wave_depth: int=3,
            dropout:    float=0.2,
        ):
        super().__init__(categories)

        self.num_waves  = num_waves
        self.wave_depth = wave_depth
        self.dropout    = dropout

        self.dynawavenet = DynawaveNet(num_waves, wave_depth)

        features_dim = self.dynawavenet.features_dim
        expanded_dim = features_dim * 4

        self.predict_clss = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(features_dim, expanded_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(expanded_dim, self.num_classes)
        )

    def forward(self, x:Tensor) -> Tensor:
        x = self.dynawavenet(x)
        return self.predict_clss(x)

    @classmethod
    def load_from_state(cls, state:Mapping[str, Any]) -> 'DynawaveClf':
        hyps = state['hyperparameters']
        model = cls(
            categories = hyps['categories'],
            num_waves  = hyps['num_waves'],
            wave_depth = hyps['wave_depth'],
            dropout    = hyps['dropout'],
        )
        model.load_state_dict(state['model'])
        return model

    def hyperparameters(self) -> Dict[str, Any]:
        return dict(
            categories = self.categories,
            num_waves  = self.num_waves,
            wave_depth = self.wave_depth,
            dropout    = self.dropout,
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
