from typing import List, Any, Dict, Mapping

from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from .classifier import Classifier
from .trimnetx import TrimNetX
from .component import DEFAULT_ACTIVATION


class TrimNetClf(Classifier):
    '''A light-weight and easy-to-train model for image classification

    Args:
        categories: Target categories.
        num_scans: Number of the Trim-Units.
        scan_range: Range factor of the Trim-Unit convolution.
        backbone: Specify a basic model as a feature extraction module.
        backbone_pretrained: Whether to load backbone pretrained weights.
    '''

    def __init__(
            self,
            categories:          List[str],
            num_scans:           int=3,
            scan_range:          int=4,
            dropout_p:           float=0.2,
            backbone:            str='mobilenet_v3_small',
            backbone_pretrained: bool=True,
        ):
        super().__init__(categories)

        self.dropout_p = dropout_p

        self.trimnetx = TrimNetX(
            num_scans, scan_range, backbone, backbone_pretrained)

        merged_dim = self.trimnetx.merged_dim
        expanded_dim = merged_dim * 4

        self.predictor = nn.Sequential(
            nn.Linear(merged_dim, expanded_dim),
            DEFAULT_ACTIVATION(inplace=False),
            nn.Dropout(dropout_p, inplace=False),
            nn.Linear(expanded_dim, self.num_classes),
        )

        self.alphas = nn.Parameter(torch.ones(
            1, self.num_classes, num_scans) / num_scans**0.5)

    def train_features(self, flag:bool):
        self.trimnetx.train_features(flag)

    def forward(self, x:Tensor) -> Tensor:
        hs, _ = self.trimnetx(x)
        alphas = self.alphas.softmax(dim=-1)
        p = 0.
        times = len(hs)
        for t in range(times):
            h = F.adaptive_avg_pool2d(hs[t], 1).flatten(start_dim=1)
            p = p + self.predictor(h) * alphas[..., t]
        return p

    @classmethod
    def load_from_state(cls, state:Mapping[str, Any]) -> 'TrimNetClf':
        hyps = state['hyperparameters']
        model = cls(
            categories          = hyps['categories'],
            num_scans           = hyps['num_scans'],
            scan_range          = hyps['scan_range'],
            dropout_p           = hyps['dropout_p'],
            backbone            = hyps['backbone'],
            backbone_pretrained = False,
        )
        model.load_state_dict(state['model'])
        return model

    def hyperparameters(self) -> Dict[str, Any]:
        return dict(
            categories = self.categories,
            dropout_p  = self.dropout_p,
            num_scans  = self.trimnetx.num_scans,
            scan_range = self.trimnetx.scan_range,
            backbone   = self.trimnetx.backbone,
        )

    def classify(
            self,
            image:      Image.Image,
            top_k:      int=10,
            align_size: int=224,
        ) -> List[Dict[str, Any]]:

        device = self.get_model_device()
        x = self.preprocess(image, align_size)
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

        if len(target.shape) == 2:
            predict = torch.softmax(inputs, dim=-1)
            accuracy = (1 - 0.5 * torch.abs(predict - target).sum(dim=-1)).mean()
        else:
            predict = torch.argmax(inputs, dim=-1)
            accuracy = (predict == target).sum() / len(predict)

        return dict(
            accuracy=accuracy,
        )
