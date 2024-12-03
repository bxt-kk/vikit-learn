from typing import List, Any, Dict, Mapping

from torch import Tensor

import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from .ocr import OCR
from .trimnetx import TrimNetX


class TrimNetOcr(OCR):
    '''A light-weight and easy-to-train model for ocr

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
            dropout_p:           float=0.,
            num_scans:           int | None=0,
            scan_range:          int | None=None,
            backbone:            str | None='cares_large',
            backbone_pretrained: bool | None=False,
        ):
        super().__init__(categories)

        self.dropout_p = dropout_p

        self.trimnetx = TrimNetX(
            num_scans, scan_range, backbone, backbone_pretrained)

        features_dim = self.trimnetx.features_dim

        self.rnn = nn.LSTM(features_dim, features_dim // 2, batch_first=True, bidirectional=True)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p, inplace=False),
            nn.Linear(features_dim, self.num_classes),
        )

    def train_features(self, flag:bool):
        self.trimnetx.train_features(flag)

    def forward(self, x:Tensor) -> Tensor:
        hs, x = self.trimnetx(x)
        # n, c, 1, w -> n, c, w -> n, w, c
        x = x.squeeze(dim=2).transpose(1, 2)
        # print('rnn out shape:', self.rnn(x)[0].shape)
        s, _ = self.rnn(x.detach())
        x = self.classifier(x + s)
        return x

    @classmethod
    def load_from_state(cls, state:Mapping[str, Any]) -> 'TrimNetOcr':
        hyps = state['hyperparameters']
        model = cls(
            categories          = hyps['categories'],
            dropout_p           = hyps['dropout_p'],
            num_scans           = hyps['num_scans'],
            scan_range          = hyps['scan_range'],
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

    def recognize(
            self,
            image:      Image.Image,
            top_k:      int=10,
            align_size: int=32,
        ) -> Dict[str, Any]:

        device = self.get_model_device()
        x = self.preprocess(image, align_size)
        x = x.to(device)
        x = self.forward(x)

        preds = x.argmax(dim=2) # n, T
        mask = (F.pad(preds, [0, 1], value=0)[:, 1:] - preds) != 0
        preds = preds * mask
        text = ''.join(self._categorie_arr[preds[0].cpu().numpy()])

        top_k = min(self.num_classes, top_k)
        topk = x.squeeze(dim=0).softmax(dim=-1).topk(top_k)
        probs = topk.values.cpu().numpy()
        labels = topk.indices.cpu().numpy()
        return dict(
            probs=probs,
            labels=labels,
            text=text,
        )
