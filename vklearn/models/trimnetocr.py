from typing import List, Any, Dict, Mapping

from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from .ocr import OCR
from .trimnetx import TrimNetX
from .component import DEFAULT_ACTIVATION


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
            dropout_p:           float=0.2,
            num_scans:           int | None=None,
            scan_range:          int | None=None,
            backbone:            str | None=None,
            backbone_pretrained: bool | None=None,
        ):
        super().__init__(categories)

        self.dropout_p = dropout_p

        self.trimnetx = TrimNetX(
            num_scans, scan_range, backbone, backbone_pretrained)

        merged_dim = self.trimnetx.merged_dim
        expanded_dim = merged_dim * 2

        self.predictor = nn.Sequential(
            nn.Conv1d(merged_dim, expanded_dim, kernel_size=1),
            nn.BatchNorm1d(expanded_dim),
            DEFAULT_ACTIVATION(inplace=False),

            nn.Conv1d(expanded_dim, expanded_dim, kernel_size=3, padding=1, groups=expanded_dim),
            nn.BatchNorm1d(expanded_dim),
            DEFAULT_ACTIVATION(inplace=False),

            nn.Dropout(dropout_p, inplace=False),

            nn.Conv1d(expanded_dim, merged_dim, kernel_size=1),
            DEFAULT_ACTIVATION(inplace=False),
            nn.Conv1d(merged_dim, self.num_classes, kernel_size=1),
        )

        self.alphas = nn.Parameter(torch.zeros(
            1, merged_dim, 1, self.trimnetx.num_scans))

    def train_features(self, flag:bool):
        self.trimnetx.train_features(flag)

    def forward_(self, x:Tensor) -> Tensor:
        hs, _ = self.trimnetx(x)
        alphas = self.alphas.softmax(dim=-1)
        p = 0.
        times = len(hs)
        bs, cs, _, _ = hs[0].shape
        for t in range(times):
            # n, c, r, w -> n, (w, r), c: N, T, C
            h = hs[t].permute(0, 3, 2, 1).reshape(bs, -1, cs)
            p = p + self.predictor(h) * alphas[..., t]
        return p

    def forward(self, x:Tensor) -> Tensor:
        hs, _ = self.trimnetx(x)
        alphas = self.alphas.softmax(dim=-1)
        h = 0.
        times = len(hs)
        bs, cs, _, _ = hs[0].shape
        for t in range(times):
            # n, c, r, w -> n, c, (w, r): N, C, T
            h = hs[t].permute(0, 1, 3, 2).reshape(bs, cs, -1) * alphas[..., t] + t
        p = self.predictor(h).transpose(1, 2)
        return p

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
        kernel = torch.tensor([[[-1, 1]]]).type_as(preds)
        mask = torch.conv1d(
            F.pad(preds.unsqueeze(1), [0, 1], value=0), kernel).squeeze(1) != 0
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
