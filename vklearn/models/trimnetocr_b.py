from typing import List, Any, Dict, Mapping, Tuple

from torch import Tensor

import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from .ocr import OCR
from .trimnetx import TrimNetX
from .component import LayerNorm2dChannel, CBANet


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
            num_scans:           int | None=None,
            scan_range:          int | None=None,
            backbone:            str | None='cares_large',
            backbone_pretrained: bool | None=False,
        ):

        assert backbone.startswith('cares')

        super().__init__(categories)

        self.dropout_p = dropout_p

        self.trimnetx = TrimNetX(
            num_scans, scan_range, backbone, backbone_pretrained,
            norm_layer=LayerNorm2dChannel)

        features_dim = self.trimnetx.features_dim
        merged_dim = self.trimnetx.merged_dim

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p, inplace=False),
            nn.Linear(merged_dim, self.num_classes),
        )

        self.auxi_clf = nn.Sequential(
            nn.Dropout(dropout_p, inplace=False),
            nn.Linear(features_dim, self.num_classes),
        )

        for m in list(self.trimnetx.trim_units.modules()):
            if not isinstance(m, CBANet): continue
            m.channel_attention = nn.Identity()
            m.spatial_attention = nn.Identity()

    def train_features(self, flag:bool):
        self.trimnetx.train_features(flag)

    def forward(self, x:Tensor) -> Tuple[Tensor, Tensor, List[Tensor]]:
        hs, f = self.trimnetx(x)
        # n, c, 1, w -> n, c, w -> n, w, c
        x = hs[-1]
        x = x.squeeze(dim=2).transpose(1, 2)
        x = self.classifier(x)
        return x, f, hs[:-1]

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

    def calc_loss(
            self,
            inputs:         Tuple[Tensor, Tensor, List[Tensor]],
            targets:        Tensor,
            target_lengths: Tensor,
            zero_infinity:  bool=False,
            weights:        Dict[str, float] | None=None,
        ) -> Dict[str, Any]:

        predicts, features, hiddens = inputs

        losses = super().calc_loss(
            predicts, targets, target_lengths, zero_infinity=zero_infinity)

        weights = weights or dict()
        auxi_weight = weights.get('auxi', 0)
        if auxi_weight == 0:
            return losses

        loss = losses['loss']
        losses['hf_loss'] = loss

        features = features.squeeze(dim=2).transpose(1, 2)
        auxi_predicts = self.auxi_clf(features)
        auxi_losses = super().calc_loss(
            auxi_predicts, targets, target_lengths, zero_infinity=zero_infinity)
        auxi_loss = auxi_losses['loss']
        losses['auxi_loss'] = auxi_loss

        hidden_weight = weights.get('hidden', 0)
        if hidden_weight == 0: hiddens = []

        hidden_loss = 0
        for hid, hidden in enumerate(hiddens):
            hidden = hidden.squeeze(dim=2).transpose(1, 2)
            predicts = self.classifier(hidden)
            hidden_i_loss = super().calc_loss(
                predicts, targets, target_lengths, zero_infinity=zero_infinity)['loss']
            hidden_loss = hidden_loss + hidden_i_loss
            losses[f'h{hid}_loss'] = hidden_i_loss

        losses['loss'] = (
            loss * max(0, 1 - auxi_weight - len(hiddens) * hidden_weight) +
            auxi_loss * auxi_weight + hidden_loss * hidden_weight)
        return losses

    def calc_score(
            self,
            inputs:         Tuple[Tensor, Tensor, List[Tensor]],
            targets:        Tensor,
            target_lengths: Tensor,
        ) -> Dict[str, Any]:
        return super().calc_score(inputs[0], targets, target_lengths)

    def update_metric(
            self,
            inputs:         Tuple[Tensor, Tensor, List[Tensor]],
            targets:        Tensor,
            target_lengths: Tensor,
        ):
        return super().update_metric(inputs[0], targets, target_lengths)
