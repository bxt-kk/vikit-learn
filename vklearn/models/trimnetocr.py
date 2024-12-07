from typing import List, Any, Dict, Mapping, Tuple

from torch import Tensor

import torch
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

        self.project = nn.Sequential(
            nn.Linear(features_dim + merged_dim, features_dim),
            nn.LayerNorm((features_dim, )),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p, inplace=False),
            nn.Linear(features_dim, self.num_classes),
        )

        for m in list(self.trimnetx.trim_units.modules()):
            if not isinstance(m, CBANet): continue
            m.channel_attention = nn.Identity()
            # m.spatial_attention = nn.Identity()

        self._temp_num_scans = self.trimnetx.num_scans

    def train_features(self, flag:bool):
        self.trimnetx.train_features(flag)

    def set_num_scans(self, num_scans:int):
        self._temp_num_scans = num_scans

    def forward(self, x:Tensor) -> Tuple[Tensor, List[Tensor]]:
        hs, f = self.trimnetx(x, self._temp_num_scans)
        fs = [f.squeeze(dim=2).transpose(1, 2)]
        for h in hs:
            h = h.squeeze(dim=2).transpose(1, 2)
            p = self.project(torch.cat([h, fs[-1]], dim=-1))
            fs.append(p + fs[-1])
        x = fs.pop()
        x = self.classifier(x)
        return x, fs

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
            to_gray:    bool=True,
            whitelist:  List[str] | None=None,
        ) -> Dict[str, Any]:

        device = self.get_model_device()
        if to_gray and (image.mode != 'L'):
            image = image.convert('L')
        if image.mode != 'RGB':
            image = image.convert('RGB')
        x = self.preprocess(image, align_size)
        x = x.to(device)
        x, _ = self.forward(x)

        if whitelist is not None:
            white_ixs = [self.categories.index(char) for char in whitelist]
            for ix in range(1, len(self.categories)):
                if ix in white_ixs: continue
                x[..., ix] = -10000

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
            inputs:         Tuple[Tensor, List[Tensor]],
            targets:        Tensor,
            target_lengths: Tensor,
            zero_infinity:  bool=False,
            weights:        Dict[str, float] | None=None,
        ) -> Dict[str, Any]:

        predicts, features = inputs

        losses = super().calc_loss(
            predicts, targets, target_lengths, zero_infinity=zero_infinity)

        weights = weights or dict()
        auxi_weight = weights.get('auxi', 0)
        if auxi_weight == 0: features = []

        loss = losses['loss']
        losses['pf_loss'] = loss

        auxi_loss = 0
        for fid, feature in enumerate(features):
            predicts = self.classifier(feature)
            auxi_i_loss = super().calc_loss(
                predicts, targets, target_lengths, zero_infinity=zero_infinity)['loss']
            auxi_loss = auxi_loss + auxi_i_loss
            losses[f'p{fid}_loss'] = auxi_i_loss

        losses['loss'] = (
            loss * max(0, 1 - len(features) * auxi_weight) +
            auxi_loss * auxi_weight)
        return losses

    def calc_score(
            self,
            inputs:         Tuple[Tensor, List[Tensor]],
            targets:        Tensor,
            target_lengths: Tensor,
        ) -> Dict[str, Any]:
        return super().calc_score(inputs[0], targets, target_lengths)

    def update_metric(
            self,
            inputs:         Tuple[Tensor, List[Tensor]],
            targets:        Tensor,
            target_lengths: Tensor,
        ):
        return super().update_metric(inputs[0], targets, target_lengths)
