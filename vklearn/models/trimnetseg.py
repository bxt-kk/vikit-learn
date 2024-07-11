from typing import List, Any, Dict, Mapping
import math

from torch import Tensor

import torch
import torch.nn.functional as F

from PIL import Image

from .segment import Segment
from .trimnetx import TrimNetX
from .component import SegPredictor


class TrimNetSeg(Segment):
    '''A light-weight and easy-to-train model for image segmentation

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
            backbone:            str='mobilenet_v3_small',
            backbone_pretrained: bool=True,
        ):
        super().__init__(categories)

        self.trimnetx = TrimNetX(
            num_scans, scan_range, backbone, backbone_pretrained)

        merged_dim = self.trimnetx.merged_dim

        # self.predict = SegPredictor(merged_dim, self.num_classes, upscale_factor=16)
        self.predict = SegPredictor(
            merged_dim, self.num_classes, num_layers=4)

    def train_features(self, flag:bool):
        self.trimnetx.train_features(flag)

    def forward(self, x:Tensor) -> Tensor:
        hs = self.trimnetx(x)
        p = self.predict(hs[0])
        ps = [p]
        times = len(hs)
        for t in range(1, times):
            a = torch.sigmoid(p)
            p = self.predict(hs[t]) * a + p * (1 - a)
            ps.append(p)
        return torch.cat([p[..., None] for p in ps], dim=-1)

    @classmethod
    def load_from_state(cls, state:Mapping[str, Any]) -> 'TrimNetSeg':
        hyps = state['hyperparameters']
        model = cls(
            categories          = hyps['categories'],
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
            num_scans  = self.trimnetx.num_scans,
            scan_range = self.trimnetx.scan_range,
            backbone   = self.trimnetx.backbone,
        )

    def segment(
            self,
            image:       Image.Image,
            conf_thresh: float=0.5,
            align_size:  int=448,
        ) -> List[Dict[str, Any]]:

        device = self.get_model_device()
        x, scale, pad_x, pad_y = self.preprocess(
            image, align_size, limit_size=32, fill_value=127)
        x = x.to(device)
        x = self.forward(x)
        src_w, src_h = image.size
        dst_w, dst_h = round(scale * src_w), round(scale * src_h)
        x = torch.sigmoid(x[..., pad_y:pad_y + dst_h, pad_x:pad_x + dst_w, -1])
        x[x < conf_thresh] = 0.
        x = F.interpolate(x, (src_h, src_w), mode='bilinear')
        return x[0].cpu().numpy()

    def dice_loss(
            self,
            inputs: Tensor,
            target: Tensor,
            eps:    float=1e-5,
        ) -> Tensor:

        predict = torch.sigmoid(inputs).flatten(1)
        ground = target.flatten(1)
        intersection = predict * ground
        dice = (
            intersection.sum(dim=1) * 2 /
            (predict.sum(dim=1) + ground.sum(dim=1) + eps)
        )
        dice_loss = 1 - dice
        return dice_loss

    def calc_loss(
            self,
            inputs:  Tensor,
            target:  Tensor,
            weights: Dict[str, float] | None=None,
        ) -> Dict[str, Any]:

        # reduction = 'mean'

        times = inputs.shape[-1]
        F_sigma = lambda t: 1 - (math.cos((t + 1) / times * math.pi) + 1) * 0.5
        target = target.type_as(inputs)

        # alpha = (target.mean(dim=(1, 2, 3))**0.5 + 1) * 0.5
        alpha = (target.mean(dim=(1, 2, 3)) + 1) * 0.5
        grand_sigma = 0.

        loss = 0.
        for t in range(times):

            sigma = F_sigma(t)
            grand_sigma += sigma
            bce = F.binary_cross_entropy_with_logits(
                inputs[..., t],
                target,
                reduction='none',
            ).mean(dim=(1, 2, 3))
            dice = self.dice_loss(
                inputs[..., t],
                target)
            loss_t = alpha * bce + (1 - alpha) * dice
            loss = loss + loss_t.mean() * sigma

        #     sigma = F_sigma(t)
        #     loss = loss + F.binary_cross_entropy_with_logits(
        #         inputs[..., t],
        #         target,
        #         reduction=reduction,
        #     ) * sigma
        #     if sigma < 1:
        #         loss = loss + self.dice_loss(
        #             inputs[..., t],
        #             target,
        #         ) * (1 - sigma)
        # loss = loss / times
        loss = loss / grand_sigma

        return dict(
            loss=loss,
            alpha=alpha.mean(),
        )

    def calc_score(
            self,
            inputs: Tensor,
            target: Tensor,
            eps:    float=1e-5,
        ) -> Dict[str, Any]:

        predicts = torch.sigmoid(inputs[..., -1])
        distance = torch.abs(predicts - target).mean(dim=(2, 3)).mean()

        return dict(
            mae=distance,
        )

    def update_metric(
            self,
            inputs:      Tensor,
            target:      Tensor,
            conf_thresh: float=0.5,
        ):

        predicts = torch.sigmoid(inputs[..., -1]) > conf_thresh
        self.m_iou.update(predicts.to(torch.int), target.to(torch.int))
