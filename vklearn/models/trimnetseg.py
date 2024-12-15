from typing import List, Any, Dict, Mapping
import math

from torch import Tensor

import torch
import torch.nn as nn
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
            num_scans:           int | None=None,
            scan_range:          int | None=None,
            backbone:            str | None=None,
            backbone_pretrained: bool | None=None,
        ):
        super().__init__(categories)

        self.trimnetx = TrimNetX(
            num_scans, scan_range, backbone, backbone_pretrained)

        merged_dim = self.trimnetx.merged_dim
        
        self.scale_compensate = self.trimnetx.cell_size / 2**4

        self.predictor = SegPredictor(merged_dim, self.num_classes, self.trimnetx.num_scans)
        self.decoder = nn.Conv2d(self.predictor.embeded_dim, self.num_classes, 1)

    def train_features(self, flag:bool):
        self.trimnetx.train_features(flag)

    def forward_latent(self, x:Tensor) -> List[Tensor]:
        hs, _ = self.trimnetx(x)
        ps = self.predictor(hs)
        return ps

    def forward(self, x:Tensor) -> Tensor:
        ps = self.forward_latent(x)
        ps = [self.decoder(p) for p in ps]
        times = len(ps)
        for t in range(times):
            scale_factor = 2**(3 - t)
            if scale_factor == 1: continue
            ps[t] = F.interpolate(
                ps[t], scale_factor=scale_factor * self.scale_compensate, mode='bilinear')
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
        ps = self.forward_latent(x)
        p = self.decoder(ps[-1])

        scale_factor = 2**(3 - (len(ps) - 1))
        if scale_factor != 1:
            p = F.interpolate(
                p, scale_factor=scale_factor, mode='bilinear')

        src_w, src_h = image.size
        dst_w, dst_h = round(scale * src_w), round(scale * src_h)
        p = torch.sigmoid(p[..., pad_y:pad_y + dst_h, pad_x:pad_x + dst_w])
        p[p < conf_thresh] = 0.
        p = F.interpolate(p, (src_h, src_w), mode='bilinear')
        return p[0].cpu().numpy()

    # def dice_loss(
    #         self,
    #         inputs: Tensor,
    #         target: Tensor,
    #         eps:    float=1e-5,
    #     ) -> Tensor:
    #
    #     predict = torch.sigmoid(inputs).flatten(1)
    #     ground = target.flatten(1)
    #     intersection = predict * ground
    #     dice = (
    #         intersection.sum(dim=1) * 2 /
    #         (predict.sum(dim=1) + ground.sum(dim=1) + eps)
    #     )
    #     dice_loss = 1 - dice
    #     return dice_loss

    def dice_loss(
            self,
            inputs: Tensor,
            target: Tensor,
            smooth: float=1.,
        ) -> Tensor:

        predict = torch.softmax(inputs, dim=1).flatten(2)
        ground = target.flatten(2)
        intersection = predict * ground
        dice = (
            (intersection.sum(dim=2) * 2 + smooth) /
            (predict.sum(dim=2) + ground.sum(dim=2) + smooth)
        ).mean(dim=1)
        dice_loss = 1 - dice
        return dice_loss

    # def calc_loss(
    #         self,
    #         inputs:  Tensor,
    #         target:  Tensor,
    #         weights: Dict[str, float] | None=None,
    #         gamma:   float=0.5,
    #     ) -> Dict[str, Any]:
    #
    #     target = target.type_as(inputs)
    #     times = inputs.shape[-1]
    #     F_alpha = lambda t: (math.cos(t / times * math.pi) + 1) * 0.5
    #     loss = 0.
    #     for t in range(times):
    #         alpha = F_alpha(t)
    #         dice = self.dice_loss(
    #             inputs[..., t],
    #             target,
    #         ).mean()
    #         bce = torch.zeros_like(dice)
    #         if alpha < 1:
    #             bce = F.binary_cross_entropy_with_logits(
    #                 inputs[..., t],
    #                 target,
    #                 reduction='mean',
    #             )
    #         loss_t = alpha * dice + (1 - alpha) * bce
    #         loss = loss + loss_t / times
    #
    #     return dict(
    #         loss=loss,
    #         dice_loss=dice,
    #         bce_loss=bce,
    #     )

    def calc_loss(
            self,
            inputs:  Tensor,
            target:  Tensor,
            weights: Dict[str, float] | None=None,
            gamma:   float=0.5,
            alpha:   float=0.,
        ) -> Dict[str, Any]:

        target = target.type_as(inputs)
        times = inputs.shape[-1]
        F_alpha = lambda t: (math.cos(t / times * math.pi) + 1) * 0.5
        ce_weight = target.transpose(0, 1).flatten(1).sum(1)
        ce_weight = 1 - ce_weight / torch.clamp_min(ce_weight.sum(), 1)
        loss = 0.
        for t in range(times):
            if alpha > 0.:
                alpha_t = alpha
            else:
                alpha_t = F_alpha(t)
            dice = self.dice_loss(
                inputs[..., t],
                target,
            ).mean()
            ce = torch.zeros_like(dice)
            if alpha_t < 1:
                ce = F.cross_entropy(
                    inputs[..., t],
                    target.argmax(dim=1),
                    weight=ce_weight,
                    reduction='mean',
                )
            loss_t = alpha_t * dice + (1 - alpha_t) * ce
            loss = loss + loss_t / times

        return dict(
            loss=loss,
            dice_loss=dice,
            ce_loss=ce,
        )

    # def calc_score(
    #         self,
    #         inputs: Tensor,
    #         target: Tensor,
    #         eps:    float=1e-5,
    #     ) -> Dict[str, Any]:
    #
    #     predicts = torch.sigmoid(inputs[..., -1])
    #     distance = torch.abs(predicts - target).mean(dim=(2, 3)).mean()
    #
    #     return dict(
    #         mae=distance,
    #     )

    def calc_score(
            self,
            inputs: Tensor,
            target: Tensor,
            eps:    float=1e-5,
        ) -> Dict[str, Any]:

        predicts = torch.softmax(inputs[..., -1], dim=1)
        distance = torch.abs(predicts - target).mean(dim=(2, 3)).mean()

        return dict(
            mae=distance,
        )

    # def update_metric(
    #         self,
    #         inputs:      Tensor,
    #         target:      Tensor,
    #         conf_thresh: float=0.5,
    #     ):
    #
    #     predicts = torch.sigmoid(inputs[..., -1]) > conf_thresh
    #     self.m_iou_metric.update(predicts.to(torch.int), target.to(torch.int))

    def update_metric(
            self,
            inputs:      Tensor,
            target:      Tensor,
            conf_thresh: float=0.5,
        ):

        predicts = F.one_hot(inputs[..., -1].argmax(dim=1)).permute(0, 3, 1, 2)
        self.m_iou_metric.update(predicts.to(torch.int), target.to(torch.int))
