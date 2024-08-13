from typing import List, Any, Dict, Mapping
import math

from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from .segment import Segment
from .trimnetx import TrimNetX
# from .component import UpSample, ConvNormActive
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

        # embeded_dim = int((self.num_classes + 2)**0.5)
        # self.decoder = nn.Conv2d(embeded_dim, self.num_classes, 1)
        # self.upsamples = nn.ModuleDict()
        # self.predicts = nn.ModuleList()
        # for t in range(num_scans):
        #     in_planes = merged_dim
        #     out_planes = max(in_planes // 2, embeded_dim)
        #     if t > 0:
        #         self.upsamples[f'{t}'] = nn.Sequential(
        #             UpSample(in_planes),
        #             ConvNormActive(in_planes, out_planes, 1),
        #         )
        #         in_planes = out_planes + embeded_dim
        #         out_planes = max(out_planes // 2, embeded_dim)
        #     for k in range(t - 1):
        #         self.upsamples[f'{t}_{k}'] = nn.Sequential(
        #             UpSample(in_planes),
        #             ConvNormActive(in_planes, out_planes, 1),
        #         )
        #         in_planes = out_planes + embeded_dim
        #         out_planes = max(out_planes // 2, embeded_dim)
        #     self.predicts.append(nn.Sequential(
        #         UpSample(in_planes),
        #         ConvNormActive(in_planes, out_planes, 1),
        #         ConvNormActive(out_planes, out_planes, groups=out_planes),
        #         ConvNormActive(out_planes, embeded_dim, kernel_size=1),
        #     ))
        self.predictor = SegPredictor(merged_dim, self.num_classes, num_scans)
        self.decoder = nn.Conv2d(self.predictor.embeded_dim, self.num_classes, 1)

    def train_features(self, flag:bool):
        self.trimnetx.train_features(flag)

    def forward_latent(self, x:Tensor) -> List[Tensor]:
        hs, _ = self.trimnetx(x)
        # pt = self.predicts[0](hs[0])
        # ps = [pt]
        # times = len(hs)
        # for t in range(1, times):
        #     u = self.upsamples[f'{t}'](hs[t])
        #     for k in range(t - 1):
        #         u = self.upsamples[f'{t}_{k}'](torch.cat([
        #             u, ps[k]], dim=1))
        #     pt = (
        #         self.predicts[t](torch.cat([u, pt], dim=1)) +
        #         F.interpolate(pt, scale_factor=2, mode='bilinear')
        #     )
        #     ps.append(pt)
        ps = self.predictor(hs)
        return ps

    def forward(self, x:Tensor) -> Tensor:
        ps = self.forward_latent(x)
        ps = [self.decoder(p) for p in ps]
        times = len(ps)
        for t in range(times):
            scale_factor = 2**(3 - t)
            if scale_factor == 1: continue
            ps[t] = F.interpolate(ps[t], scale_factor=scale_factor, mode='bilinear')
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
            gamma:   float=0.5,
        ) -> Dict[str, Any]:

        target = target.type_as(inputs)
        times = inputs.shape[-1]
        F_alpha = lambda t: (math.cos(t / times * math.pi) + 1) * 0.5
        loss = 0.
        for t in range(times):
            alpha = F_alpha(t)
            dice = self.dice_loss(
                inputs[..., t],
                target,
            ).mean()
            bce = torch.zeros_like(dice)
            if alpha < 1:
                bce = F.binary_cross_entropy_with_logits(
                    inputs[..., t],
                    target,
                    reduction='mean',
                )
            loss_t = alpha * dice + (1 - alpha) * bce
            loss = loss + loss_t / times

        return dict(
            loss=loss,
            dice_loss=dice,
            bce_loss=bce,
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
