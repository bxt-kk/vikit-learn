from typing import List, Tuple
import math

from torch import Tensor
import torch
import torch.nn.functional as F

from torchvision.ops import (
    sigmoid_focal_loss,
)


def focal_boost_iter(
        inputs:       Tensor,
        target_index: List[Tensor],
        sample_mask:  Tensor | None,
        conf_id:      int,
        num_confs:    int,
        alpha:        float,
        gamma:        float,
    ) -> Tuple[Tensor, Tensor, Tensor]:

    reduction = 'mean'

    pred_conf = inputs[..., conf_id]
    targ_conf = torch.zeros_like(pred_conf)
    targ_conf[target_index] = 1.

    if sample_mask is None:
        sample_mask = targ_conf >= -1

    sampled_pred = torch.masked_select(pred_conf, sample_mask)
    sampled_targ = torch.masked_select(targ_conf, sample_mask)
    sampled_loss = sigmoid_focal_loss(
        inputs=sampled_pred,
        targets=sampled_targ,
        alpha=alpha,
        gamma=gamma,
        reduction=reduction,
    )

    obj_loss = 0.
    obj_mask = targ_conf > 0.5
    if obj_mask.sum() > 0:
        # Lab code <<<
        # obj_pred = torch.masked_select(pred_conf, obj_mask)
        # obj_targ = torch.masked_select(targ_conf, obj_mask)
        # obj_loss = F.binary_cross_entropy_with_logits(
        #     obj_pred, obj_targ, reduction=reduction)
        # instance_weight = (
        #     1 / torch.clamp_min(targ_conf.flatten(start_dim=1).sum(dim=1), 1)
        # )[target_index[0]]
        instance_weight = 1 / target_index[0].bincount().type_as(targ_conf)[target_index[0]]
        obj_pred = pred_conf[target_index]
        obj_targ = targ_conf[target_index]
        obj_loss = (instance_weight * F.binary_cross_entropy_with_logits(
            obj_pred, obj_targ, reduction='none')).sum() / inputs.shape[0]
        # >>>

        obj_pred_min = obj_pred.detach().min()
        sample_mask = pred_conf.detach() >= obj_pred_min

    F_sigma = lambda t: (math.cos(t / num_confs * math.pi) + 1) * 0.5
    sigma_0 = 0.25

    sigma_T = F_sigma(num_confs - 1)
    scale_r = (1 - sigma_0) / (1 - sigma_T)
    sigma_k = F_sigma(conf_id) * scale_r - scale_r + 1
    # num_foreground_per_img = (sampled_targ.sum() / len(pred_conf)).item()
    sn_ratio = min(1, 2 * targ_conf.sum().item() / targ_conf.numel())
    an_ratio = 1 - sn_ratio

    conf_loss = (
        # obj_loss * sigma_k / max(1, num_foreground_per_img**0.5) +
        obj_loss * sigma_k * an_ratio +
        sampled_loss) / num_confs

    return conf_loss, sampled_loss, sample_mask


def focal_boost_loss(
        inputs:       Tensor,
        target_index: List[Tensor],
        num_confs:    int,
        alpha:        float=0.25,
        gamma:        float=2.,
    ) -> Tuple[Tensor, Tensor]:

    conf_loss, sampled_loss, sample_mask = focal_boost_iter(
        inputs, target_index, None, 0, num_confs, alpha, gamma)
    for conf_id in range(1, num_confs):
        conf_loss_i, sampled_loss, sample_mask = focal_boost_iter(
            inputs, target_index, sample_mask, conf_id, num_confs, alpha, gamma)
        conf_loss += conf_loss_i
    return conf_loss, sampled_loss


def focal_boost_predict(
        inputs:        Tensor,
        num_confs:     int,
        recall_thresh: float,
    ) -> Tensor:

    predict = torch.ones_like(inputs[..., 0])
    for conf_id in range(num_confs - 1):
        predict[torch.sigmoid(inputs[..., conf_id]) < recall_thresh] = 0.
    predict = predict * torch.sigmoid(inputs[..., num_confs - 1])
    return predict


def focal_boost_positive(
        inputs:        Tensor,
        num_confs:     int,
        conf_thresh:   float=0.5,
        recall_thresh: float=0.5,
        top_k:         int=0,
    ) -> Tensor:

    predict = focal_boost_predict(inputs, num_confs, recall_thresh)
    if top_k <= 0:
        return predict >= conf_thresh
    samples = predict.flatten(start_dim=1)
    top_k = min(samples.shape[1], top_k)
    conf_threshes = torch.clamp_min(
        samples.topk(top_k).values[:, -1], conf_thresh)
    for _ in range(len(predict.shape) - 1):
        conf_threshes.unsqueeze_(dim=-1)
    return predict >= conf_threshes
