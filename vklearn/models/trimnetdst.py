from typing import Any, Dict, Mapping

# from torch import Tensor

# import torch
import torch.nn as nn
import torch.nn.functional as F

# from PIL import Image

from .distiller import Distiller
from .component import MobileNetFeatures, DinoFeatures, ConvNormActive


class TrimNetDst(Distiller):
    '''A light-weight and easy-to-train model for knowledge distillation

    Args:
        teacher: Teacher model object.
        num_scans: Number of the Trim-Units.
        scan_range: Range factor of the Trim-Unit convolution.
        backbone: Specify a basic model as a feature extraction module.
        backbone_pretrained: Whether to load backbone pretrained weights.
    '''

    def __init__(
            self,
            teacher:    MobileNetFeatures | DinoFeatures,
            arch:       str='mobilenet_v3_small',
            pretrained: bool=True,
        ):

        student = MobileNetFeatures(arch, pretrained)

        in_transform = None
        if teacher.cell_size != student.cell_size:
            scale_factor = teacher.cell_size / student.cell_size
            in_transform = lambda x: F.interpolate(
                x, scale_factor=scale_factor, mode='bilinear')

        in_planes, out_planes = student.features_dim, teacher.features_dim
        out_project = nn.Sequential(
            ConvNormActive(in_planes, in_planes, 3, groups=in_planes),
            ConvNormActive(in_planes, out_planes, 1, norm_layer=None, activation=None),
        )

        super().__init__(teacher, student, in_transform, out_project)

    @classmethod
    def load_from_state(cls, state:Mapping[str, Any]) -> 'TrimNetDst':
        hyps = state['hyperparameters']
        model = cls(
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
