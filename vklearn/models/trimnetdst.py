from typing import Any, Dict, Mapping

import torch.nn as nn
import torch.nn.functional as F

from .distiller import Distiller
from .component import MobileNetFeatures, DinoFeatures, ConvNormActive


class TrimNetDst(Distiller):
    '''A light-weight and easy-to-train model for knowledge distillation

    Args:
        teacher_arch: The architecture name of the teacher model.
        student_arch: The architecture name of the student model.
        pretrained: Whether to load student model pretrained weights.
    '''

    def __init__(
            self,
            teacher_arch: str='dinov2_vits14',
            student_arch: str='mobilenet_v3_small',
            pretrained:   bool=True,
        ):

        if teacher_arch.startswith('mobilenet'):
            teacher = MobileNetFeatures(teacher_arch, pretrained=True)
        elif teacher_arch.startswith('dinov2'):
            teacher = DinoFeatures(teacher_arch)
        else:
            raise ValueError(f'Unsupported arch `{teacher_arch}`')

        student = MobileNetFeatures(student_arch, pretrained)

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

        self.teacher_arch = teacher_arch
        self.student_arch = student_arch

    @classmethod
    def load_from_state(cls, state:Mapping[str, Any]) -> 'TrimNetDst':
        hyps = state['hyperparameters']
        model = cls(
            teacher_arch = hyps['teacher_arch'],
            student_arch = hyps['student_arch'],
        )
        model.load_state_dict(state['model'])
        return model

    def hyperparameters(self) -> Dict[str, Any]:
        return dict(
            teacher_arch = self.teacher_arch,
            student_arch = self.student_arch,
        )
