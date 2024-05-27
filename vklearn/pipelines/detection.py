from typing import List, Dict, Any, Mapping

import torch
from PIL.Image import Image as PILImage
from numpy import ndarray
from PIL import Image

from ..models.detection import Detection as Model


class Detection:

    def __init__(self, model:Model):
        self.model = model

    @classmethod
    def load_from_state(
            cls,
            model: Model,
            state: Mapping[str, Any] | str,
        ) -> 'Detection':

        if isinstance(state, str):
            state = torch.load(state, map_location='cpu')
        return cls(model.load_from_state(state).eval())

    def to(self, device:torch.device) -> 'Detection':
        self.model.to(device)
        return self

    def __call__(
            self,
            image:       PILImage | str | ndarray,
            conf_thresh: float=0.6,
            iou_thresh:  float=0.55,
            align_size:  int=448,
        ) -> List[Dict[str, Any]]:

        if isinstance(image, str):
            image = Image.open(image)
        if isinstance(image, ndarray):
            image = Image.fromarray(image, mode='RGB')

        with torch.no_grad():
            result = self.model.detect(
                image=image,
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh,
                align_size=align_size,
            )
        return result
