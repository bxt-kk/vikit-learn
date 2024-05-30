from typing import List, Dict, Any, Mapping
import io

import torch
from PIL.Image import Image as PILImage
from numpy import ndarray
from PIL import Image

from ..models.classifier import Classifier as Model


class Classifier:

    def __init__(self, model:Model):
        self.model = model

    @classmethod
    def load_from_state(
            cls,
            model: Model,
            state: Mapping[str, Any] | str,
        ) -> 'Classifier':

        if isinstance(state, str):
            state = torch.load(state, map_location='cpu')
        return cls(model.load_from_state(state).eval())

    def export_onnx(self, f: str | io.BytesIO):
        inputs = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            model=self.model,
            args=inputs,
            f=f,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size' },
                'output': {0: 'batch_size'},
            },
        )

    def to(self, device:torch.device) -> 'Classifier':
        self.model.to(device)
        return self

    def __call__(
            self,
            image:      PILImage | str | ndarray,
            top_k:      int=10,
            align_size: int=224,
        ) -> List[Dict[str, Any]]:

        if isinstance(image, str):
            image = Image.open(image)
        if isinstance(image, ndarray):
            image = Image.fromarray(image, mode='RGB')

        with torch.no_grad():
            result = self.model.classify(
                image=image,
                top_k=top_k,
                align_size=align_size,
            )
        return result
