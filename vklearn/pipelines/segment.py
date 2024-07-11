from typing import Any, Mapping
import io

from PIL.Image import Image as PILImage
from numpy import ndarray
from PIL import Image
from matplotlib.pyplot import Figure
import torch
import numpy as np

from ..models.segment import Segment as Model


class Segment:

    def __init__(self, model:Model):
        self.model = model

    @classmethod
    def load_from_state(
            cls,
            model: Model,
            state: Mapping[str, Any] | str,
        ) -> 'Segment':

        if isinstance(state, str):
            state = torch.load(state, map_location='cpu')
        return cls(model.load_from_state(state).eval())

    def export_onnx(self, f: str | io.BytesIO):
        inputs = torch.randn(1, 3, 448, 448)
        torch.onnx.export(
            model=self.model,
            args=inputs,
            f=f,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {
                    0: 'batch_size',
                    2: 'input_height',
                    3: 'input_width'},
                'output': {
                    0: 'batch_size',
                    2: 'output_height',
                    3: 'output_width'},
            },
        )

    def to(self, device:torch.device) -> 'Segment':
        self.model.to(device)
        return self

    def __call__(
            self,
            image:       PILImage | str | ndarray,
            conf_thresh: float=0.6,
            align_size:  int=448,
        ) -> ndarray:

        if isinstance(image, str):
            image = Image.open(image)
        if isinstance(image, ndarray):
            image = Image.fromarray(image, mode='RGB')

        with torch.no_grad():
            result = self.model.segment(
                image=image,
                conf_thresh=conf_thresh,
                align_size=align_size,
            )
        return result

    def plot_result(
            self,
            image:      PILImage,
            result:     ndarray,
            fig:        Figure,
        ):

        plot_cols = len(result)
        for i in range(plot_cols):
            ax = fig.add_subplot(1, plot_cols, i + 1)
            mask = 1 - result[i]
            frame = np.array(image, dtype=np.uint8)
            frame[..., 1] = (frame[..., 1] * mask).astype(np.uint8)
            ax.imshow(frame)
