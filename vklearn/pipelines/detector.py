from typing import List, Dict, Any, Mapping
import io

import torch
from PIL.Image import Image as PILImage
from numpy import ndarray
from PIL import Image
from matplotlib.pyplot import Figure, Rectangle

from ..models.detector import Detector as Model


class Detector:

    def __init__(self, model:Model):
        self.model = model

    @classmethod
    def load_from_state(
            cls,
            model: Model,
            state: Mapping[str, Any] | str,
        ) -> 'Detector':

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

    def to(self, device:torch.device) -> 'Detector':
        self.model.to(device)
        return self

    def __call__(
            self,
            image:       PILImage | str | ndarray,
            conf_thresh: float=0.6,
            iou_thresh:  float=0.55,
            align_size:  int=448,
            mini_side:   int=1,
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
                mini_side=mini_side,
            )
        return result

    def plot_result(
            self,
            image:      PILImage,
            result:     List[Dict[str, Any]],
            fig:        Figure,
            color:      str='red',
            text_color: str='white',
        ):

        ax = fig.add_subplot()
        ax.imshow(image)
        for obj in result:
            x1, y1, x2, y2 = obj['box']
            ax.add_patch(Rectangle(
                (x1, y1), x2 - x1, y2 - y1, color=color, fill=False))
            ax.annotate(
                obj['label'],
                (x1, y1),
                color=text_color,
                ha='left',
                va='bottom',
                bbox=dict(
                    color=color,
                    pad=0,
                ),
            )
