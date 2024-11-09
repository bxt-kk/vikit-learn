from typing import List, Dict, Any, Mapping
import io

import torch
from PIL.Image import Image as PILImage
from numpy import ndarray
from PIL import Image
from matplotlib.pyplot import Figure, Rectangle

from ..models.detector import Detector as Model


class Detector:
    '''This class is used for handling object detection tasks. 

    Args:
        model: Object detector model.
    '''

    def __init__(self, model:Model):
        self.model = model

    @classmethod
    def load_from_state(
            cls,
            model: Model,
            state: Mapping[str, Any] | str,
        ) -> 'Detector':

        if isinstance(state, str):
            state = torch.load(state, map_location='cpu', weights_only=True)
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
            image:         PILImage | str | ndarray,
            conf_thresh:   float=0.5,
            recall_thresh: float=0.5,
            iou_thresh:    float=0.5,
            align_size:    int=448,
            mini_side:     int=1,
        ) -> List[Dict[str, Any]]:
        '''Invoke the method for object detection.

        Args:
            image: The image to be detected.
            conf_thresh: Confidence threshold.
            recall_thresh: Recall score threshold.
            iou_thresh: Intersection over union threshold.
            align_size: The size to which the image will be aligned after preprocessing.
            mini_side: Minimum bounding box side length.
        '''

        if isinstance(image, str):
            image = Image.open(image)
        if isinstance(image, ndarray):
            image = Image.fromarray(image, mode='RGB')

        with torch.no_grad():
            result = self.model.detect(
                image=image,
                conf_thresh=conf_thresh,
                recall_thresh=recall_thresh,
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
        '''This method visualizes the model prediction results.

        Args:
            image: The image used for classification.
            result: The data returned after the model performs the detection.
            fig: The matplotlib Figure object.
            color: The color of annotates.
            text_color: The color of the label text.
        '''

        ax = fig.add_subplot()
        ax.imshow(image)
        for obj in result:
            x1, y1, x2, y2 = obj['box']
            ax.add_patch(Rectangle(
                (x1, y1), x2 - x1, y2 - y1, color=color, fill=False))
            ax.annotate(
                f"{obj['label']}: {round(obj['score'], 3)}",
                (x1, y1),
                color=text_color,
                ha='left',
                va='bottom',
                bbox=dict(
                    color=color,
                    pad=0,
                ),
            )
