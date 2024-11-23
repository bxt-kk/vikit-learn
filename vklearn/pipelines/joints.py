from typing import List, Dict, Any, Mapping, Sequence, Tuple
import io

import torch
from PIL.Image import Image as PILImage
from PIL import Image
from numpy import ndarray
import cv2 as cv
from matplotlib.pyplot import Figure, Circle, Polygon

from ..models.joints import Joints as Model


class Joints:
    '''This class is used for handling keypoint&joint detection tasks. 

    Args:
        model: Keypoint&joint detection model.
    '''

    def __init__(self, model:Model):
        self.model = model

    @classmethod
    def load_from_state(
            cls,
            model: Model,
            state: Mapping[str, Any] | str,
        ) -> 'Joints':

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

    def to(self, device:torch.device) -> 'Joints':
        self.model.to(device)
        return self

    def __call__(
            self,
            image:        PILImage | str | ndarray,
            joints_type:  str='normal',
            conf_thresh:  float=0.5,
            iou_thresh:   float=0.5,
            align_size:   int=448,
            score_thresh: float=0.5,
            ocr_params:   Sequence[Tuple[float, int]]=((0.7, 7), (0.9, 5)),
        ) -> List[Dict[str, Any]]:
        '''Invoke the method for keypoint&joint detection.

        Args:
            image: The image to be detected.
            joints_type: The type of joints operation.
            conf_thresh: Confidence threshold.
            iou_thresh: Intersection over union threshold.
            align_size: The size to which the image will be aligned after preprocessing.
        '''

        if isinstance(image, str):
            image = Image.open(image)
        if isinstance(image, ndarray):
            image = Image.fromarray(image, mode='RGB')

        with torch.no_grad():
            result = self.model.detect(
                image=image,
                joints_type=joints_type,
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh,
                align_size=align_size,
                score_thresh=score_thresh,
                ocr_params=ocr_params,
            )
        return result

    def plot_result(
            self,
            image:         PILImage,
            result:        List[Dict[str, Any]],
            fig:           Figure,
            show_annotate: bool=True,
            show_nodes:    bool=False,
            show_heatmap:  bool=False,
        ):
        '''This method visualizes the model prediction results.

        Args:
            image: The image used for keypoint&joint detection.
            result: The data returned after the model performs the detection.
            fig: The matplotlib Figure object.
        '''

        if show_heatmap:
            ax = fig.add_subplot(1, 2, 1)
            fig.add_subplot(1, 2, 2).imshow(result['heatmap'])
        else:
            ax = fig.add_subplot()
        ax.imshow(image)
        for obj in result['objs']:
            rect = obj['rect']
            pts = cv.boxPoints(rect)
            ax.add_patch(Polygon(pts, closed=True, fill=False, color='red'))
            if not show_annotate: continue
            ax.annotate(
                f"{obj['label']}: {round(obj['score'], 3)}",
                (pts[1] + pts[2]) * 0.5,
                color='white',
                ha='center',
                va='center',
                rotation=-rect[-1],
                bbox=dict(
                    boxstyle='rarrow',
                    color='red',
                    pad=0,
                ),
            )
        if show_nodes:
            for node in result['nodes']:
                x1, y1, x2, y2 = node['box']
                xy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
                radius = min(x2 - x1, y2 - y1) * 0.25
                color = 'blue'
                fill = True
                alpha = 0.5
                if node['anchor'] == 1: color = 'yellow'
                ax.add_patch(Circle(
                    xy, radius, color=color, fill=fill, linewidth=1, alpha=alpha))
