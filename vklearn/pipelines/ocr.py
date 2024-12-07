from typing import List, Dict, Any, Mapping
import io

from PIL.Image import Image as PILImage
from numpy import ndarray
from PIL import Image
from matplotlib.pyplot import Figure
import torch

from ..models.ocr import OCR as Model


class OCR:
    '''This class is used for handling image ocr tasks. 

    Args:
        model: ocr model.
    '''

    def __init__(self, model:Model):
        self.model = model

    @classmethod
    def load_from_state(
            cls,
            model: Model,
            state: Mapping[str, Any] | str,
        ) -> 'OCR':

        if isinstance(state, str):
            state = torch.load(state, map_location='cpu', weights_only=True)
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

    def to(self, device:torch.device) -> 'OCR':
        self.model.to(device)
        return self

    def __call__(
            self,
            image:      PILImage | str | ndarray,
            top_k:      int=10,
            align_size: int=224,
            to_gray:    bool=True,
            whitelist:  List[str] | None=None,
        ) -> List[Dict[str, Any]]:
        '''Invoke the method for image ocr.

        Args:
            image: The image to be recognized.
            top_k: Specifies the number of top classes to return, sorted by probability in descending order.
            align_size: The size to which the image will be aligned after preprocessing.
            to_gray: Convert the image mode to be gray, default: True.
            whitelist: The whitelist of the characters, default: None is disable the whitelist.
        '''

        if isinstance(image, str):
            image = Image.open(image)
        if isinstance(image, ndarray):
            image = Image.fromarray(image, mode='RGB')

        with torch.no_grad():
            result = self.model.recognize(
                image=image,
                top_k=top_k,
                align_size=align_size,
                to_gray=to_gray,
                whitelist=whitelist,
            )
        return result

    def plot_result(
            self,
            image:      PILImage,
            result:     List[Dict[str, Any]],
            fig:        Figure,
        ):
        '''This method visualizes the model prediction results.

        Args:
            image: The image used for ocr.
            result: The data returned after the model performs the ocr.
            fig: The matplotlib Figure object.
        '''

        text = result['text']
        ax = fig.add_subplot()
        ax.imshow(image)
        ax.set_xlabel(text)
