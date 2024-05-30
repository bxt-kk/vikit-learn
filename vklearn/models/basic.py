from typing import Any, Dict, Mapping, Tuple, List
import math

from torch import Tensor
import torch
import torch.nn as nn
from torchvision.transforms import v2
from PIL import Image


class Basic(nn.Module):

    def __init__(self):
        super().__init__()
        self._keep_features = False

    @classmethod
    def get_transforms(
            cls,
            task_name: str='default',
        ) -> Tuple[v2.Transform, v2.Transform]:
        assert not 'this is an empty func'

    @classmethod
    def load_from_state(cls, state:Mapping[str, Any]) -> 'Basic':
        assert not 'this is an empty func'

    def hyperparameters(self) -> Dict[str, Any]:
        assert not 'this is an empty func'

    def collate_fn(
            self,
            batch: List[Any],
        ) -> Any:
        assert not 'this is an empty func'

    def preprocess(
            self,
            image:      Image.Image,
            align_size: int,
            limit_size: int,
            fill_value: int,
        ) -> Tuple[Tensor, float, int, int]:

        src_w, src_h = image.size
        _align_size = math.ceil(align_size / limit_size) * limit_size
        scale = _align_size / max(src_w, src_h)
        dst_w, dst_h = round(scale * src_w), round(scale * src_h)
        sample = image.resize(
            size=(dst_w, dst_h),
            resample=Image.Resampling.BILINEAR)
        frame = Image.new(
            mode='RGB',
            size=(_align_size, _align_size),
            color=(fill_value, ) * 3)
        pad_x = (align_size - dst_w) // 2
        pad_y = (align_size - dst_h) // 2
        frame.paste(sample, box=(pad_x, pad_y))
        inputs = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])(frame).unsqueeze(dim=0)
        return inputs, scale, pad_x, pad_y

    def train_features(self, flag:bool):
        self._keep_features = not flag
