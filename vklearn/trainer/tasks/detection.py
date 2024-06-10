from typing import Any, Tuple
from dataclasses import dataclass

from ..task import Task
from ...models.detector import Detector as Model


@dataclass
class Detection(Task):
    model:       Model
    key_metrics: Tuple[str]=('map', 'map_50', 'map_75')

    def sample_convert(self, sample: Any) -> Tuple[Any, Any]:
        inputs, target_labels, target_bboxes = [
            sample[i].to(self.device) for i in [0, 2, 3]]
        target_index = sample[1]
        target = target_index, target_labels, target_bboxes
        return inputs, target
