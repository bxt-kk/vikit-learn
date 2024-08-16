from typing import Any, Tuple
from dataclasses import dataclass

from ..task import Task
from ...models.joints import Joints as Model


@dataclass
class Joints(Task):
    model:       Model
    key_metrics: Tuple[str]=('mjoin', 'map', 'map_50', 'map_75', 'miou')

    def sample_convert(self, sample: Any) -> Tuple[Any, Any]:
        inputs, target_labels, target_bboxes, target_masks = [
            sample[i].to(self.device) for i in [0, 2, 3, 4]]
        target_index = sample[1]
        target = target_index, target_labels, target_bboxes, target_masks
        return inputs, target
