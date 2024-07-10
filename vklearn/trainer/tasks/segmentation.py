from typing import Any, Tuple
from dataclasses import dataclass

from ..task import Task
from ...models.segment import Segment as Model


@dataclass
class Segmentation(Task):
    model:       Model
    key_metrics: Tuple[str]=('miou', )

    def sample_convert(self, sample: Any) -> Tuple[Any, Any]:
        inputs, target = [item.to(self.device) for item in sample]
        return inputs, [target]
