from typing import Any, Tuple
from dataclasses import dataclass

from ..task import Task
from ...models.classifier import Classifier as Model


@dataclass
class Classification(Task):
    model:       Model
    key_metrics: Tuple[str]=['f1_score', 'precision', 'recall']

    def sample_convert(self, sample: Any) -> Tuple[Any, Any]:
        inputs, target = [item.to(self.device) for item in sample]
        return inputs, [target]
