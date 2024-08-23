from typing import Any, Tuple
from dataclasses import dataclass

from ..task import Task
from ...models.distiller import Distiller as Model


@dataclass
class Distillation(Task):
    '''This `class` is used to configure a set of parameters relevant to a specific task in distiller model training. 

    Args:
        model: Specify a distillation model object.
        device: Computation device supported by PyTorch.
        metric_start_epoch: Sets the epoch from which metric calculation starts, defaults to 0.
        fit_features_start: Sets the epoch from which feature extractor training starts, -1 means no training, defaults to -1.
        loss_options: Set optional parameters for the distillation model's loss function.
        score_options: Set optional parameters for the distillation model's scoring function.
        metric_options: Set optional parameters for the distillation model's metric evaluation function.
        key_metrics: Specifies which key evaluation metrics to track.
        best_metric: Current best metric score, initialized to 0.
    '''

    model:       Model
    key_metrics: Tuple[str]=('mss', 'mse', 'mae')

    def sample_convert(self, sample: Any) -> Tuple[Any, Any]:
        inputs = sample[0].to(self.device)
        return inputs, [None]
