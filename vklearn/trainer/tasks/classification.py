from typing import Any, Tuple
from dataclasses import dataclass

from ..task import Task
from ...models.classifier import Classifier as Model


@dataclass
class Classification(Task):
    '''This `class` is used to configure a set of parameters relevant to a specific task in classifier model training. 

    Args:
        model: Specify a classification model object.
        device: Computation device supported by PyTorch.
        metric_start_epoch: Sets the epoch from which metric calculation starts, defaults to 0.
        fit_features_start: Sets the epoch from which feature extractor training starts, -1 means no training, defaults to -1.
        loss_options: Set optional parameters for the classification model's loss function.
        score_options: Set optional parameters for the classification model's scoring function.
        metric_options: Set optional parameters for the classification model's metric evaluation function.
        key_metrics: Specifies which key evaluation metrics to track.
        best_metric: Current best metric score, initialized to 0.
    '''

    model:       Model
    key_metrics: Tuple[str]=('f1_score', 'precision', 'recall')

    def sample_convert(self, sample: Any) -> Tuple[Any, Any]:
        inputs, target = [item.to(self.device) for item in sample]
        return inputs, [target]
