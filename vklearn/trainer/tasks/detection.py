from typing import Any, Tuple
from dataclasses import dataclass

from ..task import Task
from ...models.detector import Detector as Model


@dataclass
class Detection(Task):
    '''This `class` is used to configure a set of parameters relevant to a specific task in object detector model training. 

    Args:
        model: Specify a object detection model object.
        device: Computation device supported by PyTorch.
        metric_start_epoch: Sets the epoch from which metric calculation starts, defaults to 0.
        fit_features_start: Sets the epoch from which feature extractor training starts, -1 means no training, defaults to -1.
        loss_options: Set optional parameters for the object detection model's loss function.
        score_options: Set optional parameters for the object detection model's scoring function.
        metric_options: Set optional parameters for the object detection model's metric evaluation function.
        key_metrics: Specifies which key evaluation metrics to track.
        best_metric: Current best metric score, initialized to 0.
    '''

    model:       Model
    key_metrics: Tuple[str]=('map', 'map_50', 'map_75')

    def sample_convert(self, sample: Any) -> Tuple[Any, Any]:
        inputs, target_labels, target_bboxes = [
            sample[i].to(self.device) for i in [0, 2, 3]]
        target_index = sample[1]
        target = target_index, target_labels, target_bboxes
        return inputs, target
