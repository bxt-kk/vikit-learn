from typing import Any, Tuple
from dataclasses import dataclass

from ..task import Task
from ...models.ocr import OCR as Model


@dataclass
class OCR(Task):
    '''This `class` is used to configure a set of parameters relevant to a specific task in ocr model training. 

    Args:
        model: Specify a ocr model object.
        device: Computation device supported by PyTorch.
        metric_start_epoch: Sets the epoch from which metric calculation starts, defaults to 0.
        fit_features_start: Sets the epoch from which feature extractor training starts, -1 means no training, defaults to -1.
        loss_options: Set optional parameters for the ocr model's loss function.
        score_options: Set optional parameters for the ocr model's scoring function.
        metric_options: Set optional parameters for the ocr model's metric evaluation function.
        key_metrics: Specifies which key evaluation metrics to track.
        best_metric: Current best metric score, initialized to 0.
    '''

    model:       Model
    key_metrics: Tuple[str]=('c_score', 'cer')

    def sample_convert(self, sample: Any) -> Tuple[Any, Any]:
        inputs, targets, input_lengths, target_lengths = [item.to(self.device) for item in sample]
        return inputs, (targets, input_lengths, target_lengths)
