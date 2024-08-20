## Task

> This `class` is used to configure a set of parameters relevant to a specific task in model training. 

Args:

- `model`: Model object for a specific task.
- `device`: Computation device supported by PyTorch.
- `metric_start_epoch`: Sets the epoch from which metric calculation starts, defaults to 0.
- `fit_features_start`: Sets the epoch from which feature extractor training starts, -1 means no training, defaults to -1.
- `loss_options`: Optional parameters for the given loss calculation function.
- `score_options`: Optional parameters for the given score calculation function.
- `metric_options`: Optional parameters for the given metric evaluation function.
- `key_metrics`: Specifies which key evaluation metrics to track.
- `best_metric`: Current best metric score, initialized to 0.

## Classification(Task)

> This `class` is used to configure a set of parameters relevant to a specific task in classifier model training. 

Args:

- `model`: Specify a classification model object.
- `device`: Computation device supported by PyTorch.
- `metric_start_epoch`: Sets the epoch from which metric calculation starts, defaults to 0.
- `fit_features_start`: Sets the epoch from which feature extractor training starts, -1 means no training, defaults to -1.
- `loss_options`: Set optional parameters for the classification model's loss function.
- `score_options`: Set optional parameters for the classification model's scoring function.
- `metric_options`: Set optional parameters for the classification model's metric evaluation function.
- `key_metrics`: Specifies which key evaluation metrics to track.
- `best_metric`: Current best metric score, initialized to 0.
