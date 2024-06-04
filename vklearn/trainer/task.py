from typing import Any, Dict
from dataclasses import dataclass, field

from torch.optim import Optimizer
import torch

from .logging import Logger
from ..models.basic import Basic as Model


@dataclass
class Task:
    model:              Model
    device:             torch.device
    metric_start_epoch: int=0
    fit_features_start: int=0
    loss_options:       Dict[str, Any]=field(default_factory=dict)
    score_options:      Dict[str, Any]=field(default_factory=dict)
    metric_options:     Dict[str, Any]=field(default_factory=dict)
    best_metric:        float=0

    def setting_optimizer(
            self,
            lr:           float,
            weight_decay: float,
        ) -> Optimizer:

        assert not 'this is an empty func'

    def train_on_step(
            self,
            epoch:     int,
            step:      int,
            sample:    Any,
            optimizer: Optimizer,
            logger:    Logger,
        ):

        assert not 'this is an empty func'

    def valid_on_step(
            self,
            epoch:  int,
            step:   int,
            sample: Any,
            logger: Logger,
        ):

        assert not 'this is an empty func'

    def test_on_step(
            self,
            epoch:  int,
            step:   int,
            sample: Any,
            logger: Logger,
        ):

        assert not 'this is an empty func'

    def end_on_epoch(
            self,
            epoch:  int,
            logger: Logger,
        ):

        assert not 'this is an empty func'

    def save_checkpoint(
            self,
            epoch:     int,
            output:    str,
            optimizer: Optimizer,
        ) -> str:

        assert not 'this is an empty func'

    def choose_best_model(
            self,
            output:    str,
            optimizer: Optimizer,
            logger:    Logger,
        ) -> bool:

        assert not 'this is an empty func'
