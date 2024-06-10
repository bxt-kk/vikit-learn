from typing import Any, Dict, Tuple
from dataclasses import dataclass, field
import os.path

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
    key_metrics:        Tuple[str]=field(default_factory=tuple)
    best_metric:        float=0

    def sample_convert(self, sample: Any) -> Tuple[Any, Any]:

        assert not 'this is an empty func'

    def train_on_step(
            self,
            epoch:     int,
            step:      int,
            sample:    Any,
            optimizer: Optimizer,
            logger:    Logger,
        ):

        inputs, target = self.sample_convert(sample)

        model = self.model
        model.train_features(
            epoch >= self.fit_features_start)

        outputs = model(inputs)
        losses = model.calc_loss(
            outputs, *target, **self.loss_options)
        loss = losses['loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            scores = model.calc_score(
                outputs, *target, **self.score_options)

        logger.update('train', losses, scores)

    def valid_on_step(
            self,
            epoch:  int,
            step:   int,
            sample: Any,
            logger: Logger,
        ):

        inputs, target = self.sample_convert(sample)

        model = self.model

        with torch.no_grad():
            outputs = model(inputs)
            losses = model.calc_loss(
                outputs, *target, **self.loss_options)
            scores = model.calc_score(
                outputs, *target, **self.score_options)

        logger.update('valid', losses, scores)

    def test_on_step(
            self,
            epoch:  int,
            step:   int,
            sample: Any,
            logger: Logger,
        ):

        inputs, target = self.sample_convert(sample)

        model = self.model

        with torch.no_grad():
            outputs = model(inputs)
            losses = model.calc_loss(
                outputs, *target, **self.loss_options)
            scores = model.calc_score(
                outputs, *target, **self.score_options)
            if epoch >= self.metric_start_epoch:
                model.update_metric(
                    outputs, *target, **self.metric_options)

        logger.update('test', losses, scores)

    def end_on_epoch(
            self,
            epoch:  int,
            logger: Logger,
        ):

        model = self.model
        metric = dict(zip(self.key_metrics, [0.] * len(self.key_metrics)))
        if epoch >= self.metric_start_epoch:
            metric = {k: v
                for k, v in model.compute_metric().items()
                if k in metric}
        logger.update('metric', metric)

    def save_checkpoint(
            self,
            epoch:     int,
            output:    str,
            optimizer: Optimizer,
        ) -> str:

        out_path = os.path.splitext(output)[0]
        filename = f'{out_path}-{epoch}.pt'
        torch.save({
            'model': self.model.state_dict(),
            'hyperparameters': self.model.hyperparameters(),
            'optim': optimizer.state_dict()}, filename)
        return filename

    def choose_best_model(
            self,
            output:    str,
            optimizer: Optimizer,
            logger:    Logger,
        ) -> bool:

        metric = logger.compute('metric')[self.key_metrics[0]]
        if self.best_metric >= metric: return False

        self.best_metric = metric
        out_path = os.path.splitext(output)[0]
        filename = f'{out_path}-best.pt'
        torch.save({
            'model': self.model.state_dict(),
            'hyperparameters': self.model.hyperparameters(),
            'optim': optimizer.state_dict()}, filename)
        return True
