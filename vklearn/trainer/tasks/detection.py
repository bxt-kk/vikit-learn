from typing import Any, Tuple
from dataclasses import dataclass
import os.path

from torch.optim import Optimizer

import torch

from ..logging import Logger
from ..task import Task


@dataclass
class Detection(Task):

    def sample_convert(self, sample: Any) -> Tuple[Any, Any]:
        inputs, target_labels, target_bboxes = [
            sample[i].to(self.device) for i in [0, 2, 3]]
        target_index = sample[1]
        target = target_index, target_labels, target_bboxes
        return inputs, target

    def setting_optimizer(
            self,
            lr:           float,
            weight_decay: float,
        ) -> Optimizer:

        return torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

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

        train_features = epoch >= self.fit_features_start
        outputs = model(
            inputs, train_features=train_features)
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
        metric = dict(zip(['map', 'map_50', 'map_75'], [0.] * 3))
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
