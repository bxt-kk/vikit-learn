#! /usr/bin/env python3
from pprint import pprint
from typing import Callable
from dataclasses import dataclass
import math

from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .task import Task
from .logging import Logger


@dataclass
class Trainer:
    '''This `class` is used to set general parameters for model training and starts the model training process through the `fit` method.

    Args:
        task:  The object of the training task.
        output: The path to the output file.
        train_loader: The data loader for the training set.
        valid_loader: The data loader for the validation set.
        test_loader: The data loader for the test set.
        checkpoint: The archive file for the model training parameters.
        drop_optim: Whether to drop the optimizer parameters from the archive file.
        drop_lr_scheduler: Whether to drop the learning rate parameters from the archive file.
        optim_method: The optimization method.
        lr: The learning rate.
        weight_decay: The regularization weight.
        lrf: The learning rate decay factor.
        T_num: The number of learning rate change cycles.
        grad_steps: The number of steps for gradient updates.
        epochs: The total number of training epochs.
        show_step: Set the interval for displaying the training status in steps.
        save_epoch: Set the interval for saving the model in epochs.
        logs_deque_limit: Set the limit of the logs deque.
    '''

    task:              Task
    output:            str
    train_loader:      DataLoader

    valid_loader:      DataLoader=None
    test_loader:       DataLoader=None
    checkpoint:        str | None=None
    drop_optim:        bool=False
    drop_lr_scheduler: bool=False
    optim_method:      Callable[..., Optimizer]=AdamW
    lr:                float=1e-3
    weight_decay:      float | None=None
    lrf:               float=1.
    T_num:             float=1.
    grad_steps:        int=1
    epochs:            int=1
    show_step:         int=50
    save_epoch:        int=1
    logs_deque_limit:  int=99

    def _dump_progress(
            self,
            epoch:  int,
            step:   int,
            loader: DataLoader,
        ) -> str:

        return (
            f'epoch: {epoch + 1}/{self.epochs}, '
            f'step: {step + 1}/{len(loader)}'
        )

    def initialize(self):
        print('Preparing ...')
        self.device:torch.device = self.task.device
        print('device:', self.device)

        self.model:nn.Module = self.task.model.to(self.device)

        if self.weight_decay is None:
            self.weight_decay = 0.
            if self.optim_method is AdamW:
                self.weight_decay = 0.01

        self.optimizer:Optimizer = self.optim_method(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay)
        print('optimizer:', self.optimizer)

        for kind, loader in zip(
            ['train', 'valid', 'test'],
            [self.train_loader, self.valid_loader, self.test_loader],
        ):
            if loader is None: continue
            print(f'{kind} dataset:', loader.dataset)
            pprint(dict(
                batch_size=loader.batch_size,
                num_workers=loader.num_workers,
            ))

        assert self.lrf <= 1.
        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=
            lambda epoch:
            (1 + math.cos(epoch / (self.epochs / self.T_num) * math.pi)) *
            0.5 * (1 - self.lrf) + self.lrf)

        if self.checkpoint is not None:
            print('checkpoint:', self.checkpoint)
            state_dict = torch.load(self.checkpoint, weights_only=True)
            self.model.load_state_dict(state_dict['model'], strict=False)
            if not self.drop_optim:
                self.optimizer.load_state_dict(state_dict['optim'])
            if not self.drop_lr_scheduler:
                self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])

    def fit(
            self,
            max_train_step: int=0,
            max_test_step:  int=0,
        ):

        task         = self.task
        optimizer    = self.optimizer
        lr_scheduler = self.lr_scheduler
        logger       = Logger.create_by_output(self.output)
        print('-' * 80)
        print('Training ...')
        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            logger.reset(maxlen=self.logs_deque_limit)

            train_loader = self.train_loader

            valid_loader = self.valid_loader
            if valid_loader is not None:
                valid_generator = iter(valid_loader)

            print('train mode:', self.model.training)
            print(f'lr={lr_scheduler.get_last_lr()}')
            for step, sample in enumerate(train_loader):
                task.train_on_step(epoch, step, sample, logger, self.grad_steps)

                if (step + 1) % self.show_step == 0:
                    print(self._dump_progress(epoch, step, train_loader))
                    print(logger.dumps('train'))

                if valid_loader is not None:
                    try:
                        sample = next(valid_generator)
                    except StopIteration:
                        valid_generator = iter(valid_loader)
                        sample = next(valid_generator)
                    task.valid_on_step(epoch, step, sample, logger)

                    if (step + 1) % self.show_step == 0:
                        print('valid:', logger.dumps('valid'))

                if (step + 1) % self.grad_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                if (max_train_step > 0) and ((step + 1) >= max_train_step):
                    break

            optimizer.zero_grad()
            lr_scheduler.step()

            if (epoch + 1) % self.save_epoch == 0:
                checkpoint_filename = task.save_checkpoint(
                    epoch, self.output, optimizer, lr_scheduler)
                print('save checkpoint -> {}'.format(checkpoint_filename))

            self.model.eval()
            test_loader = self.test_loader

            if test_loader is not None:
                print('train mode:', self.model.training)
                for step, sample in enumerate(test_loader):
                    task.test_on_step(epoch, step, sample, logger)

                    if (step + 1) % self.show_step == 0:
                        print(self._dump_progress(epoch, step, test_loader))
                        print(logger.dumps('test'))

                    if (max_test_step > 0) and ((step + 1) >= max_test_step):
                        break

            task.end_on_epoch(epoch, logger)
            print(logger.dumpf())

            print('A new best model emerges:', task.choose_best_model(
                self.output, optimizer, lr_scheduler, logger))

        checkpoint_filename = task.save_checkpoint(
            epoch, self.output, optimizer, lr_scheduler)
        print('finished and save checkpoint -> {}'.format(checkpoint_filename))
        logger.plot()
