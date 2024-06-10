#! /usr/bin/env python3
from pprint import pprint
from typing import Callable
from dataclasses import dataclass
import math

from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import LambdaLR

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .task import Task
from .logging import Logger


@dataclass
class Trainer:
    task:         Task
    output:       str
    train_loader: DataLoader

    valid_loader: DataLoader=None
    test_loader:  DataLoader=None
    checkpoint:   str=None
    drop_optim:   bool=False
    optim_method: Callable[..., Optimizer]=Adam
    lr:           float=1e-3
    weight_decay: float=0.
    lrf:          float=1.
    T_num:        int=1
    epochs:       int=1
    show_step:    int=50
    save_epoch:   int=1

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

        if self.checkpoint is not None:
            print('checkpoint:', self.checkpoint)
            state_dict = torch.load(self.checkpoint)
            self.model.load_state_dict(state_dict['model'], strict=False)
            if not self.drop_optim:
                self.optimizer.load_state_dict(state_dict['optim'])

        assert self.lrf <= 1.
        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=
            lambda epoch:
            (1 + math.cos(epoch / (self.epochs / self.T_num) * math.pi)) *
            0.5 * (1 - self.lrf) + self.lrf)

    def fit(
            self,
            max_train_step: int=0,
            max_test_step:  int=0,
        ):

        task      = self.task
        optimizer = self.optimizer
        logger    = Logger.create_by_output(self.output)
        print('-' * 80)
        print('Training ...')
        for epoch in range(self.epochs):
            self.model.train()
            logger.reset()

            train_loader = self.train_loader

            valid_loader = self.valid_loader
            if valid_loader is not None:
                valid_generator = iter(valid_loader)

            print('train mode:', self.model.training)
            print(f'lr={self.lr_scheduler.get_last_lr()}')
            for step, sample in enumerate(train_loader):
                task.train_on_step(epoch, step, sample, optimizer, logger)

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

                if (max_train_step > 0) and (step >= max_train_step):
                    break

            self.lr_scheduler.step()

            self.model.eval()
            test_loader = self.test_loader

            if test_loader is not None:
                print('train mode:', self.model.training)
                for step, sample in enumerate(test_loader):
                    task.test_on_step(epoch, step, sample, logger)

                    if (step + 1) % self.show_step == 0:
                        print(self._dump_progress(epoch, step, test_loader))
                        print(logger.dumps('test'))

                    if (max_test_step > 0) and (step >= max_test_step):
                        break

            task.end_on_epoch(epoch, logger)
            print(logger.dumpf())

            print('A new best model emerges:',
                task.choose_best_model(self.output, optimizer, logger))

            if (epoch + 1) % self.save_epoch == 0:
                checkpoint_filename = task.save_checkpoint(
                    epoch, self.output, optimizer)
                print('save checkpoint -> {}'.format(checkpoint_filename))

        checkpoint_filename = task.save_checkpoint(
            epoch, self.output, optimizer)
        print('finished and save checkpoint -> {}'.format(checkpoint_filename))
        logger.plot()
